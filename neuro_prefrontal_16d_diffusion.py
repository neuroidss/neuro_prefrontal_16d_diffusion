#!/usr/bin/env python3
"""
🧠 NEUROCANVAS LIVE v116.0: ZERO-NOISE CONTINUOUS PREFRONTAL MANIFOLD
- 100% Искоренение сырого шума: все буферы всегда содержат чистый отрендеренный мир.
- При срыве фазы Тень непрерывно и органично морфится в новый "План В" (strength 0.75).
- Синхронизация эпох: нулевой шанс отката или стробоскопа.
- Полная хирургия Arena (apply_surgery) + Полный дебаг-HUD.
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import argparse
import time
import math
import threading
import numpy as np
import cv2
import pygame
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from multiprocessing.connection import Client

from neuro_heterarchy_core import HeterarchicalBrainEngine, NUM_MAX_DEVICES, DEVICE

WIDTH, HEIGHT = 1600, 960
PI = math.pi
TWO_PI = 2.0 * math.pi

# ==============================================================================
# 1. ТОКЕНИЗИРОВАННЫЙ SLERP В ПРОСТРАНСТВЕ CLIP (77x768)
# ==============================================================================
def token_slerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    low_norm = low / (torch.norm(low, dim=-1, keepdim=True) + 1e-7)
    high_norm = high / (torch.norm(high, dim=-1, keepdim=True) + 1e-7)
    dot = (low_norm * high_norm).sum(dim=-1, keepdim=True).clamp(-0.9995, 0.9995)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high

# ==============================================================================
# 2. ХИРУРГИЯ АРЕНЫ (ARENA_LATENT_MAIN.py)
# ==============================================================================
def apply_surgery(img_np, old_f32):
    res = img_np.astype(np.float32)
    g_channel_nerf = 1.0
    if g_channel_nerf > 0:
        mu = np.mean(res, axis=(0,1))
        target_g = (mu[0] + mu[2]) / 2.0
        if mu[1] > target_g: 
            res[:,:,1] -= (mu[1] - target_g) * g_channel_nerf
            
    color_p = 0.01
    if color_p > 0:
        green = res[:, :, 1] * color_p
        res[:, :, 1] -= green
        res[:, :, 0] += green * 0.5
        res[:, :, 2] += green * 0.5
        
    mu_t, std_t = cv2.meanStdDev(res)
    mu_s, std_s = cv2.meanStdDev(old_f32)
    
    inertia = 0.0
    t_std = std_t * (1 - inertia) + std_s * inertia
    
    res = (res - mu_t.reshape(1,1,3)) * (t_std / (std_t + 1e-5)).reshape(1,1,3) + mu_t.reshape(1,1,3)
    return np.clip(res, 0, 255).astype(np.uint8)

# ==============================================================================
# 3. 16D ТОЧНАЯ ФИЗИКА (RISING EDGE TRIGGER)
# ==============================================================================
class CUDA_16D_ExactManifold:
    def __init__(self):
        self.state_16d = torch.zeros((4, 4), device=DEVICE, dtype=torch.float32)
        self.vel_16d   = torch.zeros((4, 4), device=DEVICE, dtype=torch.float32)

        # Тень Fpz изначально ортогональна
        self.state_16d[0] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=DEVICE)
        self.state_16d[1] = torch.tensor([0.0, 0.0, 0.8, 0.0], device=DEVICE)
        self.state_16d[2] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=DEVICE)
        self.state_16d[3] = torch.tensor([PI/2, PI/2, 0.4, 0.0], device=DEVICE) 

        self.flash_timer = 0.0
        self.switch_readiness = 0.0
        self.was_surging = False
        
        self.grid_u, self.grid_v = 24, 12
        u = torch.linspace(0, TWO_PI, self.grid_u, device=DEVICE)
        v = torch.linspace(0, TWO_PI, self.grid_v, device=DEVICE)
        self.U, self.V = torch.meshgrid(u, v, indexing='ij')

    @torch.inference_mode()
    def update_physics(self, inputs_16d_gpu, dt):
        self.vel_16d = self.vel_16d * 0.85 + inputs_16d_gpu * 0.15 * 3.5
        self.state_16d = (self.state_16d + self.vel_16d * dt) % TWO_PI

        if self.flash_timer > 0:
            self.flash_timer = max(0.0, self.flash_timer - dt * 2.0)

        fpz_rx = self.state_16d[3, 2].item()
        fpz_ry = inputs_16d_gpu[3, 3].item()
        self.switch_readiness = float(np.clip(fpz_rx * 0.6 + fpz_ry * 0.4, 0.0, 1.0))

        is_surging = (fpz_ry > 0.75 or self.switch_readiness >= 0.95)
        triggered_reset = False

        # Срабатывание строго 1 раз по фронту импульса
        if is_surging and not self.was_surging and self.flash_timer <= 0.0:
            self.state_16d[0] = self.state_16d[3].clone()
            self.state_16d[1, 0] = self.state_16d[3, 1].clone() 

            self.state_16d[3, 0] = (self.state_16d[3, 0] + 2.0 * PI / 3.0) % TWO_PI
            self.state_16d[3, 1] = (self.state_16d[3, 1] + PI / 2.0) % TWO_PI
            self.state_16d[3, 2] = 0.25 
            self.state_16d[3, 3] = 0.0

            self.flash_timer = 0.7
            triggered_reset = True

        self.was_surging = is_surging
        return self.state_16d, self.switch_readiness, triggered_reset

    @torch.inference_mode()
    def compute_gyroscope_tori(self, node_idx, fixed_pitch=0.75):
        th1, ph1, th2, ph2 = self.state_16d[node_idx]
        R1, r1 = 55.0, 20.0
        x_m = (R1 + r1 * torch.cos(self.V)) * torch.cos(self.U)
        y_m = (R1 + r1 * torch.cos(self.V)) * torch.sin(self.U)
        z_m = r1 * torch.sin(self.V)

        cp, sp = math.cos(fixed_pitch), math.sin(fixed_pitch)
        y_proj = y_m * cp - z_m * sp

        bx_macro = (R1 + r1 * torch.cos(ph1)) * torch.cos(th1)
        by_macro = (R1 + r1 * torch.cos(ph1)) * torch.sin(th1)
        bz_macro = r1 * torch.sin(ph1)
        by_p = by_macro * cp - bz_macro * sp

        orbit_radius = 12.0
        mx = bx_macro.item() + orbit_radius * math.cos(th2.item())
        my = by_p.item() + orbit_radius * math.sin(ph2.item())
        return x_m.cpu().numpy(), y_proj.cpu().numpy(), (int(bx_macro.item()), int(by_p.item())), (int(mx), int(my))

# ==============================================================================
# 4. ДВОЙНОЙ ВОРКЕР (ZERO-NOISE CONTINUOUS BUFFERING)
# ==============================================================================
class DualDiffusionWorker:
    def __init__(self, port=6000):
        self.connected = False
        self.conn = None
        
        self.current_rgb_main = np.zeros((384, 512, 3), dtype=np.uint8)
        self.current_rgb_shadow = np.zeros((384, 512, 3), dtype=np.uint8)
        self.initialized = False
        
        self.epoch = 0
        
        self.lock = threading.Lock()
        self.running = True
        self.fps = 0.0

        print("[Dual SD-LCM] Загрузка CLIP Slerp-базиса...")
        model_id = "openai/clip-vit-large-patch14"
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id).to(DEVICE)
        self.text_encoder.eval()

        self.raw_prompts = [
            "photograph of a giant snow covered mountain peak, rocky cliffs, clear blue sky, sharp focus, 8k",
            "photograph of an ancient medieval stone castle, giant stone fortress towers, daytime, sharp focus, 8k",
            "photograph of a stormy dark blue ocean, giant ocean waves, sea foam, dramatic clouds, 8k",
            "photograph of a modern glass skyscraper building, reflective windows, geometric architecture, 8k"
        ]

        self.c_bases = []
        for p in self.raw_prompts:
            tokens = self.tokenizer(p, padding="max_length", max_length=77, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                self.c_bases.append(self.text_encoder(tokens.input_ids)[0])

        self.latent_main = self.c_bases[0].clone()
        self.latent_shadow = self.c_bases[1].clone()
        self.strength_main = 0.55
        self.strength_shadow = 0.55

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _connect(self):
        try:
            self.conn = Client(('localhost', 6000), authkey=b'brain')
            self.connected = True
            print("✅ Подключен к серверу! Первичный поджиг реальностей...")
            
            dummy_img = np.random.randint(100, 150, (384, 512, 3), dtype=np.uint8)
            dummy_f32 = dummy_img.astype(np.float32)
            
            # Поджиг Главного
            self.conn.send({'cmd': 'generate', 'image_np': dummy_img, 'prompt_embeds': self.latent_main.cpu().numpy(), 'strength': 1.0})
            resp_m = self.conn.recv()
            
            # Поджиг Тени
            self.conn.send({'cmd': 'generate', 'image_np': dummy_img, 'prompt_embeds': self.latent_shadow.cpu().numpy(), 'strength': 1.0})
            resp_sh = self.conn.recv()
            
            with self.lock:
                if isinstance(resp_m, np.ndarray):
                    self.current_rgb_main = apply_surgery(resp_m, dummy_f32)
                if isinstance(resp_sh, np.ndarray):
                    self.current_rgb_shadow = apply_surgery(resp_sh, dummy_f32)
                self.initialized = True
                
            print("🎉 Двойной генеративный конвейер готов (0% шума)!")
        except Exception:
            self.connected = False

    def update_targets(self, g_m, s_m, g_sh, s_sh, triggered_reset):
        with torch.inference_mode():
            c00, c10, c01, c11 = self.c_bases[0], self.c_bases[1], self.c_bases[2], self.c_bases[3]
            
            c_top_m = token_slerp(g_m, c00, c10)
            c_bot_m = token_slerp(g_m, c01, c11)
            t_main = token_slerp(s_m, c_top_m, c_bot_m)

            c_top_sh = token_slerp(g_sh, c00, c10)
            c_bot_sh = token_slerp(g_sh, c01, c11)
            t_shadow = token_slerp(s_sh, c_top_sh, c_bot_sh)

            with self.lock:
                if triggered_reset and self.initialized:
                    self.epoch += 1
                    self.current_rgb_main = self.current_rgb_shadow.copy()
                    self.latent_main = self.latent_shadow.clone()
                    # Мощная очистка при переключении, чтобы смыть старые колонны
                    self.strength_main = 0.88
                    self.latent_shadow = t_shadow.clone()
                    self.strength_shadow = 0.88
                else:
                    # Оптимальная граница: держит ~5.5 FPS и не дает запекаться полосам
                    self.strength_main = 0.54
                    self.strength_shadow = 0.54

                self.latent_main = self.latent_main * 0.75 + t_main * 0.25
                self.latent_shadow = self.latent_shadow * 0.75 + t_shadow * 0.25

    def _worker_loop(self):
        times = []
        frame_turn = 0
        while self.running:
            if not self.connected or not self.initialized:
                self._connect()
                time.sleep(1.0)
                continue

            try:
                t0 = time.time()
                
                # Приоритет 2:1 (Центр -> Центр -> Тень)
                with self.lock:
                    req_epoch = self.epoch
                    if frame_turn % 3 != 2:
                        target_task = 'main'
                        embed = self.latent_main.cpu().numpy()
                        str_val = self.strength_main
                        img_np = self.current_rgb_main.copy()
                        old_f32 = img_np.astype(np.float32)
                    else:
                        target_task = 'shadow'
                        embed = self.latent_shadow.cpu().numpy()
                        str_val = self.strength_shadow
                        img_np = self.current_rgb_shadow.copy()
                        old_f32 = img_np.astype(np.float32)

                self.conn.send({
                    'cmd': 'generate',
                    'image_np': img_np,
                    'prompt_embeds': embed,
                    'strength': float(str_val)
                })
                resp = self.conn.recv()

                with self.lock:
                    if self.epoch == req_epoch and isinstance(resp, np.ndarray):
                        if target_task == 'main':
                            self.current_rgb_main = apply_surgery(resp, old_f32)
                        else:
                            self.current_rgb_shadow = apply_surgery(resp, old_f32)

                frame_turn += 1
                dt = time.time() - t0
                times.append(dt)
                if len(times) > 5: times.pop(0)
                self.fps = 1.0 / (np.mean(times) + 1e-6)

            except Exception as e:
                self.connected = False
                self.initialized = False
                time.sleep(0.5)

# ==============================================================================
# 5. ГЛАВНЫЙ ЦИКЛ ПРИЛОЖЕНИЯ
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', action='store_true', help="Запустить честного агента")
    parser.add_argument('--debug', action='store_true', help="Включить HUD по умолчанию")
    args = parser.parse_args()

    agent = None
    if args.sim:
        try:
            from synthetic_16d_diffusion_agent import Synthetic16DDiffusionAgent
            agent = Synthetic16DDiffusionAgent()
        except: pass

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NeuroCanvas: Zero-Noise Continuous Manifold")
    clock = pygame.time.Clock()

    font_debug_b = pygame.font.SysFont("consolas", 13, bold=True)
    font_debug_sm = pygame.font.SysFont("consolas", 11)

    engine = HeterarchicalBrainEngine()
    engine.start()

    manifold = CUDA_16D_ExactManifold()
    worker = DualDiffusionWorker(port=6000)

    show_debug = args.debug
    effects_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    gyro_centers = [(190, 180), (190, HEIGHT - 180), (WIDTH - 190, 180), (WIDTH - 190, HEIGHT - 180)]
    slot_roles = ["F3: SHAPE MATRIX", "F4: OPTIC STYLE", "AFz: RULE LATTICE", "Fpz: SHADOW STREAM"]
    slot_colors = [(0, 220, 255), (255, 100, 220), (255, 200, 50), (160, 80, 255)]

    cx, cy = WIDTH // 2, HEIGHT // 2

    running = True
    try:
        while running:
            dt = max(0.001, clock.tick(60) / 1000.0)

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_F1, pygame.K_TAB, pygame.K_d):
                    show_debug = not show_debug

            frame = engine.get_frame()
            inputs_16d_np = np.zeros((4, 4), dtype=np.float32)
            for i in range(min(4, len(frame.nodes))):
                k = frame.nodes[i].kinematics
                inputs_16d_np[i] = [k.lx, k.ly, k.rx, k.ry]
            inputs_16d = torch.from_numpy(inputs_16d_np).to(DEVICE)

            state_16d, switch_readiness, triggered_reset = manifold.update_physics(inputs_16d, dt)

            g_m = (math.sin(state_16d[0, 0].item()) + 1.0) / 2.0
            s_m = (math.sin(state_16d[1, 0].item()) + 1.0) / 2.0
            g_sh = (math.sin(state_16d[3, 0].item()) + 1.0) / 2.0
            s_sh = (math.sin(state_16d[3, 1].item()) + 1.0) / 2.0

            worker.update_targets(g_m, s_m, g_sh, s_sh, triggered_reset)

            honest_match = 0.0
            tgt_info, mode_str, hold_timer = {"name": "", "g": 0.0, "s": 0.0}, "LIVE", 0.0
            if agent is not None:
                t_idx, tgt_info, mode_str, hold_timer = agent.get_status()
                dist = math.hypot(g_m - tgt_info["g"], s_m - tgt_info["s"])
                honest_match = float(np.clip(1.0 - (dist / 1.414), 0.0, 1.0))
                agent.update_feedback(honest_match)

            # ЭКРАН ЗАГРУЗКИ (Пока не сгенерированы первые кадры — 0% шума на мониторе!)
            with worker.lock:
                if not worker.initialized:
                    screen.fill((5, 7, 10))
                    loading_txt = font_debug_b.render("GENERATING INITIAL REALITIES (ZERO NOISE)...", True, (0, 255, 200))
                    screen.blit(loading_txt, (cx - loading_txt.get_width()//2, cy))
                    pygame.display.flip()
                    continue
                rgb_m = worker.current_rgb_main.copy()
                rgb_sh_small = cv2.resize(worker.current_rgb_shadow, (256, 192), interpolation=cv2.INTER_AREA)

            screen.fill((5, 7, 10))
            effects_surface.fill((0, 0, 0, 0))

            # 1. Главный Холст
            main_surf = pygame.image.frombuffer(rgb_m.tobytes(), (512, 384), 'RGB')
            main_x, main_y = cx - 256, cy - 192
            screen.blit(main_surf, (main_x, main_y))

            border_col = (40, 50, 70)
            if honest_match > 0.70:
                gold_intensity = (honest_match - 0.70) / 0.30
                border_col = (255, int(215 + 40 * gold_intensity), int(50 * (1.0 - gold_intensity)))
            pygame.draw.rect(screen, border_col, (main_x, main_y, 512, 384), 2, border_radius=8)

            # 2. Тень (Чистый мир без шума)
            shadow_surf = pygame.image.frombuffer(rgb_sh_small.tobytes(), (256, 192), 'RGB')
            shadow_x, shadow_y = main_x + 512 + 60, cy - 96
            
            shadow_alpha = int(160 + switch_readiness * 95)
            shadow_surf.set_alpha(shadow_alpha)
            screen.blit(shadow_surf, (shadow_x, shadow_y))
            pygame.draw.rect(screen, (160, 80, 255), (shadow_x, shadow_y, 256, 192), max(1, int(1 + switch_readiness * 3)), border_radius=6)

            # 3. Плазменные нити
            if switch_readiness > 0.05:
                num_filaments = int(3 + switch_readiness * 14)
                for f_i in range(num_filaments):
                    sy = main_y + int((f_i / float(num_filaments)) * 384)
                    gy = shadow_y + int((f_i / float(num_filaments)) * 192)
                    fil_col = (int(160 * switch_readiness), int(80 + 120 * switch_readiness), 255, int(150 * switch_readiness))
                    pygame.draw.line(effects_surface, fil_col, (main_x + 512, sy), (shadow_x, gy), max(1, int(switch_readiness * 4)))

            # Вспышка
            if manifold.flash_timer > 0:
                flash_alpha = int(manifold.flash_timer * 180)
                pygame.draw.rect(effects_surface, (255, 230, 100, flash_alpha), (main_x, main_y, 512, 384), border_radius=8)

            screen.blit(effects_surface, (0, 0))

            # 4. ПОЛНЫЙ ДЕБАГ HUD
            if show_debug:
                pygame.draw.rect(screen, (14, 18, 26), (20, 15, WIDTH - 40, 65), border_radius=6)
                pygame.draw.rect(screen, (0, 255, 200), (20, 15, WIDTH - 40, 65), 1, border_radius=6)
                
                debug_txt = f"[DEBUG F1] Nodes: {frame.num_live}/4 | DUAL SD-LCM: {worker.fps:.1f} FPS | Match: {honest_match*100:.1f}% | Readiness: {switch_readiness*100:.0f}%"
                screen.blit(font_debug_b.render(debug_txt, True, (0, 255, 200)), (35, 22))

                if agent is not None:
                    goal_txt = f"Goal: {tgt_info['name']} [{mode_str}] | g={g_m:.2f} s={s_m:.2f} -> Target: g={tgt_info['g']:.2f} s={tgt_info['s']:.2f} | Hold: {hold_timer:.1f}s"
                    screen.blit(font_debug_sm.render(goal_txt, True, (255, 220, 100)), (35, 42))

                state_np = state_16d.cpu().numpy()
                for i in range(4):
                    gx, gy = gyro_centers[i]
                    col = slot_colors[i]
                    panel_w, panel_h = 260, 220
                    px, py = gx - panel_w // 2, gy - panel_h // 2
                    
                    pygame.draw.rect(screen, (10, 14, 20), (px, py, panel_w, panel_h), border_radius=8)
                    pygame.draw.rect(screen, col, (px, py, panel_w, panel_h), 1, border_radius=8)
                    screen.blit(font_debug_b.render(slot_roles[i], True, col), (px + 10, py + 8))

                    tx_r, ty_p, b_macro, b_micro = manifold.compute_gyroscope_tori(i)
                    for u_i in range(0, manifold.grid_u, 3):
                        pts_t = [(int(gx + tx_r[u_i, v_i]), int(gy - ty_p[u_i, v_i] - 10)) for v_i in range(manifold.grid_v)]
                        pygame.draw.lines(screen, (28, 38, 52), True, pts_t, 1)

                    bm_x, bm_y = gx + b_macro[0], gy - b_macro[1] - 10
                    pygame.draw.circle(screen, col, (bm_x, bm_y), 7)
                    pygame.draw.circle(screen, (255, 255, 255), (bm_x, bm_y), 2)

                    mm_x, mm_y = gx + b_micro[0], gy - b_micro[1] - 10
                    pygame.draw.line(screen, (255, 80, 180), (bm_x, bm_y), (mm_x, mm_y), 1)
                    pygame.draw.circle(screen, (255, 80, 180), (mm_x, mm_y), 4)

                    s = state_np[i]
                    screen.blit(font_debug_sm.render(f"θ1(lx): {s[0]:.2f} | φ1(ly): {s[1]:.2f}", True, (200, 220, 240)), (px + 10, py + panel_h - 36))
                    screen.blit(font_debug_sm.render(f"θ2(rx): {s[2]:.2f} | φ2(ry): {s[3]:.2f}", True, (255, 120, 180)), (px + 10, py + panel_h - 20))

            pygame.display.flip()

    finally:
        worker.running = False
        if agent is not None: agent.stop()
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
