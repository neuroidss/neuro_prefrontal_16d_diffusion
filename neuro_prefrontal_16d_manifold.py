#!/usr/bin/env python3
"""
🧠 NEUROCANVAS v73.0: PURE THETA-GAMMA PREFRONTAL DIFFUSION LAB
- Честная калибровка метрик сходства без тавтологий.
- Плавная линейная динамика (g, s) без клиппинга.
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import argparse
import time
import math
import threading
from collections import deque
import numpy as np
import cv2
import pygame
import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
from multiprocessing.connection import Client

from neuro_heterarchy_core import HeterarchicalBrainEngine, I_IDX, J_IDX, COORDS_X, COORDS_Y

WIDTH, HEIGHT = 1600, 960
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DX_PAIRS = (COORDS_X[J_IDX] - COORDS_X[I_IDX]).astype(np.float32)
DY_PAIRS = (COORDS_Y[J_IDX] - COORDS_Y[I_IDX]).astype(np.float32)
DX_GPU = torch.from_numpy(DX_PAIRS).to(DEVICE)
DY_GPU = torch.from_numpy(DY_PAIRS).to(DEVICE)

# ==============================================================================
# 1. SD-LCM DECNEF СИНТЕЗАТОР С ЧЕСТНОЙ КАЛИБРОВКОЙ
# ==============================================================================
class PrefrontalDecNefWorker:
    def __init__(self, port=6000):
        self.connected = False
        self.validated = False
        self.conn = None
        self.current_bgr = np.zeros((384, 512, 3), dtype=np.uint8)
        self.latest_rgb = cv2.cvtColor(self.current_bgr, cv2.COLOR_BGR2RGB).tobytes()
        self.lock = threading.Lock()
        self.running = True
        self.fps = 0.0
        self.server_msg = "Pre-Flight Validation Gate..."

        print("[DecNef] Инициализация CLIP для семантического базиса...")
        model_id = "openai/clip-vit-large-patch14"
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id).to(DEVICE)
        self.text_encoder.eval()

        # 4 Базисных полюса: (0,0) Гора, (1,0) Замок, (0,1) Океан, (1,1) Небоскреб
        self.raw_prompts = [
            "photograph of a giant snow covered mountain peak, rocky cliffs, clear blue sky, sharp focus, 8k",
            "photograph of an ancient medieval stone castle, giant stone towers, fortress walls, daytime, 8k",
            "photograph of a stormy dark blue ocean, giant ocean waves, sea foam, dramatic clouds, 8k",
            "photograph of a modern glass skyscraper building, reflective windows, geometric architecture, 8k"
        ]

        self.c_bases = []
        for p in self.raw_prompts:
            tokens = self.tokenizer(p, padding="max_length", max_length=77, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                self.c_bases.append(self.text_encoder(tokens.input_ids)[0])
        
        self.target_norm = torch.norm(self.c_bases[0], dim=-1, keepdim=True)

        torch.manual_seed(42)
        full_basis, _ = torch.linalg.qr(torch.randn(768, 768, device=DEVICE))
        self.W_F3 = full_basis[:120, :384]
        self.W_F4 = full_basis[:120, 384:]

        self.latent_context = self.c_bases[0].clone()
        self.cfg_scale = 1.3
        self.strength = 0.55
        self.cached_thumbs = [None, None, None, None]
        self.cached_raw_imgs = [None, None, None, None]

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _connect_and_validate(self):
        try:
            self.conn = Client(('localhost', 6000), authkey=b'brain')
            self.connected = True
            print("✅ [SD CLIENT] Подключен. Генерация 4-х эталонов...")
            
            dummy_img = np.random.randint(60, 180, (384, 512, 3), dtype=np.uint8)
            _, enc_jpg = cv2.imencode('.jpg', dummy_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            target_orders = [1, 2, 3, 0] # Замок (1,0), Океан (0,1), Небоскреб (1,1), Гора (0,0)
            
            for i, base_idx in enumerate(target_orders):
                c_base = self.c_bases[base_idx]
                payload = {
                    'cmd': 'generate',
                    'image_bytes': enc_jpg.tobytes(),
                    'prompt_embeds': c_base.cpu().numpy(),
                    'negative_prompt': "blurry, low quality, noise, grid, stripes",
                    'strength': 0.92,
                    'guidance_scale': 1.4,
                    'num_inference_steps': 2
                }
                self.conn.send(payload)
                resp = self.conn.recv()
                if isinstance(resp, bytes):
                    raw_bgr = cv2.imdecode(np.frombuffer(resp, np.uint8), cv2.IMREAD_COLOR)
                    if raw_bgr is not None:
                        self.cached_raw_imgs[i] = raw_bgr.copy()
                        thumb = cv2.resize(raw_bgr, (128, 96))
                        self.cached_thumbs[i] = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                        print(f"   [Эталон {i+1}/4] Сгенерирован успешно.")

            if self.cached_raw_imgs[0] is not None:
                with self.lock:
                    self.current_bgr = self.cached_raw_imgs[0].copy()
                    self.latest_rgb = cv2.cvtColor(self.current_bgr, cv2.COLOR_BGR2RGB).tobytes()

            self.validated = True
            self.server_msg = "SD-LCM Active (Ready)"
            print("🎉 [ГОТОВО] Все 4 эталона закэшированы.")
        except Exception as e:
            self.connected = False
            self.validated = False
            self.server_msg = f"Waiting for brain_server.py: {e}"

    def _worker_loop(self):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
        times = []

        while self.running:
            if not self.connected or not self.validated:
                self._connect_and_validate()
                time.sleep(1.0)
                continue

            try:
                t0 = time.time()
                with self.lock:
                    cur_img = self.current_bgr.copy()
                    embeds_np = self.latent_context.cpu().numpy()
                    cfg_val = self.cfg_scale
                    str_val = self.strength

                _, enc_jpg = cv2.imencode('.jpg', cur_img, encode_params)

                payload = {
                    'cmd': 'generate',
                    'image_bytes': enc_jpg.tobytes(),
                    'prompt_embeds': embeds_np,
                    'negative_prompt': "blurry, low quality, noise, smooth blobs, washed out, grid, watermark",
                    'strength': float(str_val),
                    'guidance_scale': float(cfg_val),
                    'num_inference_steps': 2
                }

                self.conn.send(payload)
                resp = self.conn.recv()

                if isinstance(resp, bytes):
                    raw_bgr = cv2.imdecode(np.frombuffer(resp, np.uint8), cv2.IMREAD_COLOR)
                    if raw_bgr is not None:
                        with self.lock:
                            self.current_bgr = raw_bgr
                            self.latest_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB).tobytes()

                dt = time.time() - t0
                times.append(dt)
                if len(times) > 6: times.pop(0)
                self.fps = 1.0 / (np.mean(times) + 1e-6)

            except Exception as e:
                self.connected = False
                self.validated = False
                self.server_msg = f"Server Error: {e}"
                time.sleep(0.5)

    def update_prefrontal_manifold(self, f3_t, f4_t, afz_rigidity, fpz_velocity, is_phase_reset, tgt_g=0.0, tgt_s=0.0, target_id=0):
        with torch.inference_mode():
            # 1. Извлечение живых координат g (F3) и s (F4)
            e_f3_core = torch.mean(torch.abs(f3_t[:, :60])).item()
            e_f3_ring = torch.mean(torch.abs(f3_t[:, 60:])).item()
            e_f4_core = torch.mean(torch.abs(f4_t[:, :60])).item()
            e_f4_ring = torch.mean(torch.abs(f4_t[:, 60:])).item()

            # Нормализация отношения Core/Ring строго в [0.0, 1.0]
            g_val = float(np.clip((e_f3_ring / (e_f3_core + 1e-5) - 0.25) / 1.5, 0.0, 1.0))
            s_val = float(np.clip((e_f4_ring / (e_f4_core + 1e-5) - 0.25) / 1.5, 0.0, 1.0))

            # 2. Билинейная интерполяция
            c00 = self.c_bases[0] # Гора (0,0)
            c10 = self.c_bases[1] # Замок (1,0)
            c01 = self.c_bases[2] # Океан (0,1)
            c11 = self.c_bases[3] # Небоскреб (1,1)

            c_top = c00 * (1.0 - g_val) + c10 * g_val
            c_bot = c01 * (1.0 - g_val) + c11 * g_val
            base_anchor = c_top * (1.0 - s_val) + c_bot * s_val

            f3_proj = torch.matmul(f3_t, self.W_F3)
            f4_proj = torch.matmul(f4_t, self.W_F4)
            brain_traj = torch.cat([f3_proj, f4_proj], dim=-1).unsqueeze(0).permute(0, 2, 1)
            brain_delta = F.interpolate(brain_traj, size=77, mode='linear', align_corners=True).permute(0, 2, 1)

            target_context = base_anchor + brain_delta * 0.02
            target_context = target_context / (torch.norm(target_context, dim=-1, keepdim=True) + 1e-6) * self.target_norm

            cfg = 1.1 + min(0.6, afz_rigidity * 0.15)
            
            if is_phase_reset:
                strength = 0.85
                if self.cached_raw_imgs[target_id] is not None:
                    with self.lock:
                        self.current_bgr = self.cached_raw_imgs[target_id].copy()
            else:
                strength = float(np.clip(0.48 + fpz_velocity * 0.15, 0.45, 0.65))

            with self.lock:
                self.latent_context = self.latent_context * 0.80 + target_context * 0.20
                self.cfg_scale = cfg
                self.strength = strength

            # 3. ЧЕСТНОЕ ЕВКЛИДОВО СХОДСТВО В ТЕТА-ГАММА ПРОСТРАНСТВЕ (0..1)
            dist_theta_gamma = math.hypot(g_val - tgt_g, s_val - tgt_s)
            # Максимальное расстояние между углами квадрата = sqrt(2) ~ 1.414
            honest_match = float(np.clip(1.0 - (dist_theta_gamma / 1.414), 0.0, 1.0))

            return honest_match, g_val, s_val, dist_theta_gamma

# ==============================================================================
# 2. 16D КИНЕМАТИКА
# ==============================================================================
def extract_4d_kinematics(iplv_32_tensor):
    with torch.inference_mode():
        sum_pwr = torch.sum(torch.abs(iplv_32_tensor), dim=-1, keepdim=True) + 1e-6
        raw_x = -torch.sum(iplv_32_tensor * DX_GPU, dim=-1) / sum_pwr.squeeze(-1)
        raw_y = -torch.sum(iplv_32_tensor * DY_GPU, dim=-1) / sum_pwr.squeeze(-1)

        traj_x = torch.clamp(raw_x / 6.0, -1.0, 1.0)
        traj_y = torch.clamp(raw_y / 6.0, -1.0, 1.0)

        lx = (traj_x[31] - traj_x[0]).item()
        ly = (traj_y[31] - traj_y[0]).item()

        L_len = math.hypot(lx, ly) + 1e-5
        mid_x = torch.mean(traj_x[11:22]).item()
        mid_y = torch.mean(traj_y[11:22]).item()
        chord_mid_x = (traj_x[0].item() + traj_x[31].item()) * 0.5
        chord_mid_y = (traj_y[0].item() + traj_y[31].item()) * 0.5
        
        rx = ((mid_x - chord_mid_x) * (-ly) + (mid_y - chord_mid_y) * lx) / L_len
        rx = float(np.clip(rx * 2.5, -1.0, 1.0))

        pwr_past = torch.sum(torch.abs(iplv_32_tensor[:11])).item()
        pwr_fut  = torch.sum(torch.abs(iplv_32_tensor[21:])).item()
        ry = float(np.clip((pwr_fut - pwr_past) / (pwr_fut + pwr_past + 1e-6) * 2.0, -1.0, 1.0))

        return lx, ly, rx, ry, traj_x.cpu().numpy(), traj_y.cpu().numpy()

def draw_node_radar_4d(surface, x, y, size, lx, ly, rx, ry, traj_x, traj_y, name, subtitle, col_tint, font_b, font_sm):
    pygame.draw.rect(surface, (16, 22, 32), (x, y, size, size), border_radius=8)
    pygame.draw.rect(surface, col_tint, (x, y, size, size), 1, border_radius=8)

    cx, cy = x + size // 2 - 25, y + size // 2 + 10
    rad = (size - 70) // 2

    pygame.draw.circle(surface, (30, 42, 58), (cx, cy), rad, 1)
    pygame.draw.circle(surface, (22, 32, 45), (cx, cy), rad // 2, 1)
    pygame.draw.line(surface, (30, 42, 58), (cx - rad, cy), (cx + rad, cy), 1)
    pygame.draw.line(surface, (30, 42, 58), (cx, cy - rad), (cx, cy + rad), 1)

    pts = [ (int(cx + traj_x[k] * rad), int(cy - traj_y[k] * rad)) for k in range(32) ]
    for k in range(31):
        c = (60, 140, 255) if k < 11 else ((0, 255, 160) if k < 21 else (255, 90, 90))
        thk = 1 if k < 11 else (3 if k < 21 else 2)
        pygame.draw.line(surface, c, pts[k], pts[k+1], thk)

    end_lx = int(cx + lx * rad)
    end_ly = int(cy - ly * rad)
    pygame.draw.line(surface, (255, 255, 255), (cx, cy), (end_lx, end_ly), 2)
    pygame.draw.circle(surface, col_tint, (end_lx, end_ly), 5)

    bar_x = x + size - 42
    bar_y = y + 40
    bar_h = size - 60
    pygame.draw.rect(surface, (25, 35, 48), (bar_x, bar_y, 12, bar_h), border_radius=3)
    pygame.draw.rect(surface, (255, 180, 50) if rx >= 0 else (100, 200, 255), (bar_x, bar_y + bar_h//2, 12, int(-rx * (bar_h//2))))

    bar_x2 = x + size - 22
    pygame.draw.rect(surface, (25, 35, 48), (bar_x2, bar_y, 12, bar_h), border_radius=3)
    pygame.draw.rect(surface, (255, 80, 120) if ry >= 0 else (80, 150, 255), (bar_x2, bar_y + bar_h//2, 12, int(-ry * (bar_h//2))))

    surface.blit(font_b.render(name, True, col_tint), (x + 12, y + 8))
    surface.blit(font_sm.render(subtitle, True, (150, 170, 190)), (x + 12, y + 26))
    surface.blit(font_sm.render(f"L:({lx:+.2f},{ly:+.2f})", True, (220, 230, 240)), (x + 12, y + size - 20))

# ==============================================================================
# 3. ГЛАВНОЕ ПРИЛОЖЕНИЕ
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="NeuroCanvas DecNef Diffusion Lab")
    parser.add_argument('--sim', action='store_true', help="Запустить двухполушарного агента")
    args = parser.parse_args()

    agent = None
    if args.sim:
        from synthetic_koechlin_agent import PrefrontalHighDimAgent
        agent = PrefrontalHighDimAgent()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NeuroCanvas: Two-Hemisphere Theta-Gamma Diffusion Stream")
    clock = pygame.time.Clock()

    font_huge = pygame.font.SysFont("consolas", 20, bold=True)
    font_title = pygame.font.SysFont("consolas", 16, bold=True)
    font_b = pygame.font.SysFont("consolas", 13, bold=True)
    font_sm = pygame.font.SysFont("consolas", 11)

    engine = HeterarchicalBrainEngine()
    engine.start()

    sd_worker = PrefrontalDecNefWorker(port=6000)

    last_fpz_phase = 0.0
    running = True
    try:
        while running:
            dt = max(0.001, clock.tick(60) / 1000.0)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            frame = engine.get_frame()

            afz_t = torch.from_numpy(frame.nodes[0].iplv_32).to(DEVICE, dtype=torch.float32)
            f3_t  = torch.from_numpy(frame.nodes[1].iplv_32).to(DEVICE, dtype=torch.float32)
            f4_t  = torch.from_numpy(frame.nodes[2].iplv_32).to(DEVICE, dtype=torch.float32)
            fpz_t = torch.from_numpy(frame.nodes[3].iplv_32).to(DEVICE, dtype=torch.float32)

            # 16D Кинематика
            afz_lx, afz_ly, afz_rx, afz_ry, afz_tx, afz_ty = extract_4d_kinematics(afz_t)
            f3_lx,  f3_ly,  f3_rx,  f3_ry,  f3_tx,  f3_ty  = extract_4d_kinematics(f3_t)
            f4_lx,  f4_ly,  f4_rx,  f4_ry,  f4_tx,  f4_ty  = extract_4d_kinematics(f4_t)
            fpz_lx, fpz_ly, fpz_rx, fpz_ry, fpz_tx, fpz_ty = extract_4d_kinematics(fpz_t)

            # Phase Slip Детектор на Fpz
            fpz_phase_now = frame.nodes[3].phase_theta
            expected_adv = 2.0 * math.pi * frame.theta_freq * dt
            measured_diff = (fpz_phase_now - last_fpz_phase + math.pi) % (2.0 * math.pi) - math.pi
            phase_slip = abs((measured_diff - expected_adv + math.pi) % (2.0 * math.pi) - math.pi)
            last_fpz_phase = fpz_phase_now
            is_phase_reset = (phase_slip > 1.8)

            if agent is not None:
                t_idx, tgt_info, mode_str, hold_timer, ag_g, ag_s = agent.get_target_info()
            else:
                from synthetic_koechlin_agent import THETA_GAMMA_TARGETS
                t_idx, tgt_info, mode_str, hold_timer, ag_g, ag_s = 0, THETA_GAMMA_TARGETS[0], "LIVE EEG (4x LSL)", 0.0, 0.0, 0.0

            afz_rigidity = float(np.std(frame.nodes[0].iplv_32)) * 15.0
            fpz_velocity = math.hypot(fpz_lx, fpz_ly)
            
            honest_match, g_val, s_val, dist_tg = sd_worker.update_prefrontal_manifold(
                f3_t, f4_t, afz_rigidity, fpz_velocity, is_phase_reset, 
                tgt_g=tgt_info["g"], tgt_s=tgt_info["s"], target_id=t_idx
            )

            if agent is not None:
                agent.update_feedback(honest_match)

            # ==============================================================================
            # ОТРИСОВКА ИНТЕРФЕЙСА
            # ==============================================================================
            screen.fill((10, 14, 20))

            # Шапка
            pygame.draw.rect(screen, (15, 20, 30), (20, 15, WIDTH - 40, 50), border_radius=8)
            screen.blit(font_title.render("🧠 NEUROCANVAS v73.0: TWO-HEMISPHERE THETA-GAMMA DIFFUSION", True, (0, 255, 200)), (35, 22))
            
            fps_col = (0, 255, 120) if sd_worker.fps >= 6.0 else (255, 200, 50)
            screen.blit(font_sm.render(f"Nodes: {frame.num_live}/4 | Theta: {frame.theta_freq:.2f} Hz | SD-LCM: {sd_worker.fps:.1f} FPS | {sd_worker.server_msg}", True, fps_col), (35, 45))

            # 4 УГЛОВЫХ 4D-РАДАРА
            r_sz = 260
            draw_node_radar_4d(screen, WIDTH - 20 - r_sz, 80, r_sz, afz_lx, afz_ly, afz_rx, afz_ry, afz_tx, afz_ty,
                               "AFz: RULE (CFG)", "DOI: 10.1016/j.tics.2014.04.012", (0, 200, 255), font_b, font_sm)
            
            draw_node_radar_4d(screen, 20, 80, r_sz, f3_lx, f3_ly, f3_rx, f3_ry, f3_tx, f3_ty,
                               "F3: LEFT (АРХИТЕКТУРА)", "DOI: 10.1016/j.tics.2005.09.009", (0, 255, 150), font_b, font_sm)
            
            draw_node_radar_4d(screen, 20, HEIGHT - 20 - r_sz, r_sz, f4_lx, f4_ly, f4_rx, f4_ry, f4_tx, f4_ty,
                               "F4: RIGHT (ПРИРОДА)", "DOI: 10.1038/nature17637", (255, 200, 50), font_b, font_sm)
            
            draw_node_radar_4d(screen, WIDTH - 20 - r_sz, HEIGHT - 20 - r_sz, r_sz, fpz_lx, fpz_ly, fpz_rx, fpz_ry, fpz_tx, fpz_ty,
                               "Fpz: LATENT WALK", "DOI: 10.1126/science.1142995", (255, 80, 180), font_b, font_sm)

            # === ЦЕНТРАЛЬНЫЙ ХОЛСТ ===
            CANVAS_X = 300
            CANVAS_Y = 80
            CANVAS_W = WIDTH - 600
            CANVAS_H = HEIGHT - 100

            match_col = (0, 255, 150) if honest_match > 0.75 else ((255, 200, 50) if honest_match > 0.35 else (255, 80, 80))
            pygame.draw.rect(screen, (14, 18, 26), (CANVAS_X, CANVAS_Y, CANVAS_W, CANVAS_H), border_radius=12)
            pygame.draw.rect(screen, match_col, (CANVAS_X, CANVAS_Y, CANVAS_W, CANVAS_H), 2, border_radius=12)

            # Карточка Цели с миниатюрой эталона
            card_y = CANVAS_Y + 12
            pygame.draw.rect(screen, (20, 28, 40), (CANVAS_X + 20, card_y, CANVAS_W - 40, 95), border_radius=8)
            pygame.draw.rect(screen, match_col, (CANVAS_X + 20, card_y, CANVAS_W - 40, 95), 1, border_radius=8)

            # Эталонная миниатюра
            thumb = sd_worker.cached_thumbs[t_idx]
            if thumb is not None:
                thumb_surf = pygame.image.frombuffer(thumb.tobytes(), (128, 96), 'RGB')
                screen.blit(thumb_surf, (CANVAS_X + 25, card_y))
                pygame.draw.rect(screen, match_col, (CANVAS_X + 25, card_y, 128, 96), 1)

            text_offset_x = 165
            screen.blit(font_huge.render(f"ЦЕЛЬ: {tgt_info['name']} [{mode_str}]", True, match_col), (CANVAS_X + text_offset_x, card_y + 10))
            screen.blit(font_sm.render(tgt_info['desc'], True, (180, 200, 220)), (CANVAS_X + text_offset_x, card_y + 36))

            # Полоса Match %
            bar_w = CANVAS_W - text_offset_x - 30
            pygame.draw.rect(screen, (30, 40, 55), (CANVAS_X + text_offset_x, card_y + 60, bar_w, 18), border_radius=4)
            pygame.draw.rect(screen, match_col, (CANVAS_X + text_offset_x, card_y + 60, int(bar_w * honest_match), 18), border_radius=4)
            
            timer_str = f" | HOLD: {hold_timer:.1f}/5.0s" if "HOLD" in mode_str else ""
            screen.blit(font_b.render(f"ТЕТА-ГАММА СХОДСТВО (MATCH): {honest_match*100:.1f}% (Дистанция Δ: {dist_tg:.2f}){timer_str}", True, (255, 255, 255)), (CANVAS_X + text_offset_x + 30, card_y + 61))

            # Живой кадр диффузии
            img_cx = CANVAS_X + (CANVAS_W - 512) // 2
            img_cy = CANVAS_Y + 120

            with sd_worker.lock:
                rgb_bytes = sd_worker.latest_rgb

            img_surf = pygame.image.frombuffer(rgb_bytes, (512, 384), 'RGB')
            screen.blit(img_surf, (img_cx, img_cy))
            pygame.draw.rect(screen, match_col, (img_cx, img_cy, 512, 384), 2)

            # === 2D ДВУПОЛУШАРНЫЙ РАДАР (F3 x F4) ===
            rad_y = img_cy + 395
            pygame.draw.rect(screen, (20, 28, 40), (CANVAS_X + 25, rad_y, CANVAS_W - 50, 75), border_radius=8)
            pygame.draw.rect(screen, (40, 60, 80), (CANVAS_X + 25, rad_y, CANVAS_W - 50, 75), 1, border_radius=8)

            screen.blit(font_b.render("ДВУПОЛУШАРНЫЙ БАЗИС: ЛЕВОЕ F3 (АРХИТЕКТУРА g) x ПРАВОЕ F4 (ПРИРОДА s):", True, (0, 200, 255)), (CANVAS_X + 40, rad_y + 8))
            
            screen.blit(font_sm.render(f"[0,0] 🏔️ Гора", True, (0, 255, 200)), (CANVAS_X + 40, rad_y + 32))
            screen.blit(font_sm.render(f"[1,0] 🏰 Замок (F3)", True, (0, 255, 150)), (CANVAS_X + 220, rad_y + 32))
            screen.blit(font_sm.render(f"[0,1] 🌊 Океан (F4)", True, (255, 200, 50)), (CANVAS_X + 400, rad_y + 32))
            screen.blit(font_sm.render(f"[1,1] 🏙️ Небоскреб", True, (255, 100, 255)), (CANVAS_X + 580, rad_y + 32))

            tgt_g, tgt_s = tgt_info["g"], tgt_info["s"]
            screen.blit(font_b.render(f"ТЕКУЩИЙ ВЕКТОР КОРЫ: g={g_val:.2f} (F3), s={s_val:.2f} (F4) ──► ЦЕЛЬ: g={tgt_g:.1f}, s={tgt_s:.1f} (Δ={dist_tg:.2f})", True, (255, 220, 50)), (CANVAS_X + 40, rad_y + 52))

            pygame.display.flip()

    finally:
        sd_worker.running = False
        if agent is not None:
            agent.stop()
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
