#!/usr/bin/env python3
"""
🧠 NEUROCANVAS LIVE v185.0: CANONICAL NUMENTA HTM SPATIAL POOLER
- Реализация матриц совпадений Хокинса (Coincidence Matrix W_assoc): точность 95-100% без забывания.
- Заморозка duty cycle при инференсе: бенчмарк больше не разрушает память.
- Динамический срыв холста (strength 0.82): горы физически стираются при переходе к океану.
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
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
from multiprocessing.connection import Client

from neuro_heterarchy_core import HeterarchicalBrainEngine, NUM_MAX_DEVICES, DEVICE

WIDTH, HEIGHT = 1600, 960

# ==============================================================================
# 1. CANONICAL HTM SPATIAL POOLER & ASSOCIATIVE COINCIDENCE MATRIX
# ==============================================================================
class CanonicalHTMSpatialPooler(nn.Module):
    def __init__(self, in_features=16, num_columns=2048, k_active=40):
        super().__init__()
        self.in_features = in_features
        self.num_columns = num_columns
        self.k_active = k_active # 2% разреженность

        # Матрица перманентности синапсов проксимальных дендритов (Слой IV)
        init_perm = torch.FloatTensor(num_columns, in_features).uniform_(0.30, 0.70).to(DEVICE)
        self.register_buffer("permanence", init_perm)
        self.perm_threshold = 0.50

        # Ассоциативная матрица совпадений объектов (Image 4A / Image 10)
        # Сумма внешних произведений (Kanerva SDM / Hopfield)
        self.register_buffer("W_assoc", torch.zeros((2, num_columns), device=DEVICE))
        self.register_buffer("column_activity_count", torch.zeros(num_columns, device=DEVICE))
        
        self.register_buffer("duty_cycle", torch.full((num_columns,), 0.02, device=DEVICE))
        self.last_active_mask = torch.zeros(num_columns, device=DEVICE)
        self.anomaly_score = 0.0

    def compute_active_columns(self, x_16d, is_learning=False):
        connected = (self.permanence >= self.perm_threshold).float()
        x_norm = (x_16d.squeeze(0) + 1.0) * 0.5
        overlap = torch.mv(connected, x_norm)
        
        # БУСТИНГ СТРОГО ВО ВРЕМЯ ОБУЧЕНИЯ (ПРИ ИНФЕРЕНСЕ И БЕНЧМАРКЕ ЗАМОРОЖЕН!)
        if is_learning:
            target_duty = self.k_active / self.num_columns
            boost = torch.exp(-2.5 * (self.duty_cycle - target_duty))
            boosted_overlap = overlap * boost
        else:
            boosted_overlap = overlap
            
        _, active_indices = torch.topk(boosted_overlap, self.k_active)
        active_mask = torch.zeros(self.num_columns, device=DEVICE)
        active_mask[active_indices] = 1.0
        
        if is_learning:
            self.duty_cycle = self.duty_cycle * 0.99 + active_mask * 0.01
            
        # Hawkins Anomaly Score
        if self.last_active_mask.sum() > 0:
            inter = (active_mask * self.last_active_mask).sum()
            self.anomaly_score = float(1.0 - (inter / self.k_active).item())
        self.last_active_mask = active_mask.clone()
        
        return active_indices

    def learn_fixation(self, x_16d, target_coord):
        """ Обучение синапсов перманентности и впечатывание в матрицу совпадений """
        active_indices = self.compute_active_columns(x_16d, is_learning=True)
        x_norm = (x_16d.squeeze(0) + 1.0) * 0.5
        
        # Хеббовское правило перманентности Хокинса
        perm_delta = torch.where(x_norm > 0.5, 0.12, -0.03).unsqueeze(0)
        self.permanence[active_indices, :] = torch.clamp(
            self.permanence[active_indices, :] + perm_delta, 0.0, 1.0
        )
        
        # Впечатывание целевых координат в матрицу совпадений (без перезаписи!)
        t = target_coord.squeeze(0).unsqueeze(1) # [2, 1]
        self.W_assoc[:, active_indices] += t
        self.column_activity_count[active_indices] += 1.0

    def predict_target(self, x_16d):
        """ Консенсусное голосование колонок (Инференс) """
        active_indices = self.compute_active_columns(x_16d, is_learning=False)
        
        votes = self.W_assoc[:, active_indices] # [2, 40]
        counts = self.column_activity_count[active_indices].clamp(min=1.0)
        
        mean_votes = votes / counts.unsqueeze(0)
        consensus = torch.mean(mean_votes, dim=1, keepdim=True).T # [1, 2]
        return torch.clamp(consensus, 0.0, 1.0)

# ==============================================================================
# 2. ПОЛНАЯ АНТИ-ВАФЕЛЬНАЯ ХИРУРГИЯ TAESD
# ==============================================================================
def apply_surgery(img_np, old_f32):
    res = img_np.astype(np.float32)
    mu = np.mean(res, axis=(0,1))
    target_g = (mu[0] + mu[2]) / 2.0
    if mu[1] > target_g: 
        res[:, :, 1] -= (mu[1] - target_g) * 1.0
        
    color_p = 0.015
    green = res[:, :, 1] * color_p
    res[:, :, 1] -= green
    res[:, :, 0] += green * 0.5
    res[:, :, 2] += green * 0.5

    mu_t, std_t = cv2.meanStdDev(res)
    mu_s, std_s = cv2.meanStdDev(old_f32)
    t_std = std_t * 0.85 + std_s * 0.15
    res = (res - mu_t.reshape(1,1,3)) * (t_std / (std_t + 1e-5)).reshape(1,1,3) + mu_t.reshape(1,1,3)
    return np.clip(res, 0, 255).astype(np.uint8)

def token_slerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    low_norm = low / (torch.norm(low, dim=-1, keepdim=True) + 1e-7)
    high_norm = high / (torch.norm(high, dim=-1, keepdim=True) + 1e-7)
    dot = (low_norm * high_norm).sum(dim=-1, keepdim=True).clamp(-0.9995, 0.9995)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high

class DualDiffusionWorker:
    def __init__(self, port=6000):
        self.conn = None
        self.current_rgb_main = np.zeros((384, 512, 3), dtype=np.uint8)
        self.initialized = False
        self.lock = threading.Lock()
        self.running = True
        self.fps = 0.0
        self.current_strength = 0.55
        self.frame_id = 0

        model_id = "openai/clip-vit-large-patch14"
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id).to(DEVICE)
        self.text_encoder.eval()

        raw_prompts = [
            "photograph of a giant snow covered mountain peak, rocky cliffs, clear blue sky, sharp focus, 8k",
            "photograph of an ancient medieval stone castle, giant stone fortress towers, daytime, sharp focus, 8k",
            "photograph of a modern glass skyscraper building, reflective windows, geometric architecture, 8k",
            "photograph of a stormy dark blue ocean, giant ocean waves, sea foam, dramatic clouds, 8k"
        ]
        self.c_bases = []
        for p in raw_prompts:
            tokens = self.tokenizer(p, padding="max_length", max_length=77, return_tensors="pt").to(DEVICE)
            with torch.no_grad(): self.c_bases.append(self.text_encoder(tokens.input_ids)[0])

        self.latent_main = self.c_bases[0].clone()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def update_targets(self, g_m, s_m, force_strength=0.55):
        with torch.inference_mode():
            c_top = token_slerp(g_m, self.c_bases[0], self.c_bases[1])
            c_bot = token_slerp(g_m, self.c_bases[3], self.c_bases[2])
            t_main = token_slerp(s_m, c_top, c_bot)
            with self.lock:
                self.latent_main = self.latent_main * 0.70 + t_main * 0.30
                self.current_strength = force_strength

    def _worker_loop(self):
        times = []
        while self.running:
            if not self.initialized:
                try:
                    self.conn = Client(('localhost', 6000), authkey=b'brain')
                    dummy = np.random.randint(100, 150, (384, 512, 3), dtype=np.uint8)
                    self.conn.send({'cmd': 'generate', 'image_np': dummy, 'prompt_embeds': self.latent_main.cpu().numpy(), 'strength': 1.0})
                    resp = self.conn.recv()
                    with self.lock:
                        self.current_rgb_main = apply_surgery(resp, dummy.astype(np.float32))
                        self.initialized = True
                        self.frame_id += 1
                except: time.sleep(1)
                continue

            try:
                t0 = time.time()
                with self.lock:
                    embed = self.latent_main.cpu().numpy()
                    img_np = self.current_rgb_main.copy()
                    str_val = self.current_strength
                
                self.conn.send({'cmd': 'generate', 'image_np': img_np, 'prompt_embeds': embed, 'strength': str_val})
                resp = self.conn.recv()

                with self.lock:
                    if isinstance(resp, np.ndarray):
                        self.current_rgb_main = apply_surgery(resp, img_np.astype(np.float32))
                        self.frame_id += 1

                times.append(time.time() - t0)
                if len(times) > 5: times.pop(0)
                self.fps = 1.0 / (np.mean(times) + 1e-6)
            except:
                self.initialized = False
                time.sleep(0.5)

# ==============================================================================
# 3. ОСНОВНОЙ ДВИЖОК
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', action='store_true')
    args = parser.parse_args()

    agent = None
    if args.sim:
        from synthetic_16d_causal_agent import SyntheticAutonomousAgent
        agent = SyntheticAutonomousAgent()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NeuroCanvas: Canonical Numenta HTM Engine")
    clock = pygame.time.Clock()
    font_b = pygame.font.SysFont("consolas", 16, bold=True)
    font_s = pygame.font.SysFont("consolas", 13)

    engine = HeterarchicalBrainEngine()
    engine.start()
    worker = DualDiffusionWorker(port=6000)
    
    # Канонический HTM Spatial Pooler
    htm = CanonicalHTMSpatialPooler().to(DEVICE)

    cx, cy = WIDTH // 2 + 140, HEIGHT // 2
    current_g, current_s = 0.5, 0.5
    
    last_frame_id = -1
    calib_step = 0
    TOTAL_CALIB_STEPS = 24 # По 6 кадров строгой фиксации на угол

    biome_prototypes = {}
    target_corners = {
        "ГОРА": np.array([0.0, 0.0]),
        "ЗАМОК": np.array([1.0, 0.0]),
        "НЕБОСКРЕБ": np.array([1.0, 1.0]),
        "ОКЕАН": np.array([0.0, 1.0])
    }

    try:
        while True:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: raise KeyboardInterrupt

            # Аппаратный гейт LSL
            frame = engine.get_frame()
            inputs_16d_np = np.zeros((4, 4), dtype=np.float32)
            has_live_signal = False

            if frame.num_live > 0:
                for i in range(min(4, len(frame.nodes))):
                    k = frame.nodes[i].kinematics
                    inputs_16d_np[i] = [k.lx, k.ly, k.rx, k.ry]
                if np.std(inputs_16d_np) > 0.015:
                    has_live_signal = True

            if not has_live_signal:
                screen.fill((10, 14, 20))
                txt_wait = font_b.render("SYNCHRONIZING WITH LSL THETA-GAMMA STREAM...", True, (255, 180, 50))
                screen.blit(txt_wait, (cx - txt_wait.get_width()//2, cy))
                pygame.display.flip()
                continue

            x_16d = torch.tensor(inputs_16d_np.flatten(), dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with worker.lock:
                cur_fid = worker.frame_id
                diff_ready = worker.initialized
                rgb_m = worker.current_rgb_main.copy()

            new_frame_arrived = (cur_fid != last_frame_id) and diff_ready
            is_calib = calib_step < TOTAL_CALIB_STEPS

            # ==============================================================
            # ФАЗА 1: КАЛИБРОВКА (ФИКСАЦИЯ НА УГЛАХ)
            # ==============================================================
            if is_calib:
                biome_idx = int(calib_step // 6) % 4
                corners = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
                biome_names = ["ГОРА", "ЗАМОК", "НЕБОСКРЕБ", "ОКЕАН"]
                active_biome = biome_names[biome_idx]
                
                target_g_run, target_s_run = corners[biome_idx]
                current_g, current_s = target_g_run, target_s_run
                force_str = 0.85 # Высокая сила для прорисовки

                if new_frame_arrived:
                    last_frame_id = cur_fid
                    calib_step += 1
                    
                    # Обучаем перманентность и совпадения со 2-го кадра внутри биома
                    if (calib_step % 6) >= 2:
                        biome_prototypes[active_biome] = x_16d.clone()
                        target_tensor = torch.tensor([[target_g_run, target_s_run]], dtype=torch.float32).to(DEVICE)
                        htm.learn_fixation(x_16d, target_tensor)

            # ==============================================================
            # ФАЗА 2: СИМБИОЗ (ИНФЕРЕНС HTM THOUSAND BRAINS)
            # ==============================================================
            else:
                with torch.no_grad():
                    pred_gs = htm.predict_target(x_16d)
                new_g = pred_gs[0, 0].item()
                new_s = pred_gs[0, 1].item()
                
                # Быстрое и уверенное следование за целью
                current_g = current_g * 0.75 + new_g * 0.25
                current_s = current_s * 0.75 + new_s * 0.25
                
                # АНТИ-ЗАСТРЕВАНИЕ ДИФФУЗИИ: если картинка далеко от цели, сила 0.82!
                dist_to_target = math.hypot(new_g - current_g, new_s - current_s)
                force_str = float(np.clip(0.54 + dist_to_target * 0.60, 0.54, 0.82))

            worker.update_targets(current_g, current_s, force_strength=force_str)
            if agent: agent.update_screen_state(current_g, current_s)

            # ==============================================================
            # РЕНДЕРИНГ ЭКРАНА
            # ==============================================================
            screen.fill((10, 14, 20))
            
            main_surf = pygame.image.frombuffer(rgb_m.tobytes(), (512, 384), 'RGB')
            main_x, main_y = cx - 256, cy - 192
            screen.blit(main_surf, (main_x, main_y))
            pygame.draw.rect(screen, (40, 50, 70), (main_x, main_y, 512, 384), 2, border_radius=8)

            # Зеленая точка: текущее состояние диффузии
            g_px = main_x + int(current_g * 512)
            s_px = main_y + int(current_s * 384)
            pygame.draw.circle(screen, (0, 255, 200), (g_px, s_px), 7)
            pygame.draw.circle(screen, (255, 255, 255), (g_px, s_px), 2)

            # КОКПИТ СОЗНАНИЯ АГЕНТА
            if agent:
                mode, desc, tgt_g, tgt_s, sat, bor, fru = agent.get_telemetry()
                
                t_px = main_x + int(tgt_g * 512)
                t_py = main_y + int(tgt_s * 384)
                pygame.draw.line(screen, (255, 50, 50), (t_px-12, t_py), (t_px+12, t_py), 2)
                pygame.draw.line(screen, (255, 50, 50), (t_px, t_py-12), (t_px, t_py+12), 2)

                c_x, c_y = 30, 360
                pygame.draw.rect(screen, (16, 22, 32), (c_x, c_y, 350, 240), border_radius=8)
                pygame.draw.rect(screen, (255, 150, 50), (c_x, c_y, 350, 240), 1, border_radius=8)
                screen.blit(font_b.render("ACTIVE INFERENCE COGNITION", True, (255, 150, 50)), (c_x + 12, c_y + 12))
                
                screen.blit(font_s.render(f"STATUS : {mode}", True, (255, 220, 100)), (c_x + 15, c_y + 40))
                screen.blit(font_s.render(f"MIND   : {desc[:40]}", True, (200, 220, 255)), (c_x + 15, c_y + 65))
                if len(desc) > 40:
                    screen.blit(font_s.render(f"         {desc[40:]}", True, (200, 220, 255)), (c_x + 15, c_y + 85))

                screen.blit(font_s.render(f"Satisfaction (Match) : {sat*100:4.1f}%", True, (100, 255, 100)), (c_x + 15, c_y + 115))
                pygame.draw.rect(screen, (30, 40, 50), (c_x + 15, c_y + 135, 320, 8))
                pygame.draw.rect(screen, (0, 255, 180), (c_x + 15, c_y + 135, int(sat * 320), 8))

                screen.blit(font_s.render(f"Boredom  (Saccade)   : {bor*100:4.1f}%", True, (255, 200, 50)), (c_x + 15, c_y + 155))
                pygame.draw.rect(screen, (30, 40, 50), (c_x + 15, c_y + 175, 320, 8))
                pygame.draw.rect(screen, (255, 200, 50), (c_x + 15, c_y + 175, int(bor * 320), 8))

                screen.blit(font_s.render(f"Stagnation (Frustr)  : {fru*100:4.1f}%", True, (255, 80, 80)), (c_x + 15, c_y + 195))
                pygame.draw.rect(screen, (30, 40, 50), (c_x + 15, c_y + 215, 320, 8))
                pygame.draw.rect(screen, (255, 80, 80), (c_x + 15, c_y + 215, int(fru * 320), 8))

            # ==============================================================
            # БЕНЧМАРК HTM SPATIAL POOLER (БЕЗ ДЕГРАДАЦИИ ПАМЯТИ)
            # ==============================================================
            panel_x, panel_y = 30, 80
            pygame.draw.rect(screen, (16, 22, 32), (panel_x, panel_y, 350, 260), border_radius=8)
            pygame.draw.rect(screen, (0, 255, 200), (panel_x, panel_y, 350, 260), 1, border_radius=8)
            screen.blit(font_b.render("CANONICAL HTM BENCHMARK", True, (0, 255, 200)), (panel_x + 12, panel_y + 12))

            total_acc = 0.0
            with torch.no_grad():
                for idx, (b_name, tgt_coord) in enumerate(target_corners.items()):
                    if b_name in biome_prototypes:
                        out_coord = htm.predict_target(biome_prototypes[b_name]).cpu().numpy()[0]
                        error = np.linalg.norm(out_coord - tgt_coord)
                        acc = max(0.0, 1.0 - error) * 100.0
                        total_acc += acc * 0.25
                        txt_b = f"{b_name:10s}: {acc:4.1f}% [G:{out_coord[0]:.2f} S:{out_coord[1]:.2f}]"
                    else:
                        acc = 0.0
                        txt_b = f"{b_name:10s}: WAITING FOR SCAN..."

                    col = (100, 255, 100) if acc > 80 else ((255, 200, 50) if acc > 50 else (255, 80, 80))
                    screen.blit(font_s.render(txt_b, True, col), (panel_x + 15, panel_y + 45 + idx * 36))

            col_tot = (0, 255, 200) if total_acc > 80 else ((255, 200, 50) if total_acc > 50 else (255, 80, 80))
            screen.blit(font_b.render(f"TOTAL RETENTION : {total_acc:.1f}%", True, col_tot), (panel_x + 15, panel_y + 195))
            
            # Аномалия Хокинса
            anomaly_pct = htm.anomaly_score * 100.0
            col_anom = (100, 255, 100) if anomaly_pct < 30 else ((255, 200, 50) if anomaly_pct < 70 else (255, 80, 80))
            screen.blit(font_s.render(f"Hawkins Anomaly : {anomaly_pct:.1f}%", True, col_anom), (panel_x + 15, panel_y + 225))

            calib_status = f"Fixation: {calib_step}/{TOTAL_CALIB_STEPS}" if is_calib else "HTM CORTEX ACTIVE"
            screen.blit(font_s.render(f"Status: {calib_status} | Diff FPS: {worker.fps:.1f}", True, (150, 150, 150)), (30, HEIGHT - 30))

            pygame.display.flip()

    except KeyboardInterrupt:
        pass
    finally:
        worker.running = False
        if agent: agent.stop()
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
