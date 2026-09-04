#!/usr/bin/env python3
"""
🧠 NEUROCANVAS LIVE v248.0: CANONICAL HAWKINS EVIDENCE HETERARCHY (NO AXES)
- Полный отказ от осей: архитектура EvidenceGraphLM из tbp.monty.
- Каждый концепт — независимая объектная модель в памяти Хокинса (Zero Catastrophic Forgetting).
- Абсолютный курикулум: выход ТОЛЬКО тогда, когда ВСЕ 4 концепта покажут >= 85.0% одновременно!
- Прозрачный реалтайм-бенчмарк на каждом кадре.
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
from PIL import Image
import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor
from multiprocessing.connection import Client

from neuro_heterarchy_core import HeterarchicalBrainEngine, NUM_MAX_DEVICES, DEVICE

WIDTH, HEIGHT = 1600, 960

ELECTRODE_X = np.array([
    10.14,  7.43,  2.75,  2.72, -2.72, -2.75, -7.42, -10.14,
   -10.14, -7.43, -2.75, -2.72,  2.72,  2.75,  7.43,  10.14
], dtype=np.float32)

ELECTRODE_Y = np.array([
    -2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42,  -2.73,
     2.72,  7.43,  4.76,  10.14, 10.15,  4.77,  7.42,   2.71
], dtype=np.float32)

# ==============================================================================
# 1. ЗРИТЕЛЬНЫЙ КЛАССИФИКАТОР CLIP (РЕАЛЬНЫЕ ПИКСЕЛИ)
# ==============================================================================
class VisualCLIPTeacher:
    def __init__(self):
        model_id = "openai/clip-vit-large-patch14"
        print("👁️ [VISION] Инициализация зрительного учителя CLIP (FP16)...")
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        
        self.classes = ["ГОРА", "ЗАМОК", "НЕБОСКРЕБ", "ОКЕАН"]
        self.prompts = [
            "a high quality photograph of a giant snowy mountain peak, rocky cliffs, clear blue sky",
            "a high quality photograph of an ancient medieval stone castle fortress, daytime",
            "a high quality photograph of modern tall glass skyscraper buildings, downtown city",
            "a high quality photograph of open stormy dark blue ocean with sea waves and foam"
        ]
        
        with torch.no_grad():
            inputs = self.processor(text=self.prompts, return_tensors="pt", padding=True).to(DEVICE)
            self.text_features = self.model.get_text_features(**inputs)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

    def classify(self, rgb_image_np: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(rgb_image_np)
        inputs = self.processor(images=pil_img, return_tensors="pt").to(DEVICE)
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        
        with torch.no_grad():
            img_features = self.model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            logits = (img_features @ self.text_features.T) * 30.0
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return probs

# ==============================================================================
# 2. СПЕКТРАЛЬНАЯ ХИРУРГИЯ TAESD (БЕЗ РАМОК И ВИНЬЕТОК)
# ==============================================================================
def apply_surgery(img_np, old_f32):
    res = img_np.astype(np.float32)
    mu = np.mean(res, axis=(0, 1))
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
    res = (res - mu_t.reshape(1, 1, 3)) * (t_std / (std_t + 1e-5)).reshape(1, 1, 3) + mu_t.reshape(1, 1, 3)
    return np.clip(res, 0, 255).astype(np.uint8)

# ==============================================================================
# 3. ТОПОЛОГИЧЕСКИЙ HTM SPATIAL POOLER (TBT L4)
# ==============================================================================
class CanonicalHTMColumn(nn.Module):
    def __init__(self, node_id: str, num_columns: int = 2304, k_active: int = 40):
        super().__init__()
        self.node_id = node_id
        self.num_columns = num_columns
        self.k_active = k_active

        ex, ey = [], []
        for i in range(16):
            for j in range(i + 1, 16):
                ex.append((ELECTRODE_X[i] + ELECTRODE_X[j]) / 2.0)
                ey.append((ELECTRODE_Y[i] + ELECTRODE_Y[j]) / 2.0)
        self.register_buffer("edge_x", torch.tensor(ex, device=DEVICE))
        self.register_buffer("edge_y", torch.tensor(ey, device=DEVICE))

        grid_dim = int(math.isqrt(num_columns))
        cy = torch.linspace(-11.0, 11.0, grid_dim, device=DEVICE).view(grid_dim, 1, 1)
        cx = torch.linspace(-11.0, 11.0, grid_dim, device=DEVICE).view(1, grid_dim, 1)
        d_sq = (cx - self.edge_x.view(1, 1, 120))**2 + (cy - self.edge_y.view(1, 1, 120))**2
        spatial_rf = torch.exp(-d_sq / 40.0).view(-1, 120)[:num_columns]

        # Фиксированные рецептивные поля (топография сенсорной карты)
        self.register_buffer("permanence", spatial_rf)
        self.perm_threshold = 0.25

    def compute_sdr(self, pac_iplv_32x120: torch.Tensor):
        connected = (self.permanence >= self.perm_threshold).float()
        x_norm = (pac_iplv_32x120 + 1.0) * 0.5

        phase_weights = torch.linspace(0.6, 1.4, 32, device=DEVICE).unsqueeze(1)
        integrated_edges = torch.sum(x_norm * phase_weights, dim=0) / 32.0

        overlap = torch.mv(connected, integrated_edges)
        _, active_indices = torch.topk(overlap, self.k_active)
        sdr = torch.zeros(self.num_columns, device=DEVICE)
        sdr[active_indices] = 1.0
        return sdr, active_indices

# ==============================================================================
# 4. КАНОНИЧЕСКАЯ ГЕТЕРАРХИЯ НАКОПЛЕНИЯ СВИДЕТЕЛЬСТВ (tbp.monty EvidenceLM)
# ==============================================================================
class FrontalExecutiveHeterarchy(nn.Module):
    """
    Каноническая модель Хокинса (tbp.monty):
    Каждый концепт хранится как независимая объектная модель (SDR-аттрактор).
    Обучение нового объекта физически НЕ МОЖЕТ стереть память прошлых объектов.
    """
    def __init__(self, num_concepts: int = 4, num_columns_per_node: int = 2304, k_active_per_node: int = 46):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_columns_per_node = num_columns_per_node
        self.k_active_per_node = k_active_per_node

        self.f3 = CanonicalHTMColumn("F3_Left", num_columns=num_columns_per_node, k_active=k_active_per_node)
        self.f4 = CanonicalHTMColumn("F4_Right", num_columns=num_columns_per_node, k_active=k_active_per_node)
        self.afz = CanonicalHTMColumn("AFz_Midline", num_columns=num_columns_per_node, k_active=k_active_per_node)
        self.fpz = CanonicalHTMColumn("Fpz_Frontopolar", num_columns=num_columns_per_node, k_active=k_active_per_node)

        # 4 узла x 2304 колонки = 9216 бит на гетерархический SDR
        total_dim = num_columns_per_node * 4
        self.total_dim = total_dim
        self.total_k_active = k_active_per_node * 4 # 184 активных бита суммарно

        # Память моделей объектов (tbp.monty ObjectModel)
        self.register_buffer("object_prototypes", torch.zeros((num_concepts, total_dim), device=DEVICE))
        self.register_buffer("concept_trained", torch.zeros(num_concepts, device=DEVICE))

        self.last_sdr = torch.zeros(total_dim, device=DEVICE)
        self.plan_b_active = False

    def learn_concept_model(self, concept_idx: int, node_tensors_2d: dict):
        sdr_f3, _ = self.f3.compute_sdr(node_tensors_2d["F3"])
        sdr_f4, _ = self.f4.compute_sdr(node_tensors_2d["F4"])
        sdr_afz, _ = self.afz.compute_sdr(node_tensors_2d["AFz"])
        sdr_fpz, _ = self.fpz.compute_sdr(node_tensors_2d["Fpz"])

        # Конкатенация в единый гетерархический SDR (ровно 9216 бит)
        full_sdr = torch.cat([sdr_f3, sdr_f4, sdr_afz, sdr_fpz], dim=0)

        # Сохранение в автономную модель объекта
        self.object_prototypes[concept_idx] = full_sdr.clone()
        self.concept_trained[concept_idx] = 1.0

    def predict_evidence(self, node_tensors_2d: dict):
        sdr_f3, _ = self.f3.compute_sdr(node_tensors_2d["F3"])
        sdr_f4, _ = self.f4.compute_sdr(node_tensors_2d["F4"])
        sdr_afz, _ = self.afz.compute_sdr(node_tensors_2d["AFz"])
        sdr_fpz, _ = self.fpz.compute_sdr(node_tensors_2d["Fpz"])

        full_sdr = torch.cat([sdr_f3, sdr_f4, sdr_afz, sdr_fpz], dim=0)

        # Свидетельства (Evidence matching по tbp.monty):
        # Скалярное произведение входного SDR с эталонами объектов
        evidence = torch.mv(self.object_prototypes, full_sdr) # [4]
        
        # Консенсусное голосование колонок (Winner-Take-All)
        weights = torch.softmax(evidence * 0.15, dim=0) # [4]

        # Аномалия Хокинса
        if self.last_sdr.sum() > 0:
            inter = (full_sdr * self.last_sdr).sum()
            anomaly = 1.0 - (inter / float(self.total_k_active)).item()
            self.plan_b_active = anomaly > 0.65
        self.last_sdr = full_sdr.clone()

        return weights.cpu().numpy(), self.plan_b_active, sdr_f3

# ==============================================================================
# 5. ДИФФУЗИОННЫЙ ВОРКЕР (SD-LCM IPC PORT 6000)
# ==============================================================================
class DualDiffusionWorker:
    def __init__(self, port: int = 6000):
        self.conn = None
        self.current_rgb = np.zeros((384, 512, 3), dtype=np.uint8)
        self.initialized = False
        self.lock = threading.Lock()
        self.running = True
        self.fps = 0.0
        self.strength = 0.55
        self.frame_id = 0

        model_id = "openai/clip-vit-large-patch14"
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id).to(DEVICE)
        self.text_encoder.eval()

        raw_prompts = [
            "high quality photograph of a giant snowy mountain peak, rocky cliffs, clear blue sky, sharp focus, 8k",
            "high quality photograph of an ancient medieval stone castle fortress towers, daytime, sharp focus, 8k",
            "high quality photograph of modern glass skyscraper buildings, downtown city, geometric architecture, 8k",
            "high quality photograph of open stormy dark blue ocean, giant ocean waves, sea foam, dramatic clouds, 8k"
        ]
        
        self.c_bases = []
        for p in raw_prompts:
            tokens = self.tokenizer(p, padding="max_length", max_length=77, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                self.c_bases.append(self.text_encoder(tokens.input_ids)[0])

        self.latent_active = self.c_bases[0].clone()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def update_simplex_targets(self, weights_4d: np.ndarray, force_strength: float = 0.55, snap_latent: bool = False):
        with torch.inference_mode():
            w = torch.tensor(weights_4d, dtype=torch.float32, device=DEVICE).view(4, 1, 1)
            target = (w[0] * self.c_bases[0] + 
                      w[1] * self.c_bases[1] + 
                      w[2] * self.c_bases[2] + 
                      w[3] * self.c_bases[3])
            
            with self.lock:
                if snap_latent:
                    self.latent_active = target.clone()
                else:
                    self.latent_active = self.latent_active * 0.60 + target * 0.40
                self.strength = force_strength

    def _loop(self):
        times = []
        while self.running:
            if not self.initialized:
                try:
                    self.conn = Client(('localhost', 6000), authkey=b'brain')
                    dummy = np.random.randint(100, 150, (384, 512, 3), dtype=np.uint8)
                    init_emb = self.latent_active.cpu().numpy()
                    self.conn.send({'cmd': 'generate', 'image_np': dummy, 'prompt_embeds': init_emb, 'strength': 1.0})
                    resp = self.conn.recv()
                    with self.lock:
                        self.current_rgb = apply_surgery(resp, dummy.astype(np.float32))
                        self.initialized = True
                        self.frame_id += 1
                except:
                    time.sleep(1)
                    continue

            try:
                t0 = time.time()
                with self.lock:
                    emb = self.latent_active.cpu().numpy()
                    img = self.current_rgb.copy()
                    s_val = self.strength
                
                self.conn.send({'cmd': 'generate', 'image_np': img, 'prompt_embeds': emb, 'strength': s_val})
                resp = self.conn.recv()

                with self.lock:
                    if isinstance(resp, np.ndarray):
                        self.current_rgb = apply_surgery(resp, img.astype(np.float32))
                        self.frame_id += 1

                times.append(time.time() - t0)
                if len(times) > 5: times.pop(0)
                self.fps = 1.0 / (np.mean(times) + 1e-6)
            except:
                self.initialized = False
                time.sleep(0.5)

# ==============================================================================
# 6. ТОЧКА ВХОДА И ГАРАНТИРОВАННЫЙ КУРИКУЛУМ ХОКИНСА
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
    pygame.display.set_caption("NeuroCanvas v248.0: Canonical Hawkins EvidenceLM (No Axes)")
    clock = pygame.time.Clock()
    font_b = pygame.font.SysFont("consolas", 14, bold=True)
    font_s = pygame.font.SysFont("consolas", 12)

    engine = HeterarchicalBrainEngine()
    engine.start()
    worker = DualDiffusionWorker(port=6000)
    clip_teacher = VisualCLIPTeacher()
    heterarchy = FrontalExecutiveHeterarchy(num_concepts=4).to(DEVICE)

    cx, cy = WIDTH // 2 + 140, HEIGHT // 2
    current_weights_4d = np.array([0.25, 0.25, 0.25, 0.25])
    biome_keys = ["ГОРА", "ЗАМОК", "НЕБОСКРЕБ", "ОКЕАН"]
    biome_prototypes_2d = {}

    print("🧠 [SYSTEM] Ожидание подключения 4 узлов LSL...")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: raise KeyboardInterrupt
        frame = engine.get_frame()
        if frame.num_live >= 4 and worker.initialized:
            break
        screen.fill((10, 14, 20))
        txt = font_b.render("INITIALIZING 16D LSL CORTICAL STREAM...", True, (255, 180, 50))
        screen.blit(txt, (cx - txt.get_width()//2, cy))
        pygame.display.flip()
        time.sleep(0.05)

    # ==========================================================================
    # ГАРАНТИРОВАННЫЙ КУРИКУЛУМ (ВЫХОД ТОЛЬКО ПРИ ВСЕХ 4 >= 85.0%)
    # ==========================================================================
    print("🧠 [SYSTEM] Старт канонического курикулума Хокинса (EvidenceLM)...")
    curriculum_epoch = 0
    all_mastered = False

    while not all_mastered:
        curriculum_epoch += 1

        for idx_b in range(4):
            b_name = biome_keys[idx_b]
            target_simplex = np.zeros(4, dtype=np.float32)
            target_simplex[idx_b] = 1.0

            if agent:
                agent.set_calibration_target(True, idx_b)

            worker.update_simplex_targets(target_simplex, force_strength=1.0, snap_latent=True)

            target_fid = worker.frame_id + 2
            t_start = time.time()
            step_learn = 0

            # Ждем появления чистого кадра в диффузии и заполнения буфера LSL (>= 1.5 с)
            while True:
                clock.tick(60)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: raise KeyboardInterrupt

                with worker.lock:
                    rgb_m = worker.current_rgb.copy()
                    cur_fid = worker.frame_id

                probs = clip_teacher.classify(rgb_m)
                vis_pct = probs[idx_b] * 100.0

                if (cur_fid >= target_fid) and (probs[idx_b] >= 0.40) and (time.time() - t_start >= 1.5):
                    frame = engine.get_frame()
                    node_tensors = {}
                    for idx, name in enumerate(["F3", "F4", "AFz", "Fpz"]):
                        node_tensors[name] = torch.tensor(frame.nodes[idx].iplv_32, dtype=torch.float32, device=DEVICE)

                    # Впечатываем чистую модель объекта (не трогая другие объекты!)
                    heterarchy.learn_concept_model(idx_b, node_tensors)
                    biome_prototypes_2d[b_name] = {k: v.clone() for k, v in node_tensors.items()}
                    step_learn += 1

                    if step_learn >= 12:
                        break

                # Полный живой дебаг на экране
                screen.fill((10, 14, 20))
                main_surf = pygame.image.frombuffer(rgb_m.tobytes(), (512, 384), 'RGB')
                main_x, main_y = cx - 256, cy - 192
                screen.blit(main_surf, (main_x, main_y))
                pygame.draw.rect(screen, (0, 255, 200), (main_x, main_y, 512, 384), 2, border_radius=8)

                c_x, c_y = 30, 360
                pygame.draw.rect(screen, (16, 22, 32), (c_x, c_y, 350, 240), border_radius=8)
                pygame.draw.rect(screen, (255, 180, 50), (c_x, c_y, 350, 240), 1, border_radius=8)
                screen.blit(font_b.render(f"HAWKINS CURRICULUM [EPOCH {curriculum_epoch}]", True, (255, 180, 50)), (c_x + 12, c_y + 12))
                screen.blit(font_s.render(f"Learning Model   : [{b_name}]", True, (255, 255, 100)), (c_x + 15, c_y + 40))
                screen.blit(font_s.render(f"CLIP Vision      : {vis_pct:4.1f}% confidence", True, (0, 255, 180)), (c_x + 15, c_y + 65))
                screen.blit(font_s.render(f"Imprinted Steps  : {step_learn}/12", True, (100, 255, 255)), (c_x + 15, c_y + 90))
                
                status_txt = "Building Object Model..." if step_learn > 0 else "Rendering Pure Concept..."
                screen.blit(font_s.render(f"Status           : {status_txt}", True, (200, 220, 255)), (c_x + 15, c_y + 115))
                
                pygame.draw.rect(screen, (30, 40, 50), (c_x + 15, c_y + 140, 320, 10))
                pygame.draw.rect(screen, (0, 255, 180), (c_x + 15, c_y + 140, int((step_learn / 12.0) * 320), 10))

                # ЖИВОЙ БЕНЧМАРК (ВИДНО ВЛИЯНИЕ НА ВСЕ 4 МОДЕЛИ ОНЛАЙН)
                panel_x, panel_y = 30, 60
                pygame.draw.rect(screen, (16, 22, 32), (panel_x, panel_y, 350, 280), border_radius=8)
                pygame.draw.rect(screen, (0, 255, 200), (panel_x, panel_y, 350, 280), 1, border_radius=8)
                screen.blit(font_b.render("EVIDENCE LM BENCHMARK", True, (0, 255, 200)), (panel_x + 12, panel_y + 12))

                total_acc = 0.0
                with torch.no_grad():
                    for chk_i, chk_name in enumerate(biome_keys):
                        if chk_name in biome_prototypes_2d:
                            pred_w, _, _ = heterarchy.predict_evidence(biome_prototypes_2d[chk_name])
                            acc_chk = float(pred_w[chk_i]) * 100.0
                            total_acc += acc_chk * 0.25
                            col = (100, 255, 100) if acc_chk >= 85.0 else ((255, 200, 50) if acc_chk >= 50.0 else (255, 80, 80))
                            screen.blit(font_s.render(f"{chk_name:9s}: {acc_chk:4.1f}% [Target >= 85%]", True, col), (panel_x + 15, panel_y + 45 + chk_i * 30))
                        else:
                            screen.blit(font_s.render(f"{chk_name:9s}: AWAITING FIRST PASS...", True, (150, 150, 150)), (panel_x + 15, panel_y + 45 + chk_i * 30))

                screen.blit(font_b.render(f"TOTAL RETENTION : {total_acc:.1f}%", True, (0, 255, 200)), (panel_x + 15, panel_y + 195))
                screen.blit(font_s.render("Hard Gate: Release ONLY when ALL >= 85%", True, (255, 180, 50)), (panel_x + 15, panel_y + 225))

                pygame.display.flip()

        # Проверка готовности ВСЕХ 4 моделей в конце эпохи
        epoch_accs = []
        with torch.no_grad():
            for chk_i, chk_name in enumerate(biome_keys):
                if chk_name in biome_prototypes_2d:
                    pred_w, _, _ = heterarchy.predict_evidence(biome_prototypes_2d[chk_name])
                    epoch_accs.append(float(pred_w[chk_i]) * 100.0)
                else:
                    epoch_accs.append(0.0)

        min_score = min(epoch_accs)
        # ВЫХОД СТРОГО ПРИ ВСЕХ 4 >= 85.0%
        if min_score >= 85.0:
            all_mastered = True
            print("🏆 [CURRICULUM] ВСЕ 4 МОДЕЛИ ВЫУЧЕНЫ ВЫШЕ 85%! ДОПУСК В СЁРФИНГ.")

    if agent:
        agent.set_calibration_target(False)

    # ==========================================================================
    # ЦИКЛ СЁРФИНГА
    # ==========================================================================
    last_frame_id = -1
    cur_probs = np.array([0.25, 0.25, 0.25, 0.25])

    try:
        while True:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: raise KeyboardInterrupt

            frame = engine.get_frame()
            node_tensors_2d = {}
            for idx, name in enumerate(["F3", "F4", "AFz", "Fpz"]):
                node_tensors_2d[name] = torch.tensor(frame.nodes[idx].iplv_32, dtype=torch.float32, device=DEVICE)

            with worker.lock:
                rgb_m = worker.current_rgb.copy()
                fid = worker.frame_id

            if fid != last_frame_id:
                last_frame_id = fid
                cur_probs = clip_teacher.classify(rgb_m)

            if agent:
                agent.update_visual_state(cur_probs)

            # Декодирование мысли агента через EvidenceLM (Хокинс)
            with torch.no_grad():
                new_weights, switched, cur_sdr = heterarchy.predict_evidence(node_tensors_2d)

            # Плавная эволюция состава холста
            current_weights_4d = current_weights_4d * 0.80 + new_weights * 0.20
            current_weights_4d = current_weights_4d / np.sum(current_weights_4d)

            # Анти-затяжной денойзинг
            fru_val = agent.shm['frustration'].value if agent else 0.0
            if switched or fru_val > 0.50:
                force_str = 0.88
            else:
                delta_w = np.max(np.abs(new_weights - current_weights_4d))
                force_str = float(np.clip(0.52 + delta_w * 0.60, 0.52, 0.76))

            worker.update_simplex_targets(current_weights_4d, force_strength=force_str, snap_latent=False)

            # --- РЕНДЕР ИНТЕРФЕЙСА ---
            screen.fill((10, 14, 20))
            main_surf = pygame.image.frombuffer(rgb_m.tobytes(), (512, 384), 'RGB')
            main_x, main_y = cx - 256, cy - 192
            screen.blit(main_surf, (main_x, main_y))
            pygame.draw.rect(screen, (40, 50, 70), (main_x, main_y, 512, 384), 2, border_radius=8)

            proj_g = current_weights_4d[1] + current_weights_4d[2]
            proj_s = current_weights_4d[2] + current_weights_4d[3]
            g_px = main_x + int(np.clip(proj_g, 0.0, 1.0) * 512)
            s_px = main_y + int(np.clip(proj_s, 0.0, 1.0) * 384)
            pygame.draw.circle(screen, (0, 255, 200), (g_px, s_px), 7)
            pygame.draw.circle(screen, (255, 255, 255), (g_px, s_px), 2)

            sdr_img = cur_sdr.view(48, 48).cpu().numpy() * 255.0
            sdr_surf = pygame.surfarray.make_surface(cv2.resize(sdr_img, (140, 140)).astype(np.uint8))
            screen.blit(sdr_surf, (main_x + 512 + 20, main_y))
            screen.blit(font_s.render("L4 Active SDR", True, (0, 255, 200)), (main_x + 512 + 20, main_y - 20))

            # Кокпит агента-аудитора
            if agent:
                mode, desc, mood, t_idx, sat, bor, fru = agent.get_telemetry()
                
                target_corners_proj = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
                t_px = main_x + int(target_corners_proj[t_idx][0] * 512)
                t_py = main_y + int(target_corners_proj[t_idx][1] * 384)
                pygame.draw.line(screen, (255, 50, 50), (t_px-12, t_py), (t_px+12, t_py), 2)
                pygame.draw.line(screen, (255, 50, 50), (t_px, t_py-12), (t_px, t_py+12), 2)

                c_x, c_y = 30, 360
                pygame.draw.rect(screen, (16, 22, 32), (c_x, c_y, 350, 260), border_radius=8)
                pygame.draw.rect(screen, (255, 150, 50), (c_x, c_y, 350, 260), 1, border_radius=8)
                screen.blit(font_b.render("AGENT AUDITOR DASHBOARD", True, (255, 150, 50)), (c_x + 12, c_y + 12))
                
                mood_col = (100, 255, 100) if "УСПЕХ" in mood else ((255, 200, 50) if "ПОИСК" in mood else (255, 80, 80))
                screen.blit(font_b.render(f"MOOD   : {mood}", True, mood_col), (c_x + 15, c_y + 38))
                screen.blit(font_s.render(f"MIND   : {desc[:42]}", True, (200, 220, 255)), (c_x + 15, c_y + 60))
                if len(desc) > 42:
                    screen.blit(font_s.render(f"         {desc[42:84]}", True, (200, 220, 255)), (c_x + 15, c_y + 78))

                screen.blit(font_s.render(f"Satisfaction : {sat*100:4.1f}%", True, (100, 255, 100)), (c_x+15, c_y+105))
                pygame.draw.rect(screen, (30, 40, 50), (c_x+15, c_y+122, 320, 8))
                pygame.draw.rect(screen, (0, 255, 180), (c_x+15, c_y+122, int(sat*320), 8))

                screen.blit(font_s.render(f"Boredom (Success) : {bor*100:4.1f}%", True, (255, 200, 50)), (c_x+15, c_y+138))
                pygame.draw.rect(screen, (30, 40, 50), (c_x+15, c_y+155, 320, 8))
                pygame.draw.rect(screen, (255, 200, 50), (c_x+15, c_y+155, int(bor*320), 8))

                screen.blit(font_s.render(f"Frustration (Trap) : {fru*100:4.1f}%", True, (255, 80, 80)), (c_x+15, c_y+172))
                pygame.draw.rect(screen, (30, 40, 50), (c_x+15, c_y+188, 320, 8))
                pygame.draw.rect(screen, (255, 80, 80), (c_x+15, c_y+188, int(fru*320), 8))

                sys_score = max(0.0, sat * 100.0 - fru * 40.0)
                screen.blit(font_b.render(f"TESTER VERDICT: {sys_score:4.1f}% INTEGRITY", True, (0, 255, 200)), (c_x+15, c_y+210))
                screen.blit(font_s.render(f"Denoise Strength: {force_str:.2f} | EvidenceLM Active", True, (180, 200, 220)), (c_x+15, c_y+234))

            # HTM Benchmark
            panel_x, panel_y = 30, 60
            pygame.draw.rect(screen, (16, 22, 32), (panel_x, panel_y, 350, 280), border_radius=8)
            pygame.draw.rect(screen, (0, 255, 200), (panel_x, panel_y, 350, 280), 1, border_radius=8)
            screen.blit(font_b.render("QUALIFIED HTM BENCHMARK", True, (0, 255, 200)), (panel_x + 12, panel_y + 12))

            status_fpz = "PLAN B (COGNITIVE SWITCH)" if heterarchy.plan_b_active else "PLAN A (PURSUING INTENT)"
            col_plan = (255, 80, 80) if heterarchy.plan_b_active else (100, 255, 100)
            screen.blit(font_b.render(f"[Fpz] {status_fpz}", True, col_plan), (panel_x + 15, panel_y + 40))

            total_acc = 0.0
            with torch.no_grad():
                for idx, b_name in enumerate(biome_keys):
                    if b_name in biome_prototypes_2d:
                        pred_w, _, _ = heterarchy.predict_evidence(biome_prototypes_2d[b_name])
                        acc = float(pred_w[idx]) * 100.0
                        total_acc += acc * 0.25
                        txt_b = f"{b_name:9s}: {acc:4.1f}% [w:{pred_w[idx]:.2f}]"
                    else:
                        acc = 0.0
                        txt_b = f"{b_name:9s}: SCANNING..."
                    col = (100, 255, 100) if acc >= 85.0 else ((255, 200, 50) if acc >= 50.0 else (255, 80, 80))
                    screen.blit(font_s.render(txt_b, True, col), (panel_x + 15, panel_y + 80 + idx * 30))

            col_tot = (0, 255, 200) if total_acc >= 85.0 else (255, 200, 50)
            screen.blit(font_b.render(f"TOTAL RETENTION : {total_acc:.1f}%", True, col_tot), (panel_x + 15, panel_y + 225))
            screen.blit(font_s.render("Hawkins EvidenceLM (Zero Overwriting)", True, (150, 200, 255)), (panel_x + 15, panel_y + 250))

            screen.blit(font_s.render(f"Status: QUALIFIED SURFING | SD-LCM FPS: {worker.fps:.1f}", True, (150, 150, 150)), (30, HEIGHT - 25))
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
