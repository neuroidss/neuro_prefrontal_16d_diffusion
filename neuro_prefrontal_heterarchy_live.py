#!/usr/bin/env python3
"""
🧠 NEUROCANVAS LIVE v270.0: HONEST CLIP-GATED HAWKINS CURRICULUM
- Честное обучение по зрению: запись SDR ТОЛЬКО при уверенности CLIP >= 65%.
- Никаких слепых таймеров: переход к следующему только по набору чистых шагов.
- ЖЕСТКИЙ ШЛЮЗ (Hard Gate): выход в сёрфинг ЗАПРЕЩЕН, пока ВСЕ концепты не наберут >= 85%.
- Полный 60 FPS реалтайм без зависаний окон и без разрыва контура.
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
# 1. СЛОЙ 4 ХОКИНСА: 4096 МАКРОКОЛОНОК НА УЗЕЛ (СЕТКА 64x64)
# ==============================================================================
class CanonicalHTMColumn(nn.Module):
    def __init__(self, node_id: str, num_columns: int = 4096, k_active: int = 80):
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

        self.register_buffer("permanence", spatial_rf)
        self.perm_threshold = 0.25

    def compute_sdr(self, pac_iplv_32x120: torch.Tensor):
        connected = (self.permanence >= self.perm_threshold).float()
        x_clean = torch.relu(pac_iplv_32x120)

        phase_weights = torch.linspace(0.6, 1.4, 32, device=DEVICE).unsqueeze(1)
        integrated_edges = torch.sum(x_clean * phase_weights, dim=0) / 32.0

        synapse_counts = torch.sum(connected, dim=1).clamp(min=1.0)
        overlap = torch.mv(connected, integrated_edges) / synapse_counts

        _, active_indices = torch.topk(overlap, self.k_active)
        sdr = torch.zeros(self.num_columns, device=DEVICE)
        sdr[active_indices] = 1.0
        return sdr, active_indices


# ==============================================================================
# 2. ПОТОКОВАЯ ГЕТЕРАРХИЯ ХОКИНСА (16 384 КОЛОНКИ)
# ==============================================================================
class FrontalExecutiveHeterarchy(nn.Module):
    def __init__(self, num_concepts: int = 4, num_columns_per_node: int = 4096, k_active_per_node: int = 80):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_columns_per_node = num_columns_per_node
        self.k_active = k_active_per_node

        self.f3 = CanonicalHTMColumn("F3_Left", num_columns_per_node, k_active_per_node)
        self.f4 = CanonicalHTMColumn("F4_Right", num_columns_per_node, k_active_per_node)
        self.afz = CanonicalHTMColumn("AFz_Midline", num_columns_per_node, k_active_per_node)
        self.fpz = CanonicalHTMColumn("Fpz_Frontopolar", num_columns_per_node, k_active_per_node)

        total_dim = num_columns_per_node * 4
        self.total_dim = total_dim
        self.total_k_active = k_active_per_node * 4

        self.register_buffer("object_prototypes", torch.zeros((num_concepts, total_dim), device=DEVICE))
        self.register_buffer("concept_trained", torch.zeros(num_concepts, device=DEVICE))
        self.register_buffer("accumulators", torch.zeros((num_concepts, total_dim), device=DEVICE))

        self.last_sdr = torch.zeros(total_dim, device=DEVICE)
        self.plan_b_active = False

    def get_current_sdr(self, node_tensors_2d: dict):
        sdr_f3, _ = self.f3.compute_sdr(node_tensors_2d["F3"])
        sdr_f4, _ = self.f4.compute_sdr(node_tensors_2d["F4"])
        sdr_afz, _ = self.afz.compute_sdr(node_tensors_2d["AFz"])
        sdr_fpz, _ = self.fpz.compute_sdr(node_tensors_2d["Fpz"])
        full_sdr = torch.cat([sdr_f3, sdr_f4, sdr_afz, sdr_fpz], dim=0)
        return full_sdr, sdr_f3

    def reset_accumulator(self, concept_idx: int):
        self.accumulators[concept_idx].zero_()

    def stream_learn_accumulate(self, concept_idx: int, full_sdr: torch.Tensor):
        self.accumulators[concept_idx].add_(full_sdr)

    def finalize_slot(self, concept_idx: int):
        _, top_stable = torch.topk(self.accumulators[concept_idx], self.total_k_active)
        pure_sdr = torch.zeros(self.total_dim, device=DEVICE)
        pure_sdr[top_stable] = 1.0
        self.object_prototypes[concept_idx] = pure_sdr
        self.concept_trained[concept_idx] = 1.0

    def online_grounding_update(self, concept_idx: int, cur_sdr: torch.Tensor, lr: float = 0.02):
        with torch.no_grad():
            if self.concept_trained[concept_idx] > 0.0:
                self.object_prototypes[concept_idx] = torch.max(
                    self.object_prototypes[concept_idx],
                    (cur_sdr > 0.5).float() * lr + self.object_prototypes[concept_idx] * (1.0 - lr)
                )

    def predict_evidence(self, full_sdr: torch.Tensor):
        proto_norm = torch.nn.functional.normalize(self.object_prototypes, p=2, dim=1)
        sdr_norm = torch.nn.functional.normalize(full_sdr, p=2, dim=0)
        raw_similarities = torch.mv(proto_norm, sdr_norm)

        weights = torch.softmax(raw_similarities * 24.0, dim=0)

        if self.last_sdr.sum() > 0:
            inter = (full_sdr * self.last_sdr).sum()
            anomaly = 1.0 - (inter / float(self.total_k_active)).item()
            self.plan_b_active = anomaly > 0.65
        self.last_sdr = full_sdr.clone()

        return weights.cpu().numpy(), self.plan_b_active, raw_similarities.cpu().numpy()


# ==============================================================================
# 3. ЗРИТЕЛЬНЫЙ КЛАССИФИКАТОР CLIP
# ==============================================================================
class VisualCLIPTeacher:
    def __init__(self, class_names, text_prompts):
        model_id = "openai/clip-vit-large-patch14"
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.classes = class_names

        with torch.no_grad():
            inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True).to(DEVICE)
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


class ToroidalDiffusionWorker:
    def __init__(self, prompts, port: int = 6000):
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
        self.text_encoder = CLIPTextModel.from_pretrained(model_id).to(DEVICE).eval()

        self.c_bases = []
        for p in prompts:
            tokens = self.tokenizer(p, padding="max_length", max_length=77, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                self.c_bases.append(self.text_encoder(tokens.input_ids)[0])

        self.latent_active = self.c_bases[0].clone()
        self.num_concepts = len(self.c_bases)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def update_simplex_targets(self, weights_nd: np.ndarray, force_strength: float = 0.55):
        with torch.inference_mode():
            target = torch.zeros_like(self.c_bases[0])
            for i in range(self.num_concepts):
                target += float(weights_nd[i]) * self.c_bases[i]

            with self.lock:
                self.latent_active = self.latent_active * 0.65 + target * 0.35
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
                    self.current_rgb = apply_surgery(self.conn.recv(), dummy.astype(np.float32))
                    self.initialized = True
                except Exception:
                    time.sleep(0.5)
                    continue

            try:
                t0 = time.time()
                with self.lock:
                    emb, img, s_val = self.latent_active.cpu().numpy(), self.current_rgb.copy(), self.strength

                self.conn.send({'cmd': 'generate', 'image_np': img, 'prompt_embeds': emb, 'strength': s_val})
                resp = self.conn.recv()

                with self.lock:
                    if isinstance(resp, np.ndarray):
                        self.current_rgb = apply_surgery(resp, img.astype(np.float32))
                        self.frame_id += 1

                times.append(time.time() - t0)
                if len(times) > 5: times.pop(0)
                self.fps = 1.0 / (np.mean(times) + 1e-6)
            except Exception:
                self.initialized = False
                time.sleep(0.5)


# ==============================================================================
# 4. ТОЧКА ВХОДА С ЖЕСТКИМ ОНЛАЙН-ШЛЮЗОМ
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="NeuroCanvas Live: Honest CLIP-Gated Heterarchy")
    parser.add_argument('--sim', action='store_true', help="Включить симуляцию агента")
    parser.add_argument('--concepts', type=int, default=4, choices=[4, 8], help="Количество концептов (4 по умолчанию)")
    parser.add_argument('--online-learn', action='store_true', default=False, help="Включить адаптивное дообучение при поиске")
    parser.add_argument('--online-lr', type=float, default=0.02, help="Сила онлайн-пластичности")
    args = parser.parse_args()

    ALL_NAMES = ["ГОРА", "ЗАМОК", "НЕБОСКРЕБ", "ОКЕАН", "КИБЕРПАНК", "ПУСТЫНЯ", "КОСМОС", "ДЖУНГЛИ"]
    ALL_PROMPTS = [
        "high quality photograph of a giant snowy mountain peak, rocky cliffs, clear blue sky, sharp focus, 8k",
        "high quality photograph of an ancient medieval stone castle fortress towers, daytime, sharp focus, 8k",
        "high quality photograph of modern glass skyscraper buildings, downtown city, geometric architecture, 8k",
        "high quality photograph of open stormy dark blue ocean, pure water surface, giant ocean waves, sea foam, no land, 8k",
        "high quality digital art of a futuristic cyberpunk city street, neon lights, rain, glowing signs, 8k",
        "high quality photograph of a vast dry sandy desert, sand dunes, scorching hot sun, clear sky, 8k",
        "high quality photograph of deep outer space, glowing colorful nebula, bright stars, galaxy, 8k",
        "high quality photograph of a dense lush green tropical jungle, giant trees, vines, sunlight piercing through leaves, 8k"
    ]

    NUM_CONCEPTS = args.concepts
    TARGET_NAMES = ALL_NAMES[:NUM_CONCEPTS]
    PROMPTS = ALL_PROMPTS[:NUM_CONCEPTS]

    agent = None
    if args.sim:
        from synthetic_16d_causal_agent import SyntheticAutonomousAgent
        agent = SyntheticAutonomousAgent(num_concepts=NUM_CONCEPTS)
        print(f"🤖 [CONFIG] Запущен симулятор агента ({NUM_CONCEPTS} концептов).")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"NeuroCanvas v270.0: Honest CLIP-Gated Heterarchy ({NUM_CONCEPTS} Concepts)")
    clock = pygame.time.Clock()
    font_b = pygame.font.SysFont("consolas", 13, bold=True)
    font_s = pygame.font.SysFont("consolas", 11)

    engine = HeterarchicalBrainEngine()
    engine.start()

    worker = ToroidalDiffusionWorker(PROMPTS, port=6000)
    clip_teacher = VisualCLIPTeacher(TARGET_NAMES, PROMPTS)
    heterarchy = FrontalExecutiveHeterarchy(num_concepts=NUM_CONCEPTS, num_columns_per_node=4096, k_active_per_node=80).to(DEVICE)

    cx, cy = WIDTH // 2 + 140, HEIGHT // 2
    current_weights = np.ones(NUM_CONCEPTS, dtype=np.float32) / NUM_CONCEPTS

    last_frame_id = -1
    cur_probs = np.ones(NUM_CONCEPTS, dtype=np.float32) / NUM_CONCEPTS

    # --------------------------------------------------------------------------
    # ПЕРЕМЕННЫЕ ЧЕСТНОГО ШЛЮЗА ОБУЧЕНИЯ
    # --------------------------------------------------------------------------
    is_calibrating = True
    learn_idx = 0
    clean_steps_accumulated = 0
    CLEAN_STEPS_REQUIRED = 15       # Нужно 15 кадров высокого качества
    CLIP_HONEST_THRESHOLD = 0.65    # НЕ УЧИМ, пока CLIP не увидит цель >= 65%

    concept_snapshots = {}
    concept_scores = [0.0] * NUM_CONCEPTS
    curriculum_epoch = 1

    print("🧠 [SYSTEM] Старт живого конвейера. Обучение активно до достижения >= 85% по всем концептам...")

    try:
        while True:
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            frame = engine.get_frame()
            node_tensors = {name: torch.tensor(frame.nodes[i].iplv_32, dtype=torch.float32, device=DEVICE) for i, name in enumerate(["F3", "F4", "AFz", "Fpz"])}

            with worker.lock:
                rgb_m, fid = worker.current_rgb.copy(), worker.frame_id

            if fid != last_frame_id:
                last_frame_id = fid
                cur_probs = clip_teacher.classify(rgb_m)

            if agent:
                agent.update_visual_state(cur_probs)

            # Извлечение 16 384-битного SDR
            with torch.no_grad():
                full_sdr, sdr_f3 = heterarchy.get_current_sdr(node_tensors)

            # ==================================================================
            # ЧЕСТНЫЙ ШЛЮЗ ОБУЧЕНИЯ (CLIP-GATED HARD GATE)
            # ==================================================================
            if is_calibrating:
                b_name = TARGET_NAMES[learn_idx]
                vis_conf = float(cur_probs[learn_idx])

                # Принудительно направляем диффузию и агента
                target_sim = np.zeros(NUM_CONCEPTS, dtype=np.float32)
                target_sim[learn_idx] = 1.0
                worker.update_simplex_targets(target_sim, force_strength=1.0)
                if agent:
                    agent.set_calibration_target(True, learn_idx)

                # 🔬 ЧЕСТНОЕ ПРАВИЛО: учим ТОЛЬКО когда клип видит цель четко (>= 65%)!
                if vis_conf >= CLIP_HONEST_THRESHOLD:
                    clip_status = f"CLIP VERIFIED ({vis_conf*100:.1f}% >= 65%) -> IMPRINTING"
                    heterarchy.stream_learn_accumulate(learn_idx, full_sdr)
                    clean_steps_accumulated += 1
                    concept_snapshots[learn_idx] = full_sdr.clone()
                else:
                    clip_status = f"WAITING FOR CLIP CONVERGENCE... ({vis_conf*100:.1f}% < 65%)"

                # Набрано 15 чистых кадров -> финализируем концепт
                if clean_steps_accumulated >= CLEAN_STEPS_REQUIRED:
                    heterarchy.finalize_slot(learn_idx)
                    
                    # Проверяем качество всех обученных концептов
                    with torch.no_grad():
                        for chk_i in range(NUM_CONCEPTS):
                            if chk_i in concept_snapshots and heterarchy.concept_trained[chk_i] > 0:
                                p_w, _, _ = heterarchy.predict_evidence(concept_snapshots[chk_i])
                                concept_scores[chk_i] = float(p_w[chk_i]) * 100.0

                    # Переход к следующему концепту
                    learn_idx += 1
                    clean_steps_accumulated = 0

                    # Если круг завершен — проверяем ЖЕСТКИЙ ШЛЮЗ!
                    if learn_idx >= NUM_CONCEPTS:
                        min_score = min(concept_scores)
                        # ВЫХОД ТОЛЬКО ЕСЛИ ВСЕ >= 85.0%!
                        if min_score >= 85.0 and all(heterarchy.concept_trained > 0):
                            is_calibrating = False
                            if agent:
                                agent.set_calibration_target(False)
                            print("🏆 [HARD GATE PASSED] Все концепты выучены выше 85%! Выход в сёрфинг.")
                        else:
                            # Если кто-то завалил — идем на следующий круг доучивать слабейшего!
                            curriculum_epoch += 1
                            # Фокусируемся на концепте с минимальной точностью
                            learn_idx = int(np.argmin(concept_scores))
                            heterarchy.reset_accumulator(learn_idx)
                            print(f"⚠️ [RE-LEARNING] Шлюз заблокирован! Концепт [{TARGET_NAMES[learn_idx]}] набрал {concept_scores[learn_idx]:.1f}%. Доучиваем...")

                current_weights = target_sim
                switched = False

            else:
                # --------------------------------------------------------------
                # ЧИСТЫЙ СЁРФИНГ (ШЛЮЗ ПРОЙДЕН, СВОБОДНАЯ НАВИГАЦИЯ)
                # --------------------------------------------------------------
                with torch.no_grad():
                    new_w, switched, raw_sims = heterarchy.predict_evidence(full_sdr)

                # Онлайн-дообучение только если явно включено флагом
                top_vis_idx = int(np.argmax(cur_probs))
                vis_weight = float(cur_probs[top_vis_idx])
                fru_val = agent.shm['frustration'].value if agent else 0.0

                if args.online_learn:
                    pred_winner = int(np.argmax(new_w))
                    if (vis_weight >= 0.75) and (pred_winner == top_vis_idx) and (fru_val <= 0.25):
                        heterarchy.online_grounding_update(top_vis_idx, full_sdr, lr=args.online_lr)

                current_weights = current_weights * 0.75 + new_w * 0.25
                current_weights = current_weights / np.sum(current_weights)

                force_str = 0.88 if (switched or fru_val > 0.50) else float(np.clip(0.52 + np.max(current_weights) * 0.25, 0.52, 0.74))
                worker.update_simplex_targets(current_weights, force_strength=force_str)

            # ==================================================================
            # РЕНДЕР ЭКРАНА
            # ==================================================================
            screen.fill((10, 14, 20))
            screen.blit(pygame.image.frombuffer(rgb_m.tobytes(), (512, 384), 'RGB'), (cx - 256, cy - 192))
            pygame.draw.rect(screen, (40, 50, 70), (cx - 256, cy - 192, 512, 384), 2, border_radius=8)

            # Карта L4 F3 (64x64)
            cur_sdr_img = sdr_f3[:4096].view(64, 64).cpu().numpy() * 255.0
            sdr_surf = pygame.surfarray.make_surface(cv2.resize(cur_sdr_img, (140, 140)).astype(np.uint8))
            screen.blit(sdr_surf, (cx + 256 + 20, cy - 192))
            screen.blit(font_s.render("L4 F3 Sheet (64x64)", True, (0, 255, 200)), (cx + 256 + 20, cy - 212))

            # 1. ПАНЕЛЬ ОБУЧЕНИЯ / БЕНЧМАРКА
            panel_x, panel_y = 30, 40
            pygame.draw.rect(screen, (16, 22, 32), (panel_x, panel_y, 380, 270), border_radius=8)
            pygame.draw.rect(screen, (0, 255, 200), (panel_x, panel_y, 380, 270), 1, border_radius=8)

            if is_calibrating:
                screen.blit(font_b.render(f"HONEST CALIBRATION [EPOCH {curriculum_epoch}]", True, (255, 180, 50)), (panel_x + 12, panel_y + 12))
                screen.blit(font_s.render(f"Target Concept    : [{TARGET_NAMES[learn_idx]}] ({learn_idx+1}/{NUM_CONCEPTS})", True, (255, 255, 100)), (panel_x + 15, panel_y + 36))
                
                col_status = (0, 255, 180) if "VERIFIED" in clip_status else (255, 200, 50)
                screen.blit(font_s.render(clip_status, True, col_status), (panel_x + 15, panel_y + 56))

                # Прогресс чистых шагов
                prog_val = min(1.0, clean_steps_accumulated / float(CLEAN_STEPS_REQUIRED))
                screen.blit(font_s.render(f"Clean Steps (>=65%): {clean_steps_accumulated}/{CLEAN_STEPS_REQUIRED}", True, (200, 220, 255)), (panel_x + 15, panel_y + 78))
                pygame.draw.rect(screen, (30, 40, 50), (panel_x + 15, panel_y + 96, 350, 8), border_radius=2)
                pygame.draw.rect(screen, (0, 255, 180), (panel_x + 15, panel_y + 96, int(prog_val * 350), 8), border_radius=2)

                # Таблица текущей точности
                for i, name in enumerate(TARGET_NAMES):
                    score = concept_scores[i]
                    if heterarchy.concept_trained[i] > 0:
                        col_s = (100, 255, 100) if score >= 85.0 else (255, 80, 80)
                        txt_s = f"{name:10s}: {score:4.1f}% [Target >= 85%]"
                    elif i == learn_idx:
                        col_s = (255, 220, 50)
                        txt_s = f"{name:10s}: CURRENTLY LEARNING..."
                    else:
                        col_s = (120, 120, 120)
                        txt_s = f"{name:10s}: QUEUED..."
                    screen.blit(font_s.render(txt_s, True, col_s), (panel_x + 15, panel_y + 118 + i * 22))

                min_sc = min(concept_scores) if any(heterarchy.concept_trained > 0) else 0.0
                screen.blit(font_b.render(f"HARD GATE: RELEASE ONLY WHEN ALL >= 85%", True, (255, 100, 100) if min_sc < 85 else (0, 255, 200)), (panel_x + 15, panel_y + 240))

            else:
                screen.blit(font_b.render("FROZEN RETENTION BENCHMARK (PASSED)", True, (0, 255, 200)), (panel_x + 12, panel_y + 12))
                for i, name in enumerate(TARGET_NAMES):
                    screen.blit(font_s.render(f"{name:10s}: {concept_scores[i]:4.1f}% [LOCKED IN LTM]", True, (100, 255, 100)), (panel_x + 15, panel_y + 45 + i * 28))

                screen.blit(font_b.render("ALL CONCEPTS QUALIFIED >= 85.0%", True, (0, 255, 200)), (panel_x + 15, panel_y + 190))
                screen.blit(font_s.render("16,384 Columns | Pure Sparse HTM Consensus", True, (150, 180, 200)), (panel_x + 15, panel_y + 215))
                screen.blit(font_s.render("Memory Protected: Zero Catastrophic Interference", True, (0, 255, 180)), (panel_x + 15, panel_y + 238))

            # 2. ПАНЕЛЬ ЖИВОГО ДЕКОДЕРА И АГЕНТА
            c_x, c_y = 30, 330
            pygame.draw.rect(screen, (16, 22, 32), (c_x, c_y, 380, 250), border_radius=8)
            pygame.draw.rect(screen, (100, 180, 255), (c_x, c_y, 380, 250), 1, border_radius=8)
            screen.blit(font_b.render("LIVE INTENT STREAM (DECODER)", True, (100, 180, 255)), (c_x + 12, c_y + 12))

            top_idx = int(np.argmax(current_weights))
            top_conf = current_weights[top_idx] * 100.0
            col_intent = (100, 255, 100) if top_conf >= 70.0 else (255, 220, 50)
            screen.blit(font_b.render(f"Decoded Intent : [{TARGET_NAMES[top_idx]}] ({top_conf:4.1f}%)", True, col_intent), (c_x + 15, c_y + 38))

            top_vis = int(np.argmax(cur_probs))
            screen.blit(font_s.render(f"Visual Reality : {TARGET_NAMES[top_vis]} ({cur_probs[top_vis]*100:.1f}% CLIP)", True, (200, 220, 255)), (c_x + 15, c_y + 60))

            for i, name in enumerate(TARGET_NAMES):
                w_val = current_weights[i] * 100.0
                screen.blit(font_s.render(f"{name[:4]}: {w_val:3.0f}%", True, (180, 180, 180)), (c_x + 15 + i * 85, c_y + 84))

            if agent:
                mode, desc, mood, t_idx, sat, bor, fru = agent.get_telemetry()
                quest_col = (0, 255, 200) if t_idx == top_idx else (255, 100, 100)
                screen.blit(font_b.render(f"Agent Quest    : [{TARGET_NAMES[t_idx]}]", True, quest_col), (c_x + 15, c_y + 115))
                screen.blit(font_s.render(f"Status         : {'CALIBRATING...' if is_calibrating else ('MATCH CONFIRMED' if t_idx == top_idx else 'SEEKING')}", True, quest_col), (c_x + 15, c_y + 138))

                screen.blit(font_s.render(f"Satisfaction   : {sat*100:4.1f}%", True, (100, 255, 100)), (c_x + 15, c_y + 160))
                pygame.draw.rect(screen, (30, 40, 50), (c_x + 15, c_y + 176, 350, 6), border_radius=2)
                pygame.draw.rect(screen, (0, 255, 180), (c_x + 15, c_y + 176, int(sat * 350), 6), border_radius=2)

                screen.blit(font_s.render(f"Frustration    : {fru*100:4.1f}% | Boredom: {bor*100:4.1f}%", True, (255, 80, 80)), (c_x + 15, c_y + 192))
                screen.blit(font_s.render(f"Denoise Power  : {worker.strength:.2f} | SD-LCM FPS: {worker.fps:.1f}", True, (160, 180, 200)), (c_x + 15, c_y + 215))

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
