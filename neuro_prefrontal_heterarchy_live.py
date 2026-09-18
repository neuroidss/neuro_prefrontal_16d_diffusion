#!/usr/bin/env python3
"""
🧠 NEUROCANVAS × TBP.MONTY: EQUAL-POOLED HETERARCHICAL CHAOS
- Равноправие всех концептов: и базовые, и отпочковавшиеся живут в едином пуле.
- Режим --concepts 0: старт с 1 концепта (--start-prompt) и самозарождение сложности.
- Флаг --chaos: активация хаоса с командной строки.
- Горячая клавиша: [ПРОБЕЛ] (SPACE) — мгновенное включение/выключение хаоса на лету.
- Переключение роли девайса: клавиши 1 (F3), 2 (F4), 3 (AFz), 4 (Fpz).
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
for p in [CURRENT_DIR, CURRENT_DIR / "src", CURRENT_DIR.parent / "src"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import argparse
import time
import math
import json
import numpy as np
import cv2
import pygame
from PIL import Image
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import threading
from multiprocessing.connection import Client

from neuro_heterarchy_core import HeterarchicalBrainEngine, DEVICE
from tbp.monty.cmp import Message
from synthetic_16d_causal_agent import SyntheticAutonomousAgent, CorticalMontage

WIDTH, HEIGHT = 1800, 960
MAX_CONCEPTS_CAPACITY = 16
FEIGENBAUM_DELTA = 4.669201609

ALL_NAMES = ["ГОРА",  "ДЖУНГЛИ", "ЗАМОК", "ОКЕАН", "КОСМОС", "ПЛАНЕТА", "КИБЕРПАНК", "НЕБОСКРЕБ"]
FRACTAL_COLORS = [
    (20, 20, 60), (30, 80, 120), (100, 100, 110), (150, 100, 50),
    (40, 100, 60), (80, 40, 80), (10, 10, 20), (20, 80, 40),
    (120, 40, 40), (40, 120, 120), (80, 80, 30), (30, 30, 90),
    (90, 40, 90), (40, 90, 40), (110, 70, 30), (50, 50, 50)
]

ELECTRODE_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14,
                        -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], dtype=np.float32)
ELECTRODE_Y = np.array([-2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42, -2.73,
                         2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], dtype=np.float32)

def calculate_emergent_treemap(x, y, w, h, active_weights, k_score, lead_sign, count):
    valid_weights = active_weights[:count]
    active_indices = [i for i, val in enumerate(valid_weights) if val > 0.02]
    if not active_indices:
        top_i = int(np.argmax(valid_weights)) if count > 0 else 0
        return {top_i: (x, y, w, h, 0)}, [top_i]

    sorted_by_weight = sorted(active_indices, key=lambda idx: valid_weights[idx], reverse=True)
    if lead_sign < 0:
        sorted_by_weight = sorted_by_weight[::-1]

    tau = float(np.clip((k_score - 1.2) / 2.0, 0.0, 1.0))
    boxes = {}

    flat_boxes = {}
    nw = w / max(1, len(sorted_by_weight))
    for idx, c_idx in enumerate(sorted_by_weight):
        flat_boxes[c_idx] = (x + idx * nw, y, nw, h)

    nest_boxes = {}
    if len(sorted_by_weight) == 1:
        nest_boxes[sorted_by_weight[0]] = (x, y, w, h)
    else:
        root_idx = sorted_by_weight[0]
        nest_boxes[root_idx] = (x, y, w, h)
        margin_x = w * 0.06
        margin_y = h * 0.08
        inner_x = x + margin_x
        inner_y = y + margin_y * 1.5
        inner_w = w - 2 * margin_x
        inner_h = h - margin_y * 2.0
        children = sorted_by_weight[1:]
        child_w = inner_w / max(1, len(children))
        for i, ch_idx in enumerate(children):
            nest_boxes[ch_idx] = (inner_x + i * child_w, inner_y, child_w, inner_h)

    for rank, c_idx in enumerate(sorted_by_weight):
        fx, fy, fw, fh = flat_boxes[c_idx]
        nx, ny, nw_b, nh = nest_boxes[c_idx]
        cur_x = fx * (1 - tau) + nx * tau
        cur_y = fy * (1 - tau) + ny * tau
        cur_w = fw * (1 - tau) + nw_b * tau
        cur_h = fh * (1 - tau) + nh * tau
        boxes[c_idx] = (int(cur_x), int(cur_y), int(max(15, cur_w)), int(max(15, cur_h)), rank)

    return boxes, sorted_by_weight

def apply_color_surgery(img_np, old_f32):
    res = img_np.astype(np.float32)
    mu = np.mean(res, axis=(0, 1))
    target_g = (mu[0] + mu[2]) / 2.0
    if mu[1] > target_g: res[:, :, 1] -= (mu[1] - target_g) * 1.0
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

class CanonicalHTMColumn(nn.Module):
    def __init__(self, node_id: str = "Node", num_columns: int = 4096, k_active: int = 80):
        super().__init__()
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
        if torch.max(x_clean) < 1e-4: return torch.zeros(self.num_columns, device=DEVICE)
        phase_weights = torch.linspace(0.6, 1.4, 32, device=DEVICE).unsqueeze(1)
        integrated_edges = torch.sum(x_clean * phase_weights, dim=0) / 32.0
        synapse_counts = torch.sum(connected, dim=1).clamp(min=1.0)
        overlap = torch.mv(connected, integrated_edges) / synapse_counts
        k = min(self.k_active, overlap.shape[0])
        _, active_indices = torch.topk(overlap, k)
        sdr = torch.zeros(self.num_columns, device=DEVICE)
        sdr[active_indices] = 1.0
        return sdr

class FrontalExecutiveHeterarchy(nn.Module):
    def __init__(self, max_capacity: int = MAX_CONCEPTS_CAPACITY, num_columns_per_node: int = 4096, k_active_per_node: int = 80):
        super().__init__()
        self.max_capacity = max_capacity
        self.total_dim = num_columns_per_node * 4
        self.total_k_active = k_active_per_node * 4 
        self.nodes = nn.ModuleList([
            CanonicalHTMColumn(node_id=f"Node_{i}", num_columns=num_columns_per_node, k_active=k_active_per_node)
            for i in range(4)
        ])
        self.register_buffer("calcium_trace", torch.zeros(self.total_dim, device=DEVICE))
        self.register_buffer("synaptic_weights", torch.zeros((max_capacity, self.total_dim), device=DEVICE))
        self.register_buffer("membrane_potential", torch.zeros(max_capacity, device=DEVICE))

    def get_current_sdr(self, iplv_gamma_nodes: list[torch.Tensor]):
        num_to_process = min(4, len(iplv_gamma_nodes))
        sdrs = [self.nodes[i].compute_sdr(iplv_gamma_nodes[i]) for i in range(num_to_process)]
        while len(sdrs) < 4:
            sdrs.append(torch.zeros(self.nodes[0].num_columns, device=DEVICE))
        return torch.cat(sdrs, dim=0), sdrs[0]

    def reset_calcium(self):
        self.calcium_trace.zero_()

    def stream_learn_accumulate(self, concept_idx: int, cur_sdr: torch.Tensor, lr: float = 0.05):
        if torch.max(self.calcium_trace) > 1e-4:
            self.synaptic_weights[concept_idx] += lr * self.calcium_trace
            self.synaptic_weights[concept_idx] = torch.clamp(self.synaptic_weights[concept_idx], 0.0, 1.0)

    def inherit_synapses(self, parent_idx: int, child_idx: int):
        """Митоз памяти: дочерний концепт наследует 85% синапсов родителя"""
        self.synaptic_weights[child_idx] = self.synaptic_weights[parent_idx] * 0.85

    def get_ltm_scores(self, count: int) -> np.ndarray:
        strong_synapses = torch.sum(self.synaptic_weights[:count] > 0.5, dim=1).float()
        return (torch.clamp(strong_synapses / float(self.total_k_active), 0.0, 1.0) * 100.0).cpu().numpy()

    def predict_evidence(self, cur_sdr: torch.Tensor, active_count: int, dt: float = 0.016, tau: float = 0.250):
        self.calcium_trace = torch.max(self.calcium_trace * 0.9, cur_sdr)
        ltm_scores = self.get_ltm_scores(active_count)

        if torch.max(self.calcium_trace) < 1e-4:
            self.membrane_potential[:active_count] = self.membrane_potential[:active_count] * (1.0 - dt / tau)
            wm_scores = torch.zeros(active_count, device=DEVICE)
            weights = np.zeros(active_count, dtype=np.float32)
            if active_count > 0: weights[0] = 1.0
            return weights, wm_scores.cpu().numpy(), ltm_scores

        w_norm = torch.nn.functional.normalize(self.synaptic_weights[:active_count], p=2, dim=1)
        s_norm = torch.nn.functional.normalize(self.calcium_trace, p=2, dim=0)
        current = torch.mv(w_norm, s_norm)
        alpha = dt / tau
        self.membrane_potential[:active_count] = (1.0 - alpha) * self.membrane_potential[:active_count] + alpha * current
        wm_scores = torch.clamp(self.membrane_potential[:active_count], 0.0, 1.0) * 100.0
        weights = torch.softmax(self.membrane_potential[:active_count] * 10.0, dim=0).cpu().numpy()
        return weights, wm_scores.cpu().numpy(), ltm_scores

    def save_to_file(self, filepath: str, concept_names: list):
        count = len(concept_names)
        payload = {
            'format_version': '6.0',
            'num_concepts': count,
            'concept_names': concept_names,
            'synaptic_weights': self.synaptic_weights[:count].cpu(),
            'timestamp': time.time()
        }
        torch.save(payload, filepath)
        print(f"💾 [LTM PERSISTENCE] Сохранено в {filepath} ({count} концептов)")

    def load_from_file(self, filepath: str) -> tuple[bool, int]:
        if not os.path.exists(filepath): return False, 0
        try:
            ckpt = torch.load(filepath, map_location=DEVICE, weights_only=True)
            cnt = min(self.max_capacity, ckpt.get('num_concepts', 0))
            self.synaptic_weights[:cnt].copy_(ckpt['synaptic_weights'][:cnt].to(DEVICE))
            print(f"📂 [LTM VERIFIED] Загружен банк весов: {filepath} ({cnt} концептов).")
            return True, cnt
        except Exception as e:
            print(f"⚠️ [LTM LOAD ERROR]: {e}")
            return False, 0

class VisualCLIPTeacher:
    def __init__(self, text_prompts):
        model_id = "openai/clip-vit-large-patch14"
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.update_prompts(text_prompts)

    def update_prompts(self, text_prompts):
        with torch.no_grad():
            inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True).to(DEVICE)
            feat = self.model.get_text_features(**inputs)
            self.text_features = feat / feat.norm(dim=-1, keepdim=True)

    def classify(self, rgb_image_np: np.ndarray, count: int) -> np.ndarray:
        if rgb_image_np is None or np.max(rgb_image_np) == 0 or count == 0:
            return np.zeros(count, dtype=np.float32)
        pil_img = Image.fromarray(rgb_image_np)
        inputs = self.processor(images=pil_img, return_tensors="pt").to(DEVICE)
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        with torch.no_grad():
            img_feat = self.model.get_image_features(**inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            active_tf = self.text_features[:count]
            logits = (img_feat @ active_tf.T) * 30.0
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return probs

# =====================================================================
# ЕДИНЫЙ ПУЛ РАВНОПРАВНЫХ КОНЦЕПТОВ (EQUAL POOL WORKER)
# =====================================================================
class EqualPoolChaosWorker:
    def __init__(self, initial_prompts: list, initial_names: list, port: int = 6000, mode: str = "lcm", speed: str = "fast", use_taesd: bool = True, use_color: bool = True):
        self.conn = None
        self.speed = speed
        self.use_taesd = use_taesd
        self.use_color = use_color
        self.img_w, self.img_h = (448, 336) if speed == "fast" else (512, 384)
        self.current_rgb = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        self.initialized = False
        self.lock = threading.Lock()
        self.running = True
        self.fps = 0.0
        self.strength = 0.50
        self.frame_id = 0
        self.mode = mode
        
        self.prompts = list(initial_prompts)
        self.names = list(initial_names)
        self.active_count = len(self.prompts)
        
        # Единый пул тензоров в GPU
        self.c_bases = []
        self.pooled_bases = []
        
        # Метаданные дерева хаоса для каждого равноправного слота
        self.depths = [0] * MAX_CONCEPTS_CAPACITY
        self.energies = [0.0] * MAX_CONCEPTS_CAPACITY
        
        # Управление режимом хаоса
        self.chaos_enabled = False
        self.latent_active = None
        self.pooled_active = None
        self.is_sdxl = False
        
        self.dir_u = None
        self.dir_v = None
        self.last_split_time = time.time()
        
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def toggle_chaos(self):
        with self.lock:
            self.chaos_enabled = not self.chaos_enabled
            status = "ВКЛЮЧЕН" if self.chaos_enabled else "ВЫКЛЮЧЕН"
            print(f"\n🌀 [SPACE TRIGGER] Режим Хаотической Гетерархии: {status}")

    def bifurcate_concept(self, parent_idx: int, torus_u: float, torus_v: float, lead_sign: float):
        """Создает новый равноправный концепт через бифуркацию (Feigenbaum δ)"""
        if self.active_count >= MAX_CONCEPTS_CAPACITY: return -1
        
        new_idx = self.active_count
        self.active_count += 1
        
        p_depth = self.depths[parent_idx]
        self.depths[new_idx] = p_depth + 1
        self.energies[parent_idx] = 0.0
        self.energies[new_idx] = 0.0
        
        # Смещение по Тору Джанаты (азимут)
        drift_scale = 0.55 / (1.35 ** p_depth)
        drift = (self.dir_u * math.cos(torus_u) + self.dir_v * math.sin(torus_v)) * drift_scale
        
        parent_base = self.c_bases[parent_idx].clone()
        child_base = parent_base + drift
        child_base = child_base / torch.norm(child_base, dim=-1, keepdim=True) * torch.norm(parent_base, dim=-1, keepdim=True)
        
        self.c_bases.append(child_base)
        
        # Присваиваем равноправное имя
        base_parent_name = self.names[parent_idx].split(".")[0]
        direction_tag = "in" if lead_sign >= 0 else "out"
        new_name = f"{base_parent_name}.{self.depths[new_idx]}{direction_tag}"
        self.names.append(new_name)
        
        print(f"🌿 [BIFURCATION] {self.names[parent_idx]} разделился! Рожден: [{new_name}] (Всего концептов: {self.active_count})")
        return new_idx

    def update_cycle(self, leader_idx: int, child_idx: int, beta_f3: float, beta_f4: float, 
                     torus_u: float, torus_v: float, smooth_depth: float, lead_sign: float, 
                     rx_sagitta: float, heterarchy: FrontalExecutiveHeterarchy):
        if not self.initialized or len(self.c_bases) == 0: return

        with torch.inference_mode():
            # -------------------------------------------------------------
            # 1. ДИНАМИКА ХАОСА (АКТИВНА ПРИ НАЖАТИИ ПРОБЕЛА ИЛИ --chaos)
            # -------------------------------------------------------------
            if self.chaos_enabled:
                chaos_drive = float(np.clip(1.0 - beta_f4, 0.0, 1.0))
                order_drive = float(np.clip(beta_f3, 0.0, 1.0))
                
                # Безопасный индекс для накопления энергии
                safe_energy_idx = max(0, min(leader_idx, len(self.c_bases) - 1))
                
                # Накопление свободной энергии хаоса для ведущего концепта
                self.energies[safe_energy_idx] = np.clip(self.energies[safe_energy_idx] + chaos_drive * 0.07 - order_drive * 0.03, 0.0, 5.0)
                split_threshold = 0.70 / (FEIGENBAUM_DELTA ** self.depths[safe_energy_idx])

                # БИФУРКАЦИЯ (SPLITTING)
                if self.energies[safe_energy_idx] > split_threshold and (time.time() - self.last_split_time > 2.0):
                    new_child_idx = self.bifurcate_concept(safe_energy_idx, torus_u, torus_v, lead_sign)
                    if new_child_idx != -1:
                        self.last_split_time = time.time()
                        heterarchy.inherit_synapses(safe_energy_idx, new_child_idx)

                # СХЛОПЫВАНИЕ (INTERIOR CRISIS): резкий рост Бета-порядка
                if order_drive > 0.80 and self.active_count > 2 and (time.time() - self.last_split_time > 3.0):
                    removed_name = self.names.pop()
                    self.c_bases.pop()
                    self.active_count = len(self.c_bases)
                    self.last_split_time = time.time()
                    print(f"🕳️ [CRISIS] Порядок подавил шум: [{removed_name}] схлопнут обратно. (Осталось: {self.active_count})")

            # -------------------------------------------------------------
            # 2. ЖЕСТКАЯ ЗАЩИТА ИНДЕКСОВ ПОСЛЕ ЛЮБЫХ БИФУРКАЦИЙ / СХЛОПЫВАНИЙ
            # -------------------------------------------------------------
            current_len = len(self.c_bases)
            if current_len == 0: return

            l_idx = max(0, min(leader_idx, current_len - 1))
            c_idx = max(0, min(child_idx, current_len - 1))

            # -------------------------------------------------------------
            # 3. КОМПОЗИЦИЯ ДВУХ РАВНОПРАВНЫХ КОНЦЕПТОВ (F3 + F4)
            # -------------------------------------------------------------
            base_form = self.c_bases[l_idx]
            base_style = self.c_bases[c_idx]

            target = torch.zeros_like(base_form)
            half_d = target.shape[-1] // 2
            target[:, :half_d] = base_form[:, :half_d]
            target[:, half_d:] = base_style[:, half_d:]

            alpha_f3 = float(np.clip(1.0 - beta_f3 * 0.7 + rx_sagitta * 0.2, 0.15, 0.85))
            alpha_f4 = float(np.clip(1.0 - beta_f4 * 0.7 - rx_sagitta * 0.2, 0.15, 0.85))
            mean_alpha = (alpha_f3 + alpha_f4) / 2.0

            with self.lock:
                if self.latent_active is None:
                    self.latent_active = target.clone()
                else:
                    self.latent_active = self.latent_active * (1.0 - mean_alpha) + target * mean_alpha

    def _loop(self):
        times = []
        while self.running:
            if not self.initialized:
                try:
                    self.conn = Client(('localhost', 6000), authkey=b'brain')
                    self.conn.send({'cmd': 'init_mode', 'mode': self.mode, 'use_taesd': self.use_taesd})
                    init_ack = self.conn.recv()
                    self.is_sdxl = init_ack.get('is_sdxl', False)
                    
                    self.conn.send({'cmd': 'encode_base_prompts', 'prompts': self.prompts})
                    enc_resp = self.conn.recv()
                    
                    self.c_bases = [torch.tensor(b, dtype=torch.float32, device=DEVICE) for b in enc_resp['c_bases']]
                    
                    # Ортогональные оси для сдвигов Тора
                    gen = torch.Generator(device=DEVICE).manual_seed(42)
                    u = torch.randn_like(self.c_bases[0], generator=gen)
                    v = torch.randn_like(self.c_bases[0], generator=gen)
                    self.dir_u = u / torch.norm(u, dim=-1, keepdim=True)
                    self.dir_v = v / torch.norm(v, dim=-1, keepdim=True)
                    
                    self.initialized = True
                    print(f"✅ [EQUAL POOL] Подключен. Активно концептов: {self.active_count}.")
                except Exception:
                    time.sleep(0.5)
                    continue

            with self.lock:
                latent = self.latent_active.clone().cpu().numpy() if self.latent_active is not None else None
                img = self.current_rgb.copy()
                s_val = self.strength

            if latent is None:
                time.sleep(0.04)
                continue

            try:
                t0 = time.time()
                req = {
                    'cmd': 'generate',
                    'image_np': img,
                    'prompt_embeds': latent,
                    'strength': max(0.10, min(0.99, s_val))
                }
                if self.speed == "fast": req['num_inference_steps'] = 3
                self.conn.send(req)
                resp = self.conn.recv()

                with self.lock:
                    if isinstance(resp, np.ndarray):
                        if resp.shape[:2] != (self.img_h, self.img_w):
                            resp = cv2.resize(resp, (self.img_w, self.img_h))
                        if self.use_color:
                            self.current_rgb = apply_color_surgery(resp, img.astype(np.float32))
                        else:
                            self.current_rgb = resp
                        self.frame_id += 1

                times.append(time.time() - t0)
                if len(times) > 5: times.pop(0)
                self.fps = 1.0 / (np.mean(times) + 1e-6)
            except Exception:
                self.initialized = False
                time.sleep(0.5)

def main():
    parser = argparse.ArgumentParser(description="NeuroCanvas × tbp.monty: Equal-Pool Heterarchy")
    parser.add_argument('--config', type=str, default="swarm_config.json")
    parser.add_argument('--sim', action='store_true', default=False)
    
    # Режимы концептов: 0 = старт с 1 концепта и самообучение в открытом мире
    parser.add_argument('--concepts', type=int, default=4, help="Number of starting concepts (0 for infinite open-world)")
    parser.add_argument('--start-prompt', type=str, default="ancient medieval stone castle fortress towers, daytime, sharp focus, 8k", 
                        help="Initial prompt when starting with 0 concepts")
    parser.add_argument('--chaos', action='store_true', help="Start with chaos bifurcation active immediately")
    
    parser.add_argument('--mode', type=str, default="lcm", choices=["lcm", "turbo", "sdxl-turbo"])
    parser.add_argument('--speed', type=str, default="fast", choices=["fast", "quality"])
    parser.add_argument('--strength-high', type=float, default=0.85)
    parser.add_argument('--strength-low', type=float, default=0.60)#0.50
    parser.add_argument('--gamma-100', action='store_true')
    parser.add_argument('--use-kinematics', action='store_true')
    
    parser.add_argument('--hardcoded-bots', type=int, default=1)
    parser.add_argument('--jepa-bots', type=int, default=0)
    parser.add_argument('--weights', type=str, default="monty_ltm_weights.pt")
    parser.add_argument('--force-recalib', action='store_true')
    parser.add_argument('--no-taesd', action='store_true')
    parser.add_argument('--no-color', action='store_true')
    parser.add_argument('--sps', type=int, default=250, choices=[250, 500])
    args = parser.parse_args()

    # -------------------------------------------------------------
    # ИНИЦИАЛИЗАЦИЯ НАЧАЛЬНОГО ПУЛА
    # -------------------------------------------------------------
    if args.concepts == 0:
        # Режим 0 концептов: стартуем с 1 концепта из --start-prompt
        initial_prompts = [args.start_prompt]
        initial_names = ["МИР_0"]
        is_calibrating = False # В открытом мире не ждем калибровки по списку
    else:
        BASE_PROMPTS = [
            "giant snowy mountain peak, rocky cliffs, clear blue sky, sharp focus, 8k",
            "dense lush green tropical jungle, giant trees, vines, sunlight piercing through leaves, 8k",
            "ancient medieval stone castle fortress towers, daytime, sharp focus, 8k",
            "open stormy dark blue ocean, pure water surface, giant ocean waves, sea foam, no land, 8k",
            "deep outer space, glowing colorful nebula, bright stars, galaxy, 8k, sharp detailed",
            "spherical alien planet with atmosphere, continents and oceans in space, 8k, sharp detailed",
            "futuristic cyberpunk city street, neon lights, rain, glowing signs, sharp linework, 8k",
            "modern glass skyscraper buildings, downtown city, geometric architecture, sharp focus, 8k"
        ]
        initial_prompts = [BASE_PROMPTS[i % 8] for i in range(args.concepts)]
        initial_names = [ALL_NAMES[i % 8] for i in range(args.concepts)]
        is_calibrating = not args.chaos

    config_dict = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f: config_dict = json.load(f)

    montage = CorticalMontage(config_dict)
    f3_nodes = montage.get_nodes_by_names(["F3"])
    f4_nodes = montage.get_nodes_by_names(["F4"])
    afz_nodes = montage.get_nodes_by_names(["AFz"])
    fpz_nodes = montage.get_nodes_by_names(["Fpz"])

    f3_idx = f3_nodes[0] if f3_nodes else 0
    f4_idx = f4_nodes[0] if f4_nodes else 1
    afz_idx = afz_nodes[0] if afz_nodes else 2
    fpz_idx = fpz_nodes[0] if fpz_nodes else 3

    agent = None
    if args.sim:
        agent = SyntheticAutonomousAgent(
            config_path=args.config, num_hardcoded=args.hardcoded_bots, num_jepa=args.jepa_bots, 
            num_concepts=max(2, len(initial_names)), sps=args.sps
        )

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"NeuroCanvas × tbp.monty [{args.mode.upper()}] (Equal-Pool Heterarchy)")
    clock = pygame.time.Clock()
    font_b = pygame.font.SysFont("consolas", 13, bold=True)
    font_s = pygame.font.SysFont("consolas", 11)

    gamma_max = 100.0 if args.gamma_100 else 65.0
    engine = HeterarchicalBrainEngine(gamma_max=gamma_max)
    engine.start()

    if agent:
        while not agent.is_ready(): time.sleep(0.05)
        start_wait = time.time()
        while engine.shm['num_live'].value < 1 and (time.time() - start_wait < 10.0): time.sleep(0.05)

    # Инициализация воркера с единым пулом
    worker = EqualPoolChaosWorker(
        initial_prompts=initial_prompts, initial_names=initial_names,
        port=6000, mode=args.mode, speed=args.speed, use_taesd=not args.no_taesd, use_color=not args.no_color
    )
    if args.chaos: worker.chaos_enabled = True

    heterarchy = FrontalExecutiveHeterarchy(max_capacity=MAX_CONCEPTS_CAPACITY).to(DEVICE)
    clip_teacher = VisualCLIPTeacher(initial_prompts)

    learn_idx = 0
    if not args.force_recalib and os.path.exists(args.weights) and not args.chaos and args.concepts > 0:
        loaded, cnt = heterarchy.load_from_file(args.weights)
        if loaded and cnt == args.concepts:
            is_calibrating = False
            if agent: agent.set_calibration_target(False)

    svd_spectrum = np.zeros(4)
    lead_causal_sign = 0.0
    cur_probs = np.zeros(MAX_CONCEPTS_CAPACITY, dtype=np.float32)
    clip_lock = threading.Lock()

    def async_clip_worker():
        nonlocal cur_probs
        last_fid = -1
        while True:
            try:
                with worker.lock:
                    img_to_eval = worker.current_rgb.copy()
                    c_fid = worker.frame_id
                    c_cnt = worker.active_count
                if c_fid != last_fid and np.max(img_to_eval) > 0:
                    last_fid = c_fid
                    probs = clip_teacher.classify(img_to_eval, c_cnt)
                    with clip_lock: cur_probs[:c_cnt] = probs
                    if agent: agent.update_visual_state(probs)
                time.sleep(0.05)
            except Exception: time.sleep(0.1)

    threading.Thread(target=async_clip_worker, daemon=True).start()

    try:
        while True:
            dt = clock.tick(60) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    # РОУТИНГ ДЕВАЙСА
                    if event.key == pygame.K_1: f3_idx, f4_idx, afz_idx, fpz_idx = 0, 1, 2, 3
                    elif event.key == pygame.K_2: f3_idx, f4_idx, afz_idx, fpz_idx = 1, 0, 2, 3
                    elif event.key == pygame.K_3: f3_idx, f4_idx, afz_idx, fpz_idx = 1, 2, 0, 3
                    elif event.key == pygame.K_4: f3_idx, f4_idx, afz_idx, fpz_idx = 1, 2, 3, 0
                    # ПЕРЕКЛЮЧЕНИЕ ХАОСА НА ПРОБЕЛ!
                    elif event.key == pygame.K_SPACE:
                        worker.toggle_chaos()

            frame = engine.get_frame()
            has_live_eeg = (frame.num_live > 0)
            node_gamma_tensors = [torch.tensor(n.iplv_gamma, dtype=torch.float32, device=DEVICE) for n in frame.nodes]
            
            with torch.no_grad():
                full_sdr, sdr_f3 = heterarchy.get_current_sdr(node_gamma_tensors)
                afz_ripple = torch.tensor(frame.nodes[afz_idx].iplv_human_ripple, dtype=torch.float32, device=DEVICE)
                ripple_centered = afz_ripple - torch.mean(afz_ripple, dim=0, keepdim=True)
                if torch.sum(torch.abs(ripple_centered)) > 1e-4:
                    S_vals = torch.linalg.svdvals(ripple_centered)
                    S_norm = (S_vals[:4] / (S_vals[0] + 1e-6)).cpu().numpy()
                    svd_spectrum = S_norm
                    live_depth = float(np.sum(S_norm > 0.22))
                    smooth_depth = max(1.0, min(4.0, live_depth))
                else:
                    smooth_depth = 1.0
                    svd_spectrum = np.zeros(4)
                
                lead_causal_sign = float(np.mean(frame.nodes[afz_idx].iplv_human_ripple[:, 0]))

            with worker.lock: 
                rgb_m = worker.current_rgb.copy()
                active_pool_size = worker.active_count
                pool_names = list(worker.names)

            with clip_lock:   
                live_probs = cur_probs[:active_pool_size].copy()

            # ПРЕДСКАЗАНИЕ РАБОЧЕЙ ПАМЯТИ ПО ВСЕМ РАВНОПРАВНЫМ СЛОТАМ
            with torch.no_grad():
                live_human_w, wm_scores, ltm_scores = heterarchy.predict_evidence(full_sdr, active_pool_size, dt=dt)

            node_f3  = frame.nodes[f3_idx]
            node_f4  = frame.nodes[f4_idx]
            node_afz = frame.nodes[afz_idx]
            node_fpz = frame.nodes[fpz_idx]

            rx_sagitta = node_f3.gamepad_axes.rx if args.use_kinematics else 0.0
            ry_temp_bias = node_f3.gamepad_axes.ry if args.use_kinematics else 0.0

            if is_calibrating:
                target_sim = np.zeros(active_pool_size, dtype=np.float32)
                target_sim[learn_idx] = 1.0
                worker.update_cycle(learn_idx, learn_idx, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, heterarchy)
                worker.strength = args.strength_high 
                if agent: agent.set_calibration_target(True, learn_idx)

                vis_conf = float(live_probs[learn_idx]) if learn_idx < len(live_probs) else 0.0
                if vis_conf >= 0.20 and has_live_eeg:
                    heterarchy.stream_learn_accumulate(learn_idx, full_sdr, lr=0.04)

                if ltm_scores[learn_idx] >= 75.0:
                    heterarchy.save_to_file(args.weights, pool_names)
                    heterarchy.reset_calcium()
                    untrained = [i for i in range(active_pool_size) if ltm_scores[i] < 75.0]
                    if untrained: learn_idx = untrained[0]
                    else:
                        is_calibrating = False
                        if agent: agent.set_calibration_target(False)
                blended_weights = target_sim
            else:
                # Определение лидера и ведомого среди ВСЕХ концептов пула
                sorted_concepts = np.argsort(wm_scores)[::-1]
                leader_idx  = sorted_concepts[0] if len(sorted_concepts) > 0 else 0
                child_idx  = sorted_concepts[1] if len(sorted_concepts) > 1 else leader_idx

                worker.update_cycle(
                    leader_idx=leader_idx, child_idx=child_idx,
                    beta_f3=node_f3.beta_power, beta_f4=node_f4.beta_power,
                    torus_u=node_afz.torus_u, torus_v=node_afz.torus_v,
                    smooth_depth=smooth_depth, lead_sign=lead_causal_sign,
                    rx_sagitta=rx_sagitta, heterarchy=heterarchy
                )

                # 1. Считываем Бету с реально подключенного датчика (живой канал 0)
                # Если девайсов несколько - берем среднее между F3 и AFz
                if frame.num_live == 1:
                    live_beta = float(frame.nodes[0].beta_power)
                else:
                    live_beta = float((node_f3.beta_power + node_afz.beta_power) / 2.0)

                # 2. Плавная модуляция силы (Precision Weighting по Карлу Фристону)
                # Чем ниже Бета (глубже расслабление/десинхронизация), тем выше сила диффузии
                # active_drive меняется плавно от 0.0 до 1.0
                active_drive = float(np.clip(1.0 - live_beta, 0.0, 1.0))
                
                # Плавная интерполяция между strength_low и strength_high
                base_strength = args.strength_low + (args.strength_high - args.strength_low) * active_drive

                # 3. Учет кинематики (ry - временной наклон)
                if args.use_kinematics:
                    worker.strength = float(np.clip(base_strength + ry_temp_bias * 0.15, 0.10, 0.99))
                else:
                    worker.strength = float(np.clip(base_strength, 0.10, 0.99))

                active_mask = (wm_scores > 15.0)
                if np.any(active_mask):
                    blended_weights = wm_scores * active_mask
                    blended_weights /= np.sum(blended_weights)
                else:
                    blended_weights = np.zeros(active_pool_size, dtype=np.float32)
                    blended_weights[leader_idx] = 1.0

            # -------------------------------------------------------------
            # РЕНДЕРИНГ ИНТЕРФЕЙСА
            # -------------------------------------------------------------
            screen.fill((10, 14, 20))

            img_x, img_y = 370, 40
            surf_diff = pygame.image.frombuffer(rgb_m.tobytes(), (worker.img_w, worker.img_h), 'RGB')
            if (worker.img_w, worker.img_h) != (512, 384):
                surf_diff = pygame.transform.scale(surf_diff, (512, 384))
            screen.blit(surf_diff, (img_x, img_y))
            pygame.draw.rect(screen, (40, 50, 70), (img_x, img_y, 512, 384), 2, border_radius=8)

            cur_sdr_img = sdr_f3[:4096].view(64, 64).cpu().numpy() * 255.0
            sdr_surf = pygame.surfarray.make_surface(cv2.resize(cur_sdr_img, (140, 140)).astype(np.uint8))
            screen.blit(sdr_surf, (img_x, 445))
            screen.blit(font_s.render(f"L4 F3 Sheet (30–{int(gamma_max)} Гц)", True, (0, 255, 200)), (img_x, 428))

            dbg_x, dbg_y, dbg_w, dbg_h = img_x + 160, 430, 400, 170
            pygame.draw.rect(screen, (14, 18, 26), (dbg_x, dbg_y, dbg_w, dbg_h), border_radius=6)
            pygame.draw.rect(screen, (40, 70, 100), (dbg_x, dbg_y, dbg_w, dbg_h), 1, border_radius=6)
            
            p_title = f"ACTIVE WORKING MEMORY POOL ({active_pool_size}/{MAX_CONCEPTS_CAPACITY}):"
            screen.blit(font_b.render(p_title, True, (0, 255, 200)), (dbg_x + 10, dbg_y + 8))

            # Равноправное отображение всех слотов в пуле
            for i, name in enumerate(pool_names[:min(8, active_pool_size)]):
                score = wm_scores[i]
                col_bar = (0, 255, 180) if score > 15.0 else (80, 80, 90)
                bx = dbg_x + 10 + (i % 2) * 195
                by = dbg_y + 30 + (i // 2) * 22
                screen.blit(font_s.render(f"{name[:8]:8s}", True, (200, 200, 200)), (bx, by))
                bar_len = int((score / 100.0) * 80)
                pygame.draw.rect(screen, (30, 35, 45), (bx + 58, by + 2, 80, 10))
                if bar_len > 0: pygame.draw.rect(screen, col_bar, (bx + 58, by + 2, bar_len, 10))
                screen.blit(font_s.render(f"{score:4.1f}%", True, col_bar), (bx + 142, by))

            chaos_txt = "CHAOS BIFURCATIONS: [ACTIVE]" if worker.chaos_enabled else "CHAOS BIFURCATIONS: [OFF] (Press SPACE)"
            col_chaos = (255, 100, 255) if worker.chaos_enabled else (120, 120, 140)
            screen.blit(font_s.render(chaos_txt, True, col_chaos), (dbg_x + 10, dbg_y + 148))

            panel_x, panel_y = 20, 40
            pygame.draw.rect(screen, (16, 22, 32), (panel_x, panel_y, 330, 270), border_radius=8)

            if is_calibrating:
                pygame.draw.rect(screen, (255, 180, 50), (panel_x, panel_y, 330, 270), 1, border_radius=8)
                screen.blit(font_b.render("SYNAPTIC CALIBRATION (LTP)", True, (255, 180, 50)), (panel_x + 12, panel_y + 12))
                screen.blit(font_s.render(f"Target: [{pool_names[learn_idx]}]", True, (255, 255, 100)), (panel_x + 15, panel_y + 32))
                for i, name in enumerate(pool_names[:min(10, active_pool_size)]):
                    score = ltm_scores[i]
                    col_s = (100, 255, 100) if score >= 75.0 else ((255, 220, 50) if i == learn_idx else (160, 160, 160))
                    status = "[SAVED]" if score >= 75.0 else ("[LEARNING]" if i == learn_idx else "[QUEUED]")
                    screen.blit(font_s.render(f"{name:10s}: {score:4.1f}% {status}", True, col_s), (panel_x + 15, panel_y + 55 + i * 20))
            else:
                pygame.draw.rect(screen, (0, 255, 200), (panel_x, panel_y, 330, 270), 1, border_radius=8)
                screen.blit(font_b.render("EQUAL POOL RETENTION (LTM)", True, (0, 255, 200)), (panel_x + 12, panel_y + 12))
                for i, name in enumerate(pool_names[:min(10, active_pool_size)]):
                    score = ltm_scores[i]
                    col_s = (100, 255, 100) if score >= 75.0 else (200, 200, 220)
                    tag = f"D{worker.depths[i]}"
                    screen.blit(font_s.render(f"{name:10s}: {score:4.1f}% [{tag}]", True, col_s), (panel_x + 15, panel_y + 38 + i * 22))

            c_x, c_y = 20, 330
            pygame.draw.rect(screen, (16, 22, 32), (c_x, c_y, 330, 270), border_radius=8)
            pygame.draw.rect(screen, (100, 180, 255), (c_x, c_y, 330, 270), 1, border_radius=8)
            screen.blit(font_b.render("CORTICAL DECODER (EMERGENT TBT 2.0)", True, (100, 180, 255)), (c_x + 12, c_y + 12))

            top_idx = int(np.argmax(blended_weights)) if active_pool_size > 0 else 0
            cur_lead_name = pool_names[top_idx] if top_idx < len(pool_names) else "Unknown"
            screen.blit(font_b.render(f"Decoded Intent : [{cur_lead_name}]", True, (100, 255, 100)), (c_x + 15, c_y + 36))

            screen.blit(font_s.render(f"Janata Torus (AFz): u={node_afz.torus_u:.2f} | v={node_afz.torus_v:.2f}", True, (255, 220, 50)), (c_x + 15, c_y + 168))
            screen.blit(font_s.render(f"Lat. 768: Form[0:384]=F3 | Style[384:768]=F4", True, (150, 255, 200)), (c_x + 15, c_y + 188))
            screen.blit(font_s.render(f"Beta Gates: F3={node_f3.beta_power:.2f} | F4={node_f4.beta_power:.2f}", True, (255, 180, 180)), (c_x + 15, c_y + 208))
            screen.blit(font_s.render(f"Pacing: Theta={frame.theta_freq:.2f}Hz | Delta={frame.delta_freq:.2f}Hz", True, (200, 220, 240)), (c_x + 15, c_y + 228))
            screen.blit(font_s.render(f"Pipeline: {args.mode.upper()} ({worker.fps:.1f} FPS, Str: {worker.strength:.2f})", True, (180, 180, 220)), (c_x + 15, c_y + 248))

            BY, BH = 620, 300
            pygame.draw.rect(screen, (12, 16, 24), (20, BY, 870, BH), border_radius=8)
            pygame.draw.rect(screen, (30, 45, 65), (20, BY, 870, BH), 1, border_radius=8)
            screen.blit(font_b.render("120-EDGE DIRECTED ciPLV (65–100 Гц РИППЛЫ ДИКИ, ПИК 89.5 Гц)", True, (0, 255, 200)), (35, BY + 15))

            g120 = frame.nodes[afz_idx].iplv_human_ripple[31]
            bw = 800.0 / 120.0
            mid_line = BY + 150
            for p in range(120):
                val = g120[p]
                bh = int(abs(val) * 110)
                bx = 35 + p * bw
                col = (255, 80, 80) if val < 0 else (80, 255, 180)
                if val >= 0: pygame.draw.rect(screen, col, (bx, mid_line - bh, max(1, int(bw - 1)), bh))
                else: pygame.draw.rect(screen, col, (bx, mid_line, max(1, int(bw - 1)), bh))

            rx_p, ry_p = 920, 620
            pygame.draw.rect(screen, (20, 15, 30), (rx_p, ry_p, 840, 300), border_radius=8)
            pygame.draw.rect(screen, (255, 100, 200), (rx_p, ry_p, 840, 300), 1, border_radius=8)
            screen.blit(font_b.render("DECODED CAUSAL MANIFOLD (TBT 2.0)", True, (255, 100, 200)), (rx_p + 20, ry_p + 20))

            y_offset = ry_p + 60
            dir_str = "A ⊃ B (Вглубь / Child)" if lead_causal_sign >= 0.03 else ("B ⊃ A (Наружу / Parent)" if lead_causal_sign <= -0.03 else "A ∥ B (Вбок / Peer)")
            screen.blit(font_s.render(f"Causal Lead (90 Hz ciPLV): {lead_causal_sign:+.3f} -> {dir_str}", True, (200, 200, 220)), (rx_p + 20, y_offset))

            y_offset += 25
            screen.blit(font_s.render("SVD Спектр Рипплов 65–100 Гц:", True, (150, 150, 150)), (rx_p + 20, y_offset))
            for i, sv in enumerate(svd_spectrum):
                color = (0, 255, 180) if sv > 0.22 else (80, 80, 80)
                bh = int(sv * 35)
                pygame.draw.rect(screen, color, (rx_p + 20 + i*65, y_offset + 20 + (35 - bh), 45, bh))
                screen.blit(font_s.render(f"S{i+1}: {sv:.2f}", True, (200, 200, 200)), (rx_p + 20 + i*65, y_offset + 60))

            # -------------------------------------------------------------
            # TREEMAP: ОТОБРАЖЕНИЕ ВСЕХ РАВНОПРАВНЫХ КОНЦЕПТОВ ИЗ ПУЛА
            # -------------------------------------------------------------
            map_x, map_y = 950, 40
            map_w, map_h = 800, 560
            pygame.draw.rect(screen, (10, 10, 10), (map_x, map_y, map_w, map_h))
            pygame.draw.rect(screen, (255, 255, 255), (map_x-2, map_y-2, map_w+4, map_h+4), 2)
            
            boxes, resolved_hierarchy = calculate_emergent_treemap(
                map_x, map_y, map_w, map_h, blended_weights, smooth_depth, lead_causal_sign, active_pool_size
            )
            
            chaos_indicator = f" [CHAOS ON: δ={FEIGENBAUM_DELTA:.3f}]" if worker.chaos_enabled else ""
            treemap_title = f"EMERGENT TREEMAP (Active Pool: {active_pool_size}){chaos_indicator}"
            screen.blit(font_b.render(treemap_title, True, (255, 255, 255)), (map_x, map_y - 25))
            
            for concept_idx in resolved_hierarchy:
                if concept_idx >= active_pool_size: continue
                bx, by, bw, bh, rank = boxes[concept_idx]
                
                # Цвет берется из палитры по индексу в пуле
                c_color = FRACTAL_COLORS[concept_idx % len(FRACTAL_COLORS)]
                pygame.draw.rect(screen, c_color, (bx, by, bw, bh))
                pygame.draw.rect(screen, (220, 220, 220), (bx, by, bw, bh), 2)
                
                c_name = pool_names[concept_idx]
                txt_line1 = f"L{rank}: D{worker.depths[concept_idx]}"
                txt_line2 = f"{c_name}"
                
                surf1 = font_s.render(txt_line1, True, (255, 255, 255))
                surf2 = font_b.render(txt_line2, True, (255, 255, 255))
                
                if bw >= surf2.get_width() + 8 and bh >= 40:
                    screen.blit(surf1, (bx + 5, by + 5))
                    screen.blit(surf2, (bx + 5, by + 20))
                elif bw >= 30 and bh >= 20:
                    screen.blit(font_s.render(txt_line2[:5], True, (255, 255, 255)), (bx + 4, by + 4))

            pygame.display.flip()

    finally:
        worker.running = False
        heterarchy.save_to_file(args.weights, pool_names)
        if agent: agent.stop()
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
