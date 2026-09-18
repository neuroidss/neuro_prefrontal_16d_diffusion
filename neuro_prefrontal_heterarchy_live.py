#!/usr/bin/env python3
"""
🧠 NEUROCANVAS × TBP.MONTY: TOPOLOGICAL CONFIGURATION & SCIENTIFIC DUAL-CONTOUR
- Полная обратная совместимость по дефолтным параметрам.
- Добавлены флаги --strength-high / --strength-low (решают проблему FPS).
- Добавлен флаг --gamma-100 (переключает локальную гамму в континуум 30-100 Гц).
- Число концептов (--concepts) принимает любое значение >= 2.
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

ALL_NAMES = ["ГОРА",  "ДЖУНГЛИ", "ЗАМОК", "ОКЕАН", "КОСМОС", "ПЛАНЕТА", "КИБЕРПАНК", "НЕБОСКРЕБ"]
#ALL_NAMES = ["КОСМОС", "ПЛАНЕТА", "КИБЕРПАНК", "НЕБОСКРЕБ", "ГОРА", "ЗАМОК", "ОКЕАН", "ДЖУНГЛИ"]

ELECTRODE_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14,
                        -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], dtype=np.float32)
ELECTRODE_Y = np.array([-2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42, -2.73,
                         2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], dtype=np.float32)

def compute_empirical_mds_torus(text_features_tensor: torch.Tensor) -> np.ndarray:
    sim = torch.mm(text_features_tensor, text_features_tensor.t()).cpu().numpy()
    D = np.clip(1.0 - sim, 0.0, 2.0)
    n = D.shape[0]

    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H.dot(D ** 2).dot(H)
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    coords_2d = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0.0))
    c_min = coords_2d.min(axis=0)
    c_max = coords_2d.max(axis=0) + 1e-6
    norm_coords = (coords_2d - c_min) / (c_max - c_min)
    torus_coords = norm_coords * (2.0 * math.pi)
    return torus_coords

def calculate_emergent_treemap(x, y, w, h, active_weights, k_score, lead_sign):
    active_indices = [i for i, val in enumerate(active_weights) if val > 0.05]
    if not active_indices:
        top_i = int(np.argmax(active_weights))
        return {top_i: (x, y, w, h, 0)}, [top_i]

    sorted_by_weight = sorted(active_indices, key=lambda idx: active_weights[idx], reverse=True)
    if lead_sign < 0:
        sorted_by_weight = sorted_by_weight[::-1]

    tau = float(np.clip((k_score - 1.5) / 2.0, 0.0, 1.0))
    boxes = {}

    flat_boxes = {}
    nw = w / len(sorted_by_weight)
    for idx, c_idx in enumerate(sorted_by_weight):
        flat_boxes[c_idx] = (x + idx * nw, y, nw, h)

    nest_boxes = {}
    if len(sorted_by_weight) == 1:
        nest_boxes[sorted_by_weight[0]] = (x, y, w, h)
    else:
        root_idx = sorted_by_weight[0]
        nest_boxes[root_idx] = (x, y, w, h)
        margin_x = w * 0.08
        margin_y = h * 0.10
        inner_x = x + margin_x
        inner_y = y + margin_y * 1.5
        inner_w = w - 2 * margin_x
        inner_h = h - margin_y * 2.2
        children = sorted_by_weight[1:]
        child_w = inner_w / len(children)
        for i, ch_idx in enumerate(children):
            nest_boxes[ch_idx] = (inner_x + i * child_w, inner_y, child_w, inner_h)

    for rank, c_idx in enumerate(sorted_by_weight):
        fx, fy, fw, fh = flat_boxes[c_idx]
        nx, ny, nw_b, nh = nest_boxes[c_idx]
        cur_x = fx * (1 - tau) + nx * tau
        cur_y = fy * (1 - tau) + ny * tau
        cur_w = fw * (1 - tau) + nw_b * tau
        cur_h = fh * (1 - tau) + nh * tau
        boxes[c_idx] = (int(cur_x), int(cur_y), int(max(10, cur_w)), int(max(10, cur_h)), rank)

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
    def __init__(self, num_concepts: int = 8, num_columns_per_node: int = 4096, k_active_per_node: int = 80):
        super().__init__()
        self.num_concepts = num_concepts
        self.total_dim = num_columns_per_node * 4
        self.total_k_active = k_active_per_node * 4 
        self.nodes = nn.ModuleList([
            CanonicalHTMColumn(node_id=f"Node_{i}", num_columns=num_columns_per_node, k_active=k_active_per_node)
            for i in range(4)
        ])
        self.register_buffer("calcium_trace", torch.zeros(self.total_dim, device=DEVICE))
        self.register_buffer("synaptic_weights", torch.zeros((num_concepts, self.total_dim), device=DEVICE))
        self.register_buffer("membrane_potential", torch.zeros(num_concepts, device=DEVICE))

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

    def get_ltm_scores(self) -> np.ndarray:
        strong_synapses = torch.sum(self.synaptic_weights > 0.5, dim=1).float()
        return (torch.clamp(strong_synapses / float(self.total_k_active), 0.0, 1.0) * 100.0).cpu().numpy()

    def predict_evidence(self, cur_sdr: torch.Tensor, dt: float = 0.016, tau: float = 0.250):
        self.calcium_trace = torch.max(self.calcium_trace * 0.9, cur_sdr)
        ltm_scores = self.get_ltm_scores()

        if torch.max(self.calcium_trace) < 1e-4:
            self.membrane_potential = self.membrane_potential * (1.0 - dt / tau)
            wm_scores = torch.zeros(self.num_concepts, device=DEVICE)
            weights = np.zeros(self.num_concepts, dtype=np.float32)
            return weights, wm_scores.cpu().numpy(), ltm_scores

        w_norm = torch.nn.functional.normalize(self.synaptic_weights, p=2, dim=1)
        s_norm = torch.nn.functional.normalize(self.calcium_trace, p=2, dim=0)
        current = torch.mv(w_norm, s_norm)
        alpha = dt / tau
        self.membrane_potential = (1.0 - alpha) * self.membrane_potential + alpha * current
        wm_scores = torch.clamp(self.membrane_potential, 0.0, 1.0) * 100.0
        weights = torch.softmax(self.membrane_potential * 10.0, dim=0).cpu().numpy()
        return weights, wm_scores.cpu().numpy(), ltm_scores

    def save_to_file(self, filepath: str, concept_names: list):
        ltm_scores = self.get_ltm_scores()
        num_trained = int(np.sum(ltm_scores >= 75.0))
        payload = {
            'format_version': '5.0',
            'num_concepts': self.num_concepts,
            'concept_names': concept_names,
            'synaptic_weights': self.synaptic_weights.cpu(),
            'timestamp': time.time()
        }
        torch.save(payload, filepath)
        print(f"💾 [LTM PERSISTENCE] Сохранено в {filepath} ({num_trained}/{self.num_concepts} консолидировано >= 75%)")

    def load_from_file(self, filepath: str) -> tuple[bool, bool]:
        if not os.path.exists(filepath): return False, False
        try:
            ckpt = torch.load(filepath, map_location=DEVICE, weights_only=True)
            min_c = min(self.num_concepts, ckpt.get('num_concepts', self.num_concepts))
            self.synaptic_weights[:min_c].copy_(ckpt['synaptic_weights'][:min_c].to(DEVICE))
            ltm_scores = self.get_ltm_scores()
            num_trained = int(np.sum(ltm_scores >= 75.0))
            all_trained = (num_trained == self.num_concepts)
            print(f"📂 [LTM VERIFIED] Загружен банк весов: {filepath} ({num_trained}/{self.num_concepts} имеют вес >= 75%).")
            return True, all_trained
        except Exception as e:
            print(f"⚠️ [LTM LOAD ERROR]: {e}")
            return False, False

class VisualCLIPTeacher:
    def __init__(self, class_names, text_prompts):
        model_id = "openai/clip-vit-large-patch14"
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        with torch.no_grad():
            inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True).to(DEVICE)
            feat = self.model.get_text_features(**inputs)
            self.text_features = feat / feat.norm(dim=-1, keepdim=True)

    def classify(self, rgb_image_np: np.ndarray) -> np.ndarray:
        if rgb_image_np is None or np.max(rgb_image_np) == 0:
            return np.zeros(self.text_features.shape[0], dtype=np.float32)
        pil_img = Image.fromarray(rgb_image_np)
        inputs = self.processor(images=pil_img, return_tensors="pt").to(DEVICE)
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        with torch.no_grad():
            img_feat = self.model.get_image_features(**inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = (img_feat @ self.text_features.T) * 30.0
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return probs

class ToroidalDiffusionWorker:
    def __init__(self, prompts, port: int = 6000, mode: str = "lcm", speed: str = "fast", use_taesd: bool = True, use_color: bool = True):
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
        self.prompts = prompts
        self.num_concepts = len(prompts)
        
        self.c_bases = None
        self.pooled_bases = None
        self.is_sdxl = False
        self.latent_active = None
        self.pooled_active = None
        
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def update_hemispheric_target(self, left_form_idx: int, right_style_idx: int, beta_f3: float, beta_f4: float, rx_sagitta: float = 0.0):
        if not self.initialized or self.c_bases is None: return

        with torch.inference_mode():
            base_form = self.c_bases[left_form_idx]
            base_style = self.c_bases[right_style_idx]

            target = torch.zeros_like(base_form)
            d_dim = target.shape[-1]
            half_d = d_dim // 2

            target[:, :half_d] = base_form[:, :half_d]
            target[:, half_d:] = base_style[:, half_d:]

            alpha_form = float(np.clip(1.0 - beta_f3 * 0.7 + rx_sagitta * 0.2, 0.15, 0.85))
            alpha_style = float(np.clip(1.0 - beta_f4 * 0.7 - rx_sagitta * 0.2, 0.15, 0.85))
            mean_alpha = (alpha_form + alpha_style) / 2.0

            # ФИКС SDXL: Пулированный эмбеддинг теперь тоже смешивается и передается!
            target_pool = None
            if self.is_sdxl and self.pooled_bases is not None:
                p_form = self.pooled_bases[left_form_idx]
                p_style = self.pooled_bases[right_style_idx]
                target_pool = (p_form + p_style) * 0.5

            with self.lock:
                if self.latent_active is None:
                    self.latent_active = target.clone()
                else:
                    self.latent_active = self.latent_active * (1.0 - mean_alpha) + target * mean_alpha

                if target_pool is not None:
                    if self.pooled_active is None:
                        self.pooled_active = target_pool.clone()
                    else:
                        self.pooled_active = self.pooled_active * (1.0 - mean_alpha) + target_pool * mean_alpha

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
                    if enc_resp.get('pooled_bases') is not None:
                        self.pooled_bases = [torch.tensor(b, dtype=torch.float32, device=DEVICE) for b in enc_resp['pooled_bases']]
                    else:
                        self.pooled_bases = None
                        
                    self.initialized = True
                    print(f"✅ [SD-{self.mode.upper()}] Подключен к brain_server. TAESD: {self.use_taesd}, ColorFix: {self.use_color}.")
                except Exception:
                    time.sleep(0.5)
                    continue

            with self.lock:
                latent = self.latent_active.clone().cpu().numpy() if self.latent_active is not None else None
                pooled = self.pooled_active.clone().cpu().numpy() if self.pooled_active is not None else None
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
                if pooled is not None: req['pooled_prompt_embeds'] = pooled
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
    parser = argparse.ArgumentParser(description="NeuroCanvas × tbp.monty: Topological Config & Kinematics")
    parser.add_argument('--config', type=str, default="swarm_config.json", help="Path to config file")
    parser.add_argument('--sim', action='store_true', default=False, help="Start external agent swarm")
    
    # Гибкое количество концептов (любое int >= 2)
    parser.add_argument('--concepts', type=int, default=8, help="Number of active concepts (e.g. 2, 4, 8)")
    
    # Параметры диффузии
    parser.add_argument('--mode', type=str, default="lcm", choices=["lcm", "turbo", "sdxl-turbo"])
    parser.add_argument('--speed', type=str, default="fast", choices=["fast", "quality"])
    
    # Сила диффузии: вынесена в CLI (решает проблему 4.6 FPS)
    parser.add_argument('--strength-high', type=float, default=0.85, help="Strength when transitioning/focused")
    parser.add_argument('--strength-low', type=float, default=0.50, help="Strength during steady state")
    
    # Научные режимы спектра
    parser.add_argument('--gamma-100', action='store_true', help="Extend local gamma contour to 30-100 Hz continuum")
    parser.add_argument('--use-kinematics', action='store_true', help="Use rx for style blending and ry for dynamic strength")
    
    # Боты (если конфиг не найден)
    parser.add_argument('--hardcoded-bots', type=int, default=1)
    parser.add_argument('--jepa-bots', type=int, default=0)
    
    parser.add_argument('--weights', type=str, default="monty_ltm_weights.pt")
    parser.add_argument('--force-recalib', action='store_true')
    parser.add_argument('--no-taesd', action='store_true')
    parser.add_argument('--no-color', action='store_true')
    
    parser.add_argument('--sps', type=int, default=250, choices=[250, 500], help="Target sampling rate")
    args = parser.parse_args()

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
#    BASE_PROMPTS = [
#        "deep outer space, glowing colorful nebula, bright stars, galaxy, 8k, sharp detailed",
#        "spherical alien planet with atmosphere, continents and oceans in space, 8k, sharp detailed",
#        "futuristic cyberpunk city street, neon lights, rain, glowing signs, sharp linework, 8k",
#        "modern glass skyscraper buildings, downtown city, geometric architecture, sharp focus, 8k",
#        "giant snowy mountain peak, rocky cliffs, clear blue sky, sharp focus, 8k",
#        "ancient medieval stone castle fortress towers, daytime, sharp focus, 8k",
#        "open stormy dark blue ocean, pure water surface, giant ocean waves, sea foam, no land, 8k",
#        "dense lush green tropical jungle, giant trees, vines, sunlight piercing through leaves, 8k"
#    ]
    
    PROMPTS = []
    TARGET_NAMES = []
    for i in range(args.concepts):
        idx = i % 8
        prefix = f"Variant {i//8}: " if i >= 8 else ""
        PROMPTS.append(prefix + BASE_PROMPTS[idx])
        TARGET_NAMES.append(f"{ALL_NAMES[idx]}_{i//8}" if i >= 8 else ALL_NAMES[idx])

    config_dict = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f: 
            config_dict = json.load(f)
    else:
        print(f"⚠️ Конфиг {args.config} не найден, используем параметры из CLI.")

    montage = CorticalMontage(config_dict)
    f3_nodes = montage.get_nodes_by_names(["F3"])
    f4_nodes = montage.get_nodes_by_names(["F4"])
    afz_nodes = montage.get_nodes_by_names(["AFz"])
    fpz_nodes = montage.get_nodes_by_names(["Fpz"])

    f3_idx = f3_nodes[0] if f3_nodes else 0
    f4_idx = f4_nodes[0] if f4_nodes else 1
    afz_idx = afz_nodes[0] if afz_nodes else 2
    fpz_idx = fpz_nodes[0] if fpz_nodes else 3

    print(f"🗺️ [TOPOLOGY MAPPING UI] F3: {f3_idx}, F4: {f4_idx}, AFz: {afz_idx}, Fpz: {fpz_idx}")

    agent = None
    if args.sim:
        agent = SyntheticAutonomousAgent(
            config_path=args.config, 
            num_hardcoded=args.hardcoded_bots, 
            num_jepa=args.jepa_bots, 
            num_concepts=args.concepts,
            sps=args.sps  # Боты синхронизируются с частотой пользователя!
        )

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"NeuroCanvas × tbp.monty: Emergent Heterarchy [{args.mode.upper()}] (Concepts: {args.concepts})")
    clock = pygame.time.Clock()
    font_b = pygame.font.SysFont("consolas", 13, bold=True)
    font_s = pygame.font.SysFont("consolas", 11)

    # Инициализация ядра с выбранным верхним порогом гаммы (65 или 100 Гц)
    gamma_max = 100.0 if args.gamma_100 else 65.0
    engine = HeterarchicalBrainEngine(gamma_max=gamma_max)
    engine.start()

    if agent:
        print("⏳ Ожидание инициализации роя и LSL потоков...")
        while not agent.is_ready(): time.sleep(0.05)
        start_wait = time.time()
        while engine.shm['num_live'].value < 1 and (time.time() - start_wait < 10.0): time.sleep(0.05)

    worker = ToroidalDiffusionWorker(PROMPTS, port=6000, mode=args.mode, speed=args.speed, use_taesd=not args.no_taesd, use_color=not args.no_color)
    clip_teacher = VisualCLIPTeacher(TARGET_NAMES, PROMPTS)
    heterarchy = FrontalExecutiveHeterarchy(num_concepts=args.concepts).to(DEVICE)

    empirical_torus_coords = compute_empirical_mds_torus(clip_teacher.text_features)
    
    is_calibrating = True
    learn_idx = 0
    if not args.force_recalib and os.path.exists(args.weights):
        loaded, all_trained = heterarchy.load_from_file(args.weights)
        if loaded and all_trained:
            is_calibrating = False
            if agent: agent.set_calibration_target(False)
        else:
            curr_ltm = heterarchy.get_ltm_scores()
            untrained = [i for i in range(args.concepts) if curr_ltm[i] < 75.0]
            if untrained: learn_idx = untrained[0]

    svd_spectrum = np.zeros(4)
    lead_causal_sign = 0.0
    FRACTAL_COLORS = [(20, 20, 60), (30, 80, 120), (100, 100, 110), (150, 100, 50), (40, 100, 60), (80, 40, 80), (10, 10, 20), (20, 80, 40)]
    cur_probs = np.zeros(args.concepts, dtype=np.float32)
    clip_lock = threading.Lock()

    def async_clip_worker():
        nonlocal cur_probs
        last_fid = -1
        while True:
            try:
                with worker.lock:
                    img_to_eval = worker.current_rgb.copy()
                    c_fid = worker.frame_id
                if c_fid != last_fid and np.max(img_to_eval) > 0:
                    last_fid = c_fid
                    probs = clip_teacher.classify(img_to_eval)
                    with clip_lock: cur_probs = probs
                    if agent:
                        agent.update_visual_state(probs)
                        agent.update_visual_state(img_to_eval)
                time.sleep(0.05)
            except Exception: time.sleep(0.1)

    threading.Thread(target=async_clip_worker, daemon=True).start()

    try:
        while True:
            dt = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT: raise KeyboardInterrupt
                # ДИНАМИЧЕСКИЙ РОУТИНГ ДЕВАЙСА (Hotkeys)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        f3_idx, f4_idx, afz_idx, fpz_idx = 0, 1, 2, 3
                        print("📡 [ROUTING] Активный девайс переключен на F3 (Форма / L-dlPFC)")
                    elif event.key == pygame.K_2:
                        f3_idx, f4_idx, afz_idx, fpz_idx = 1, 0, 2, 3
                        print("📡 [ROUTING] Активный девайс переключен на F4 (Стиль / R-dlPFC)")
                    elif event.key == pygame.K_3:
                        f3_idx, f4_idx, afz_idx, fpz_idx = 1, 2, 0, 3
                        print("📡 [ROUTING] Активный девайс переключен на AFz (Тор Джанаты / dACC)")
                    elif event.key == pygame.K_4:
                        f3_idx, f4_idx, afz_idx, fpz_idx = 1, 2, 3, 0
                        print("📡 [ROUTING] Активный девайс переключен на Fpz (Когнитивное ветвление / BA10)")

            frame = engine.get_frame()
            has_live_eeg = (frame.num_live > 0)

            node_gamma_tensors = [torch.tensor(n.iplv_gamma, dtype=torch.float32, device=DEVICE) for n in frame.nodes]
            
            with torch.no_grad():
                full_sdr, sdr_f3 = heterarchy.get_current_sdr(node_gamma_tensors)

            with torch.no_grad():
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

            with worker.lock: rgb_m = worker.current_rgb.copy()
            with clip_lock:   live_probs = cur_probs.copy()

            with torch.no_grad():
                live_human_w, wm_scores, ltm_scores = heterarchy.predict_evidence(full_sdr, dt=dt)

            node_f3  = frame.nodes[f3_idx]
            node_f4  = frame.nodes[f4_idx]
            node_afz = frame.nodes[afz_idx]
            node_fpz = frame.nodes[fpz_idx]

            rx_sagitta = node_f3.gamepad_axes.rx if args.use_kinematics else 0.0
            ry_temp_bias = node_f3.gamepad_axes.ry if args.use_kinematics else 0.0

            cmp_message = Message(
                location=node_f3.disp_xyz.astype(np.float64),
                morphological_features={
                    "pose_vectors": node_f3.pose_matrix.astype(np.float64),
                    "pose_fully_defined": bool(smooth_depth >= 1.8),
                    "on_object": 1.0
                },
                non_morphological_features={
                    "object_id": int(np.argmax(wm_scores)),
                    "torus_u": node_afz.torus_u,
                    "torus_v": node_afz.torus_v,
                    "beta_f3": node_f3.beta_power,
                    "beta_f4": node_f4.beta_power
                },
                confidence=float(np.clip(node_f3.beta_stability, 0.0, 1.0)),
                pass_message=True,
                process_features_in_lm=True,
                sender_id="Prefrontal_Heterarchy_SM",
                sender_type="SM"
            )

            if is_calibrating:
                target_sim = np.zeros(args.concepts, dtype=np.float32)
                target_sim[learn_idx] = 1.0
                worker.update_hemispheric_target(learn_idx, learn_idx, 0.0, 0.0, rx_sagitta=0.0)
                worker.strength = args.strength_high 
                if agent: agent.set_calibration_target(True, learn_idx)

                vis_conf = float(live_probs[learn_idx])
                if vis_conf >= 0.25 and has_live_eeg:
                    heterarchy.stream_learn_accumulate(learn_idx, full_sdr, lr=0.04)

                if ltm_scores[learn_idx] >= 75.0:
                    heterarchy.save_to_file(args.weights, TARGET_NAMES)
                    heterarchy.reset_calcium()
                    untrained = [i for i in range(args.concepts) if ltm_scores[i] < 75.0]
                    if untrained: learn_idx = untrained[0]
                    else:
                        is_calibrating = False
                        if agent: agent.set_calibration_target(False)
                blended_weights = target_sim
            else:
                sorted_concepts = np.argsort(wm_scores)[::-1]
                leader_form_idx  = sorted_concepts[0]
                child_style_idx  = sorted_concepts[1] if len(sorted_concepts) > 1 else sorted_concepts[0]

                worker.update_hemispheric_target(
                    left_form_idx=leader_form_idx,
                    right_style_idx=child_style_idx,
                    beta_f3=node_f3.beta_power,
                    beta_f4=node_f4.beta_power,
                    rx_sagitta=rx_sagitta
                )

                base_strength = args.strength_high if (smooth_depth >= 2.5 or node_afz.beta_power < 0.3) else args.strength_low
                if args.use_kinematics:
                    worker.strength = np.clip(base_strength + ry_temp_bias * 0.15, 0.10, 0.99)
                else:
                    worker.strength = base_strength

                active_mask = (wm_scores > 20.0)
                if np.any(active_mask):
                    blended_weights = wm_scores * active_mask
                    blended_weights /= np.sum(blended_weights)
                else:
                    blended_weights = np.zeros(args.concepts, dtype=np.float32)
                    blended_weights[leader_form_idx] = 1.0

            map_x, map_y = 950, 40
            map_w, map_h = 800, 560
            boxes, resolved_hierarchy = calculate_emergent_treemap(
                map_x, map_y, map_w, map_h, blended_weights, smooth_depth, lead_causal_sign
            )
            
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
            screen.blit(font_b.render(f"LIVE WORKING MEMORY ENSEMBLES (Vm):", True, (0, 255, 200)), (dbg_x + 10, dbg_y + 8))

            for i, name in enumerate(TARGET_NAMES[:min(8, args.concepts)]):
                score = wm_scores[i]
                col_bar = (0, 255, 180) if score > 20.0 else (80, 80, 90)
                bx = dbg_x + 10 + (i % 2) * 195
                by = dbg_y + 30 + (i // 2) * 22
                screen.blit(font_s.render(f"{name[:7]:7s}", True, (200, 200, 200)), (bx, by))
                bar_len = int((score / 100.0) * 80)
                pygame.draw.rect(screen, (30, 35, 45), (bx + 55, by + 2, 80, 10))
                if bar_len > 0: pygame.draw.rect(screen, col_bar, (bx + 55, by + 2, bar_len, 10))
                screen.blit(font_s.render(f"{score:4.1f}%", True, col_bar), (bx + 140, by))

            screen.blit(font_s.render("Threshold: >20.0% Vm enters reality", True, (150, 150, 160)), (dbg_x + 10, dbg_y + 148))

            panel_x, panel_y = 20, 40
            pygame.draw.rect(screen, (16, 22, 32), (panel_x, panel_y, 330, 270), border_radius=8)

            if is_calibrating:
                pygame.draw.rect(screen, (255, 180, 50), (panel_x, panel_y, 330, 270), 1, border_radius=8)
                screen.blit(font_b.render("SYNAPTIC CALIBRATION (LTP)", True, (255, 180, 50)), (panel_x + 12, panel_y + 12))
                screen.blit(font_s.render(f"Target: [{TARGET_NAMES[learn_idx]}]", True, (255, 255, 100)), (panel_x + 15, panel_y + 32))
                for i, name in enumerate(TARGET_NAMES[:min(10, args.concepts)]):
                    score = ltm_scores[i]
                    if score >= 75.0:
                        col_s = (100, 255, 100)
                        txt_s = f"{name:10s}: {score:4.1f}% [SAVED]"
                    elif i == learn_idx:
                        col_s = (255, 220, 50)
                        txt_s = f"{name:10s}: {score:4.1f}% [LEARNING]"
                    else:
                        col_s = (160, 160, 160)
                        txt_s = f"{name:10s}: {score:4.1f}% [QUEUED]"
                    screen.blit(font_s.render(txt_s, True, col_s), (panel_x + 15, panel_y + 55 + i * 20))
            else:
                pygame.draw.rect(screen, (0, 255, 200), (panel_x, panel_y, 330, 270), 1, border_radius=8)
                screen.blit(font_b.render("FROZEN RETENTION (LTM CONSOLIDATED)", True, (0, 255, 200)), (panel_x + 12, panel_y + 12))
                for i, name in enumerate(TARGET_NAMES[:min(10, args.concepts)]):
                    score = ltm_scores[i]
                    col_s = (100, 255, 100) if score >= 75.0 else (255, 80, 80)
                    status_lbl = "[CONSOLIDATED]" if score >= 75.0 else "[UNTRAINED]"
                    screen.blit(font_s.render(f"{name:10s}: {score:4.1f}% {status_lbl}", True, col_s), (panel_x + 15, panel_y + 38 + i * 22))

            c_x, c_y = 20, 330
            pygame.draw.rect(screen, (16, 22, 32), (c_x, c_y, 330, 270), border_radius=8)
            pygame.draw.rect(screen, (100, 180, 255), (c_x, c_y, 330, 270), 1, border_radius=8)
            screen.blit(font_b.render("CORTICAL DECODER (EMERGENT TBT 2.0)", True, (100, 180, 255)), (c_x + 12, c_y + 12))

            top_idx = int(np.argmax(blended_weights))
            screen.blit(font_b.render(f"Decoded Intent : [{TARGET_NAMES[top_idx]}]", True, (100, 255, 100)), (c_x + 15, c_y + 36))

            if agent:
                status_str = agent.get_telemetry()
                screen.blit(font_b.render("Swarm Decision State:", True, (255, 200, 100)), (c_x + 15, c_y + 60))
                lines = [status_str[i:i+45] for i in range(0, len(status_str), 45)]
                for idx_l, line in enumerate(lines[:5]):
                    screen.blit(font_s.render(line, True, (200, 200, 220)), (c_x + 15, c_y + 80 + idx_l * 16))

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

            rx, ry = 920, 620
            pygame.draw.rect(screen, (20, 15, 30), (rx, ry, 840, 300), border_radius=8)
            pygame.draw.rect(screen, (255, 100, 200), (rx, ry, 840, 300), 1, border_radius=8)
            screen.blit(font_b.render("DECODED CAUSAL MANIFOLD (TBT 2.0)", True, (255, 100, 200)), (rx + 20, ry + 20))

            y_offset = ry + 60
            dir_str = "A ⊃ B (Родитель ведет)" if lead_causal_sign >= 0 else "B ⊃ A (Инверсия позы)"
            screen.blit(font_s.render(f"Causal Lead (90 Hz ciPLV): {lead_causal_sign:+.3f} -> {dir_str}", True, (200, 200, 220)), (rx + 20, y_offset))

            y_offset += 25
            screen.blit(font_s.render("SVD Спектр Рипплов 65–100 Гц:", True, (150, 150, 150)), (rx + 20, y_offset))
            for i, sv in enumerate(svd_spectrum):
                color = (0, 255, 180) if sv > 0.22 else (80, 80, 80)
                bh = int(sv * 35)
                pygame.draw.rect(screen, color, (rx + 20 + i*65, y_offset + 20 + (35 - bh), 45, bh))
                screen.blit(font_s.render(f"S{i+1}: {sv:.2f}", True, (200, 200, 200)), (rx + 20 + i*65, y_offset + 60))

            pygame.draw.rect(screen, (10, 10, 10), (map_x, map_y, map_w, map_h))
            pygame.draw.rect(screen, (255, 255, 255), (map_x-2, map_y-2, map_w+4, map_h+4), 2)
            treemap_title = f"EMERGENT TREEMAP (K={smooth_depth:.2f})"
            screen.blit(font_b.render(treemap_title, True, (255, 255, 255)), (map_x, map_y - 25))

            owners = agent.get_owners() if agent else [-1] * 8
            
            for concept_idx in resolved_hierarchy:
                if concept_idx >= len(FRACTAL_COLORS): continue
                
                bx, by, bw, bh, rank = boxes[concept_idx]
                pygame.draw.rect(screen, FRACTAL_COLORS[concept_idx % 8], (bx, by, bw, bh))
                pygame.draw.rect(screen, (220, 220, 220), (bx, by, bw, bh), 2)
                
                owner_id = owners[concept_idx] if concept_idx < len(owners) else -1
                if owner_id == 0 and frame.is_real: owner_name = "YOU"
                elif owner_id > 0: owner_name = f"Bot {owner_id}"
                else: owner_name = "CONSENSUS"

                txt_line1 = f"L{rank}: {owner_name}"
                txt_line2 = f"{TARGET_NAMES[concept_idx]}"
                surf1 = font_s.render(txt_line1, True, (255, 255, 255))
                surf2 = font_b.render(txt_line2, True, (255, 255, 255))
                
                if bw >= surf2.get_width() + 8 and bh >= 40:
                    screen.blit(surf1, (bx + 5, by + 5))
                    screen.blit(surf2, (bx + 5, by + 20))
                elif bw >= 30 and bh >= 20:
                    screen.blit(font_s.render(txt_line2[:4], True, (255, 255, 255)), (bx + 4, by + 4))

            pygame.display.flip()

    finally:
        worker.running = False
        heterarchy.save_to_file(args.weights, TARGET_NAMES)
        if agent: agent.stop()
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
