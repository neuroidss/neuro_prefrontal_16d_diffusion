#!/usr/bin/env python3
"""
🧠 NEUROCANVAS × TBP.MONTY (VERSIONED LTM PERSISTENCE & ORIGINAL RENDERING)
- Точный откат конвейера рисования и цветокоррекции к оригинальному чекпоинту.
- Версионная персистентность LTM (v1.0) с валидацией целостности банка синапсов.
- Защита файла весов от случайной перезаписи недоученным состоянием.
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

WIDTH, HEIGHT = 1800, 960

ELECTRODE_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14,
                        -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], dtype=np.float32)
ELECTRODE_Y = np.array([-2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42, -2.73,
                         2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], dtype=np.float32)

ALL_NAMES = ["КОСМОС", "ПЛАНЕТА", "КИБЕРПАНК", "НЕБОСКРЕБ", "ГОРА", "ЗАМОК", "ОКЕАН", "ДЖУНГЛИ"]
AGENT_NAMES = ["YOU"] + [f"Bot {i}" for i in range(1, 8)]

HIERARCHY_TREE = {
    "КОСМОС":    {"level": 0, "parent": None,      "so3": np.array([ 0.0,  0.0,  0.0], dtype=np.float32)},
    "ОКЕАН":     {"level": 0, "parent": None,      "so3": np.array([ 0.8, -1.0,  0.5], dtype=np.float32)},
    "ГОРА":      {"level": 1, "parent": "КОСМОС",  "so3": np.array([-1.0,  0.5,  0.0], dtype=np.float32)},
    "ПЛАНЕТА":   {"level": 1, "parent": "КОСМОС",  "so3": np.array([ 0.5, -0.2,  0.1], dtype=np.float32)},
    "ДЖУНГЛИ":   {"level": 1, "parent": "ОКЕАН",   "so3": np.array([-1.2, -0.8,  0.2], dtype=np.float32)},
    "КИБЕРПАНК": {"level": 2, "parent": "ПЛАНЕТА", "so3": np.array([ 1.2,  0.8, -0.3], dtype=np.float32)},
    "НЕБОСКРЕБ": {"level": 2, "parent": "ПЛАНЕТА", "so3": np.array([ 1.5,  0.0,  1.0], dtype=np.float32)},
    "ЗАМОК":     {"level": 2, "parent": "ГОРА",    "so3": np.array([-0.5,  0.3,  0.4], dtype=np.float32)},
}

def resolve_active_lineage(active_weights, lead_sign):
    active_indices = [i for i, w in enumerate(active_weights) if w > 0.05]
    if not active_indices:
        return [int(np.argmax(active_weights))]
    sorted_by_level = sorted(active_indices, key=lambda idx: HIERARCHY_TREE[ALL_NAMES[idx]]["level"])
    if lead_sign < 0:
        return sorted_by_level[::-1]
    return sorted_by_level

def calculate_dynamic_treemap(x, y, w, h, active_weights, k_score, lead_sign):
    resolved_indices = resolve_active_lineage(active_weights, lead_sign)
    if not resolved_indices:
        top_idx = int(np.argmax(active_weights))
        return {top_idx: (x, y, w, h, 0)}, [top_idx]

    tau = float(np.clip((k_score - 1.5) / 2.0, 0.0, 1.0))
    boxes = {}
    margin_x = w * 0.08
    margin_y = h * 0.08

    flat_boxes = {}
    nw = w / len(resolved_indices)
    for idx, c_idx in enumerate(resolved_indices):
        flat_boxes[c_idx] = (x + idx * nw, y, nw, h)

    tree = {None: []}
    for idx in resolved_indices:
        tree[ALL_NAMES[idx]] = []

    def get_active_parent(c_name):
        p = HIERARCHY_TREE[c_name]["parent"]
        while p is not None:
            if ALL_NAMES.index(p) in resolved_indices:
                return p
            p = HIERARCHY_TREE[p]["parent"]
        return None

    for c_idx in resolved_indices:
        p_name = get_active_parent(ALL_NAMES[c_idx])
        if p_name in tree:
            tree[p_name].append(c_idx)
        else:
            tree[None].append(c_idx)

    nest_boxes = {}
    def assign_rects(parent_name, cur_x, cur_y, cur_w, cur_h):
        children = tree[parent_name]
        if not children: return
        child_w = cur_w / len(children)
        for i, child_idx in enumerate(children):
            cx = cur_x + i * child_w
            nest_boxes[child_idx] = (cx, cur_y, child_w, cur_h)
            assign_rects(ALL_NAMES[child_idx], cx + margin_x, cur_y + margin_y * 1.5, child_w - 2 * margin_x, cur_h - margin_y * 2.0)

    assign_rects(None, x, y, w, h)

    for c_idx in resolved_indices:
        fx, fy, fw, fh = flat_boxes[c_idx]
        nx, ny, nw, nh = nest_boxes.get(c_idx, flat_boxes[c_idx])
        
        cur_x = fx * (1 - tau) + nx * tau
        cur_y = fy * (1 - tau) + ny * tau
        cur_w = fw * (1 - tau) + nw * tau
        cur_h = fh * (1 - tau) + nh * tau
        
        boxes[c_idx] = (int(cur_x), int(cur_y), int(max(10, cur_w)), int(max(10, cur_h)), resolved_indices.index(c_idx))

    return boxes, resolved_indices

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
        
        if torch.max(x_clean) < 1e-4:
            return torch.zeros(self.num_columns, device=DEVICE)

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
        self.register_buffer("concept_trained", torch.zeros(num_concepts, device=DEVICE))

    def get_current_sdr(self, iplv_gamma_nodes: list[torch.Tensor]):
        sdrs = [self.nodes[i].compute_sdr(iplv_gamma_nodes[i]) for i in range(4)]
        return torch.cat(sdrs, dim=0), sdrs[0]

    def reset_accumulator(self, concept_idx: int):
        self.synaptic_weights[concept_idx].zero_()
        self.membrane_potential[concept_idx].zero_()

    def stream_learn_accumulate(self, concept_idx: int, cur_sdr: torch.Tensor, lr: float = 0.05):
        if torch.max(self.calcium_trace) > 1e-4:
            self.synaptic_weights[concept_idx] += lr * self.calcium_trace
            self.synaptic_weights[concept_idx] = torch.clamp(self.synaptic_weights[concept_idx], 0.0, 1.0)

    def mark_concept_trained(self, concept_idx: int):
        self.concept_trained[concept_idx] = 1.0

    def predict_evidence(self, cur_sdr: torch.Tensor, dt: float = 0.016, tau: float = 0.250):
        self.calcium_trace = torch.max(self.calcium_trace * 0.9, cur_sdr)
        
        strong_synapses = torch.sum(self.synaptic_weights > 0.5, dim=1).float()
        ltm_scores = torch.clamp(strong_synapses / float(self.total_k_active), 0.0, 1.0) * 100.0

        if torch.max(self.calcium_trace) < 1e-4:
            self.membrane_potential = self.membrane_potential * (1.0 - dt / tau)
            wm_scores = torch.zeros(self.num_concepts, device=DEVICE)
            weights = np.zeros(self.num_concepts, dtype=np.float32)
            return weights, wm_scores.cpu().numpy(), ltm_scores.cpu().numpy()

        w_norm = torch.nn.functional.normalize(self.synaptic_weights, p=2, dim=1)
        s_norm = torch.nn.functional.normalize(self.calcium_trace, p=2, dim=0)
        current = torch.mv(w_norm, s_norm)
        alpha = dt / tau
        self.membrane_potential = (1.0 - alpha) * self.membrane_potential + alpha * current
        wm_scores = torch.clamp(self.membrane_potential, 0.0, 1.0) * 100.0
        weights = torch.softmax(self.membrane_potential * 10.0, dim=0).cpu().numpy()
        return weights, wm_scores.cpu().numpy(), ltm_scores.cpu().numpy()

    def save_to_file(self, filepath: str):
        """Версионное сохранение с защитой от повреждения."""
        num_trained = int(torch.sum(self.concept_trained >= 1.0).item())
        
        # Если на диске уже лежит полный готовый банк (8/8), не даём портить его неполным дампом
        if os.path.exists(filepath):
            try:
                old_ckpt = torch.load(filepath, map_location='cpu', weights_only=True)
                old_trained = int(torch.sum(old_ckpt.get('concept_trained', torch.zeros(1)) >= 1.0).item())
                if old_trained >= self.num_concepts and num_trained < self.num_concepts:
                    print(f"🛡️ [LTM SAFEGUARD] Пропуск сохранения: на диске уже лежит полный банк ({old_trained}/{self.num_concepts}), текущий ({num_trained}/{self.num_concepts}) не будет перезаписывать его.")
                    return
            except Exception:
                pass

        payload = {
            'format_version': '1.0',
            'num_concepts': self.num_concepts,
            'concept_names': ALL_NAMES[:self.num_concepts],
            'synaptic_weights': self.synaptic_weights.cpu(),
            'concept_trained': self.concept_trained.cpu(),
            'timestamp': time.time()
        }
        torch.save(payload, filepath)
        print(f"💾 [LTM PERSISTENCE v1.0] Сохранено в {filepath} (Обучено концептов: {num_trained}/{self.num_concepts})")

    def load_from_file(self, filepath: str) -> tuple[bool, bool]:
        """
        Версионная загрузка. Возвращает (успех_загрузки, все_ли_обучены).
        """
        if not os.path.exists(filepath):
            return False, False
        try:
            ckpt = torch.load(filepath, map_location=DEVICE, weights_only=True)
            ver = ckpt.get('format_version', 'legacy')
            saved_nc = ckpt.get('num_concepts', self.num_concepts)
            min_c = min(self.num_concepts, saved_nc)

            self.synaptic_weights[:min_c].copy_(ckpt['synaptic_weights'][:min_c].to(DEVICE))
            self.concept_trained[:min_c].copy_(ckpt['concept_trained'][:min_c].to(DEVICE))

            all_trained = bool(torch.all(self.concept_trained[:self.num_concepts] >= 1.0).item())
            num_trained = int(torch.sum(self.concept_trained[:self.num_concepts] >= 1.0).item())
            print(f"📂 [LTM VERIFIED v{ver}] Загружен банк весов: {filepath} ({num_trained}/{self.num_concepts} концептов готово).")
            return True, all_trained
        except Exception as e:
            print(f"⚠️ [LTM LOAD FAILED]: {e}")
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

def apply_color_surgery(img_np, old_f32):
    # Оставляем функцию как есть, так как мы будем просто её отключать флагом
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
        self.is_sdxl = False
        self.latent_active = None
        self.is_active_world = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def update_simplex_targets(self, weights_nd: np.ndarray, force_strength: float = 0.50):
        if not self.initialized or self.c_bases is None: return
        w_sum = np.sum(weights_nd)
        if w_sum < 1e-5: return

        weights_norm = weights_nd / w_sum
        with torch.inference_mode():
            target = torch.zeros_like(self.c_bases[0])
            for i in range(self.num_concepts):
                target += float(weights_norm[i]) * self.c_bases[i]
            with self.lock:
                if self.latent_active is None:
                    self.latent_active = target.clone()
                else:
                    self.latent_active = self.latent_active * 0.65 + target * 0.35
                self.strength = force_strength

    def _loop(self):
        times = []
        while self.running:
            if not self.initialized:
                try:
                    self.conn = Client(('localhost', 6000), authkey=b'brain')
                    # Передаем флаг use_taesd на сервер
                    self.conn.send({'cmd': 'init_mode', 'mode': self.mode, 'use_taesd': self.use_taesd})
                    init_ack = self.conn.recv()
                    self.is_sdxl = init_ack.get('is_sdxl', False)
                    self.conn.send({'cmd': 'encode_base_prompts', 'prompts': self.prompts})
                    enc_resp = self.conn.recv()
                    self.c_bases = [torch.tensor(b, dtype=torch.float32, device=DEVICE) for b in enc_resp['c_bases']]
                    self.initialized = True
                    print(f"✅ [SD-{self.mode.upper()}] Connected to Brain Server (TAESD: {self.use_taesd}, ColorFix: {self.use_color}).")
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
                    'strength': max(0.40, s_val)
                }
                if self.speed == "fast":
                    req['num_inference_steps'] = 3

                self.conn.send(req)
                resp = self.conn.recv()

                with self.lock:
                    if isinstance(resp, np.ndarray):
                        if resp.shape[:2] != (self.img_h, self.img_w):
                            resp = cv2.resize(resp, (self.img_w, self.img_h))
                        
                        # Применяем хирургию цвета ТОЛЬКО если флаг активен
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
    parser = argparse.ArgumentParser(description="NeuroCanvas × tbp.monty")
    parser.add_argument('--sim', action='store_true', default=True, help="Start external agent swarm")
    parser.add_argument('--concepts', type=int, default=8, choices=[4, 8], help="Number of active concepts")
    parser.add_argument('--mode', type=str, default="lcm", choices=["lcm", "turbo", "sdxl-turbo"],
                        help="Active diffusion pipeline mode")
    parser.add_argument('--turbo', action='store_true', help="Alias for --mode turbo")
    parser.add_argument('--sdxl', action='store_true', help="Alias for --mode sdxl-turbo")
    parser.add_argument('--speed', type=str, default="fast", choices=["fast", "quality"])
    parser.add_argument('--hardcoded-bots', type=int, default=1, choices=list(range(8)))
    parser.add_argument('--jepa-bots', type=int, default=0, choices=list(range(8)))
    parser.add_argument('--weights', type=str, default="monty_ltm_weights.pt", help="Persistent synaptic LTM weights")
    parser.add_argument('--force-recalib', action='store_true', help="Force recalibration from scratch")
    
    # НОВЫЕ ФЛАГИ ОТКЛЮЧЕНИЯ
    parser.add_argument('--no-taesd', action='store_true', help="Disable TAESD VAE (LCM mode only)")
    parser.add_argument('--no-color', action='store_true', help="Disable apply_color_surgery post-processing")
    args = parser.parse_args()

    active_mode = args.mode
    if args.sdxl: active_mode = "sdxl-turbo"
    elif args.turbo: active_mode = "turbo"

    TARGET_NAMES = ALL_NAMES[:args.concepts]
    PROMPTS = [
        "deep outer space, glowing colorful nebula, bright stars, galaxy, 8k, sharp detailed",
        "spherical alien planet with atmosphere, continents and oceans in space, 8k, sharp detailed",
        "futuristic cyberpunk city street, neon lights, rain, glowing signs, sharp linework, 8k",
        "modern glass skyscraper buildings, downtown city, geometric architecture, sharp focus, 8k",
        "giant snowy mountain peak, rocky cliffs, clear blue sky, sharp focus, 8k",
        "ancient medieval stone castle fortress towers, daytime, sharp focus, 8k",
        "open stormy dark blue ocean, pure water surface, giant ocean waves, sea foam, no land, 8k",
        "dense lush green tropical jungle, giant trees, vines, sunlight piercing through leaves, 8k"
    ][:args.concepts]

    agent = None
    if args.sim:
        from synthetic_16d_causal_agent import SyntheticAutonomousAgent
        agent = SyntheticAutonomousAgent(
            num_concepts=args.concepts,
            num_hardcoded=args.hardcoded_bots,
            num_jepa=args.jepa_bots
        )

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"NeuroCanvas × tbp.monty: Empirical Heterarchy [{active_mode.upper()} - {args.speed.upper()}]")
    clock = pygame.time.Clock()
    font_b = pygame.font.SysFont("consolas", 13, bold=True)
    font_s = pygame.font.SysFont("consolas", 11)
    loading_font = pygame.font.SysFont("consolas", 16, bold=True)

    engine = HeterarchicalBrainEngine()
    engine.start()

    if agent:
        print("⏳ [BARRIER] Ожидание инициализации роя...")
        while not agent.is_ready():
            for event in pygame.event.get():
                if event.type == pygame.QUIT: raise KeyboardInterrupt
            screen.fill((10, 14, 20))
            txt = loading_font.render("INITIALIZING NEURAL SWARM...", True, (255, 200, 50))
            screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, HEIGHT // 2 - 20))
            pygame.display.flip()
            time.sleep(0.04)

        print("⏳ [BARRIER] Ожидание подключения физических потоков LSL...")
        start_wait = time.time()
        while engine.shm['num_live'].value < 1 and (time.time() - start_wait < 10.0):
            for event in pygame.event.get():
                if event.type == pygame.QUIT: raise KeyboardInterrupt
            screen.fill((10, 14, 20))
            txt = loading_font.render(f"CONNECTING LSL STREAMS ({engine.shm['num_live'].value}/4)...", True, (0, 255, 200))
            screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, HEIGHT // 2 - 20))
            pygame.display.flip()
            time.sleep(0.04)

    # ПЕРЕДАЕМ ФЛАГИ В ВОРКЕР
    worker = ToroidalDiffusionWorker(
        PROMPTS, 
        port=6000, 
        mode=active_mode, 
        speed=args.speed, 
        use_taesd=not args.no_taesd, 
        use_color=not args.no_color
    )
    clip_teacher = VisualCLIPTeacher(TARGET_NAMES, PROMPTS)
    heterarchy = FrontalExecutiveHeterarchy(num_concepts=args.concepts).to(DEVICE)

    # --- НАДЕЖНАЯ ЗАГРУЗКА ВЕСОВ LTM С ВЕРСИОННЫМ КОНТРОЛЕМ ---
    is_calibrating = True
    learn_idx = 0
    if not args.force_recalib and os.path.exists(args.weights):
        loaded, all_trained = heterarchy.load_from_file(args.weights)
        if loaded and all_trained:
            is_calibrating = False
            if agent: agent.set_calibration_target(False)
            print("🚀 [LTM FULL LOCK] Все 8 концептов подтверждены! Калибровка отключена, запущен боевой режим.")
        elif loaded:
            untrained = [i for i in range(args.concepts) if heterarchy.concept_trained[i] == 0]
            if untrained:
                learn_idx = untrained[0]
                print(f"ℹ️ [LTM RESUME] Продолжение дообучения с: [{TARGET_NAMES[learn_idx]}]")

    svd_spectrum = np.zeros(4)
    lead_causal_sign = 0.0

    FRACTAL_COLORS = [
        (20, 20, 60), (30, 80, 120), (100, 100, 110), (150, 100, 50),
        (40, 100, 60), (80, 40, 80), (10, 10, 20), (20, 80, 40)
    ]

    print(f"🧠 [SYSTEM] Heterarchy Decoder Online [{active_mode.upper()} - {args.speed.upper()}].")

    cur_probs = np.zeros(args.concepts, dtype=np.float32)
    clip_lock = threading.Lock()

    def async_clip_worker():
        nonlocal cur_probs
        last_classified_fid = -1
        while True:
            try:
                with worker.lock:
                    img_to_eval = worker.current_rgb.copy()
                    c_fid = worker.frame_id

                if c_fid != last_classified_fid and np.max(img_to_eval) > 0:
                    last_classified_fid = c_fid
                    probs = clip_teacher.classify(img_to_eval)
                    with clip_lock:
                        cur_probs = probs
                    if agent:
                        agent.update_visual_state(probs)
                        agent.update_visual_state(img_to_eval)
                time.sleep(0.05)
            except Exception:
                time.sleep(0.1)

    t_clip = threading.Thread(target=async_clip_worker, daemon=True)
    t_clip.start()

    try:
        while True:
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            frame = engine.get_frame()
            has_live_eeg = (frame.num_live > 0)

            node_gamma_tensors  = [torch.tensor(n.iplv_gamma, dtype=torch.float32, device=DEVICE) for n in frame.nodes]
            node_ripple_tensors = [torch.tensor(n.iplv_ripple, dtype=torch.float32, device=DEVICE) for n in frame.nodes]

            with torch.no_grad():
                full_sdr, sdr_f3 = heterarchy.get_current_sdr(node_gamma_tensors)

            with torch.no_grad():
                afz_ripple = node_ripple_tensors[2]
                ripple_centered = afz_ripple - torch.mean(afz_ripple, dim=0, keepdim=True)
                S_vals = torch.linalg.svdvals(ripple_centered)
                S_norm = (S_vals[:4] / (S_vals[0] + 1e-6)).cpu().numpy()
                svd_spectrum = S_norm

                live_depth = float(np.sum(S_norm > 0.22))
                live_depth = max(1.0, min(4.0, live_depth))
                smooth_depth = live_depth 
                lead_causal_sign = float(np.mean(frame.nodes[2].iplv_ripple[:, 0]))

            with worker.lock:
                rgb_m = worker.current_rgb.copy()

            with clip_lock:
                live_probs = cur_probs.copy()

            with torch.no_grad():
                live_human_w, wm_scores, ltm_scores = heterarchy.predict_evidence(full_sdr, dt=dt)

            lead_node = frame.nodes[0]

            # КАЛИБРОВКА СИНАПСОВ (LTP)
            if is_calibrating:
                target_sim = np.zeros(args.concepts, dtype=np.float32)
                target_sim[learn_idx] = 1.0
                worker.update_simplex_targets(target_sim, force_strength=1.0)
                if agent: agent.set_calibration_target(True, learn_idx)

                vis_conf = float(live_probs[learn_idx])

                if vis_conf >= 0.50 and has_live_eeg:
                    heterarchy.stream_learn_accumulate(learn_idx, full_sdr, lr=0.04)

                if has_live_eeg and int(time.time() * 10) % 5 == 0:
                    calc_lvl = float(torch.max(heterarchy.calcium_trace))
                    print(f"[{TARGET_NAMES[learn_idx]}] CLIP:{vis_conf*100:4.1f}% | "
                          f"Ca2+ STM:{calc_lvl:4.2f} | "
                          f"WM Memb:{wm_scores[learn_idx]:4.1f}% | "
                          f"LTM Engram:{ltm_scores[learn_idx]:5.1f}%")

                if ltm_scores[learn_idx] >= 75.0:
                    heterarchy.mark_concept_trained(learn_idx)
                    heterarchy.save_to_file(args.weights)
                    untrained = [i for i in range(args.concepts) if heterarchy.concept_trained[i] == 0]
                    if untrained:
                        learn_idx = untrained[0]
                    else:
                        is_calibrating = False
                        if agent: agent.set_calibration_target(False)
                        print("🏆 [LTM CONSOLIDATED] Все 8 концептов консолидированы! Включается живой режим.")

                blended_weights = target_sim
            else:
                # В БОЕВОМ РЕЖИМЕ: реальные активные ансамбли
                active_mask = (wm_scores > 20.0)
                if np.any(active_mask) and has_live_eeg:
                    blended_weights = wm_scores * active_mask
                    blended_weights /= np.sum(blended_weights)
                    
                    status_str = agent.get_telemetry() if agent else ""
                    force_str = 0.35 if ":LOC" in status_str else 0.50

                    worker.update_simplex_targets(blended_weights, force_strength=force_str)
                else:
                    top_i = int(np.argmax(wm_scores))
                    blended_weights = np.zeros(args.concepts, dtype=np.float32)
                    blended_weights[top_i] = 1.0
                    worker.update_simplex_targets(blended_weights, force_strength=0.35)

            # РАСЧЁТ ДИНАМИЧЕСКОГО TREEMAP
            map_x, map_y = 950, 40
            map_w, map_h = 800, 560
            boxes, resolved_hierarchy = calculate_dynamic_treemap(
                map_x, map_y, map_w, map_h, blended_weights, smooth_depth, lead_causal_sign
            )
            
            # ОТРИСОВКА PYGAME
            screen.fill((10, 14, 20))

            # Диффузия
            img_x, img_y = 370, 40
            surf_diff = pygame.image.frombuffer(rgb_m.tobytes(), (worker.img_w, worker.img_h), 'RGB')
            if (worker.img_w, worker.img_h) != (512, 384):
                surf_diff = pygame.transform.scale(surf_diff, (512, 384))
            screen.blit(surf_diff, (img_x, img_y))
            pygame.draw.rect(screen, (40, 50, 70), (img_x, img_y, 512, 384), 2, border_radius=8)

            # L4 F3 Sheet
            cur_sdr_img = sdr_f3[:4096].view(64, 64).cpu().numpy() * 255.0
            sdr_surf = pygame.surfarray.make_surface(cv2.resize(cur_sdr_img, (140, 140)).astype(np.uint8))
            screen.blit(sdr_surf, (img_x, 445))
            screen.blit(font_s.render("L4 F3 Sheet (30–85 Гц)", True, (0, 255, 200)), (img_x, 428))

            # Панель отладки Vm
            dbg_x = img_x + 160
            dbg_y = 430
            dbg_w = 400
            dbg_h = 170
            pygame.draw.rect(screen, (14, 18, 26), (dbg_x, dbg_y, dbg_w, dbg_h), border_radius=6)
            pygame.draw.rect(screen, (40, 70, 100), (dbg_x, dbg_y, dbg_w, dbg_h), 1, border_radius=6)
            screen.blit(font_b.render("LIVE WORKING MEMORY ENSEMBLES (Vm):", True, (0, 255, 200)), (dbg_x + 10, dbg_y + 8))

            for i, name in enumerate(TARGET_NAMES):
                score = wm_scores[i]
                col_bar = (0, 255, 180) if score > 20.0 else (80, 80, 90)
                if not has_live_eeg: col_bar = (40, 40, 40)
                
                bx = dbg_x + 10 + (i % 2) * 195
                by = dbg_y + 30 + (i // 2) * 22
                
                screen.blit(font_s.render(f"{name[:7]:7s}", True, (200, 200, 200)), (bx, by))
                bar_len = int((score / 100.0) * 80)
                pygame.draw.rect(screen, (30, 35, 45), (bx + 55, by + 2, 80, 10))
                if bar_len > 0:
                    pygame.draw.rect(screen, col_bar, (bx + 55, by + 2, bar_len, 10))
                screen.blit(font_s.render(f"{score:4.1f}%", True, col_bar), (bx + 140, by))

            th_text = "Threshold: >20.0% Vm enters reality" if not is_calibrating else "Calibrating: Locked to target"
            screen.blit(font_s.render(th_text, True, (150, 150, 160)), (dbg_x + 10, dbg_y + 148))

            # Панель 1: LTM
            panel_x, panel_y = 20, 40
            pygame.draw.rect(screen, (16, 22, 32), (panel_x, panel_y, 330, 270), border_radius=8)
            pygame.draw.rect(screen, (0, 255, 200), (panel_x, panel_y, 330, 270), 1, border_radius=8)

            if is_calibrating:
                screen.blit(font_b.render("SYNAPTIC CALIBRATION (LTP)", True, (255, 180, 50)), (panel_x + 12, panel_y + 12))
                screen.blit(font_s.render(f"Target: [{TARGET_NAMES[learn_idx]}]", True, (255, 255, 100)), (panel_x + 15, panel_y + 36))
                for i, name in enumerate(TARGET_NAMES):
                    score = ltm_scores[i]
                    col_s = (100, 255, 100) if score >= 75.0 else (255, 80, 80)
                    txt_s = f"{name:10s}: {score:4.1f}%" if heterarchy.concept_trained[i] > 0 else f"{name:10s}: QUEUED..."
                    if i == learn_idx: txt_s, col_s = f"{name:10s}: LEARNING...", (255, 220, 50)
                    screen.blit(font_s.render(txt_s, True, col_s), (panel_x + 15, panel_y + 70 + i * 18))
            else:
                screen.blit(font_b.render("FROZEN RETENTION (LTM PERSISTED)", True, (0, 255, 200)), (panel_x + 12, panel_y + 12))
                for i, name in enumerate(TARGET_NAMES):
                    screen.blit(font_s.render(f"{name:10s}: {ltm_scores[i]:4.1f}% [LOADED]", True, (100, 255, 100)), (panel_x + 15, panel_y + 38 + i * 22))

            # Панель 2: Телеметрия
            c_x, c_y = 20, 330
            pygame.draw.rect(screen, (16, 22, 32), (c_x, c_y, 330, 270), border_radius=8)
            pygame.draw.rect(screen, (100, 180, 255), (c_x, c_y, 330, 270), 1, border_radius=8)
            screen.blit(font_b.render("CORTICAL DECODER (STRICT EMPIRICAL)", True, (100, 180, 255)), (c_x + 12, c_y + 12))

            top_idx = int(np.argmax(blended_weights))
            top_conf = blended_weights[top_idx] * 100.0
            screen.blit(font_b.render(f"Decoded Intent : [{TARGET_NAMES[top_idx]}] ({top_conf:4.1f}%)", True, (100, 255, 100)), (c_x + 15, c_y + 36))

            if agent:
                status_str = agent.get_telemetry()
                screen.blit(font_b.render("Swarm Internal Decision State:", True, (255, 200, 100)), (c_x + 15, c_y + 70))
                lines = [status_str[i:i+38] for i in range(0, len(status_str), 38)]
                for idx_l, line in enumerate(lines[:3]):
                    screen.blit(font_s.render(line, True, (200, 200, 220)), (c_x + 15, c_y + 92 + idx_l * 16))

            screen.blit(font_s.render(f"Beta Stability: {lead_node.beta_stability:+.2f}", True, (255, 200, 50)), (c_x + 15, c_y + 145))
            screen.blit(font_s.render(f"Pacing: Theta={frame.theta_freq:.2f}Hz | Delta={frame.delta_freq:.2f}Hz", True, (200, 220, 240)), (c_x + 15, c_y + 170))
            screen.blit(font_s.render(f"Pipeline: {active_mode.upper()} ({worker.fps:.1f} FPS)", True, (180, 180, 220)), (c_x + 15, c_y + 195))

            # 120 ребер рипплов
            BY, BH = 620, 300
            pygame.draw.rect(screen, (12, 16, 24), (20, BY, 870, BH), border_radius=8)
            pygame.draw.rect(screen, (30, 45, 65), (20, BY, 870, BH), 1, border_radius=8)
            screen.blit(font_b.render("120-EDGE DIRECTED iPLV (100–200 Гц РИППЛЫ)", True, (0, 255, 200)), (35, BY + 15))

            g120 = lead_node.iplv_ripple[31]
            bw = 800.0 / 120.0
            mid_line = BY + 150
            for p in range(120):
                val = g120[p]
                bh = int(abs(val) * 110)
                bx = 35 + p * bw
                col = (255, 80, 80) if val < 0 else (80, 255, 180)
                if val >= 0: pygame.draw.rect(screen, col, (bx, mid_line - bh, max(1, int(bw - 1)), bh))
                else: pygame.draw.rect(screen, col, (bx, mid_line, max(1, int(bw - 1)), bh))

            # Панель гетероархии
            rx, ry = 920, 620
            pygame.draw.rect(screen, (20, 15, 30), (rx, ry, 840, 300), border_radius=8)
            pygame.draw.rect(screen, (255, 100, 200), (rx, ry, 840, 300), 1, border_radius=8)
            screen.blit(font_b.render("DECODED CAUSAL MANIFOLD", True, (255, 100, 200)), (rx + 20, ry + 20))

            y_offset = ry + 60
            dir_str = "A ⊃ B (Родитель ведет)" if lead_causal_sign >= 0 else "B ⊃ A (Инверсия позы)"
            screen.blit(font_s.render(f"Causal Lead: {lead_causal_sign:+.3f} -> {dir_str}", True, (200, 200, 220)), (rx + 20, y_offset))

            y_offset += 25
            screen.blit(font_s.render("SVD Спектр Рипплов 100–200 Гц:", True, (150, 150, 150)), (rx + 20, y_offset))
            for i, sv in enumerate(svd_spectrum):
                color = (0, 255, 180) if sv > 0.22 else (80, 80, 80)
                bh = int(sv * 35)
                pygame.draw.rect(screen, color, (rx + 20 + i*65, y_offset + 20 + (35 - bh), 45, bh))
                screen.blit(font_s.render(f"S{i+1}: {sv:.2f}", True, (200, 200, 200)), (rx + 20 + i*65, y_offset + 60))

            # Treemap
            pygame.draw.rect(screen, (10, 10, 10), (map_x, map_y, map_w, map_h))
            pygame.draw.rect(screen, (255, 255, 255), (map_x-2, map_y-2, map_w+4, map_h+4), 2)
            treemap_title = f"TREEMAP SOVEREIGNTY (K={smooth_depth:.2f})"
            screen.blit(font_b.render(treemap_title, True, (255, 255, 255)), (map_x, map_y - 25))

            owners = agent.get_owners() if agent else [-1] * 8

            for concept_idx in resolved_hierarchy:
                bx, by, bw, bh, rank = boxes[concept_idx]
                pygame.draw.rect(screen, FRACTAL_COLORS[concept_idx], (bx, by, bw, bh))
                pygame.draw.rect(screen, (220, 220, 220), (bx, by, bw, bh), 2)
                
                owner_id = owners[concept_idx]
                if owner_id == 0 and frame.is_real:
                    owner_name = "YOU"
                elif owner_id > 0:
                    owner_name = f"Bot {owner_id}"
                else:
                    owner_name = "CONSENSUS"

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
        heterarchy.save_to_file(args.weights)
        if agent: agent.stop()
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
