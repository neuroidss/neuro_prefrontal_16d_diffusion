#!/usr/bin/env python3
"""
🧠 NEUROCANVAS × TBP.MONTY: CONTINUOUS 4-AXIS ANALOG SPATIOTEMPORAL HETERARCHY
- 100% GPU-АКСЕЛЕРАЦИЯ: Вектора обновляются со скоростью Теты (60 FPS) без задержек.
- СТРОГОЕ БИОФИЗИЧЕСКОЕ РАЗДЕЛЕНИЕ РЕГИОНОВ:
    * FCz (SMA): Моторная навигация и 4-осевой полет. Только он двигает камеру!
    * AFz (rmPFC): Когнитивный Тор (u, v) и семантическая калибровка понятий.
    * F3 (Left PFC): Бета-контроль порядка и устойчивости (Beta Order).
    * F4 (Right PFC): Бета-драйв хаоса и бифуркаций Фейгенбаума (Beta Chaos).
    * Fpz (BA 10): Когнитивное ветвление (Cognitive Branching).
- ПОЛНАЯ ТЕТА-ГАММА ИНТЕГРАЦИЯ: 32 слота PAC, 4096 L4 HTM колонок, ciPLV, SVD 89.5 Гц.
- Управление: Стрелки (X/Y), Точка/Запятая (Поворот), SPACE (Хаос), M (Sensory Sub), R (Калибровка).
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

from neuro_heterarchy_core import HeterarchicalBrainEngine, DEVICE, NodeState, PROJ_MATRICES, COORDS_X, COORDS_Y
from tbp.monty.cmp import Message
from synthetic_16d_causal_agent import SyntheticAutonomousAgent

WIDTH, HEIGHT = 1800, 960
MAX_CONCEPTS_CAPACITY = 16
FEIGENBAUM_DELTA = 4.669201609

ALL_NAMES = ["ГОРА",  "ДЖУНГЛИ", "ЗАМОК", "ОКЕАН", "КОСМОС", "ПЛАНЕТА", "КИБЕРПАНК", "НЕБОСКРЕБ"]
MOTION_NAMES = ["ВПЕРЕД", "НАЗАД", "ВПРАВО", "ВЛЕВО"]

TARGET_ANGLES_DEG = [90.0, 270.0, 0.0, 180.0]
TARGET_UNIT_VECS = [
    np.array([ 0.0,  1.0], dtype=np.float32),  # Вперед (+Y)
    np.array([ 0.0, -1.0], dtype=np.float32),  # Назад (-Y)
    np.array([ 1.0,  0.0], dtype=np.float32),  # Вправо (+X)
    np.array([-1.0,  0.0], dtype=np.float32)   # Влево (-X)
]

AXIS_NAMES = ["ОСЬ Y (ПРОДОЛЬНАЯ)", "ОСЬ X (БОКОВАЯ)"]

FRACTAL_COLORS = [
    (20, 20, 60), (30, 80, 120), (100, 100, 110), (150, 100, 50),
    (40, 100, 60), (80, 40, 80), (10, 10, 20), (20, 80, 40),
    (120, 40, 40), (40, 120, 120), (80, 80, 30), (30, 30, 90),
    (90, 40, 90), (40, 90, 40), (110, 70, 30), (50, 50, 50)
]

# =====================================================================
# 1. ДИНАМИЧЕСКИЙ КИНЕМАТИЧЕСКИЙ ПИЛОТ (ВЕБ-ФИЗИКА)
# =====================================================================
class DynamicPilot:
    def __init__(self, sensitivity=0.05, max_speed=9.0, fwd_scale=0.2, strafe_scale=0.2, turn_scale=0.5, intent_gain=1.5):
        self.x, self.y = 0.0, 0.0
        self.vx, self.vy = 0.0, 0.0
        self.persistence = 0.0
        self.last_ix, self.last_iy = 0.0, 0.0
        self.wm_curvature = 0.0
        self.temporal_bias = 0.0
        self.max_speed = max_speed

    def update(self, dt: float, force_x: float, force_y: float, wm_curvature: float, temp_bias: float):
        self.wm_curvature = wm_curvature
        self.temporal_bias = temp_bias

        mag = math.hypot(force_x, force_y)
        if mag > 0.01:
            last_mag = math.hypot(self.last_ix, self.last_iy) + 1e-6
            dot = (force_x * self.last_ix + force_y * self.last_iy) / (mag * last_mag)
            alignment = 0.5 * (dot + 1.0)
        else:
            alignment = 0.5

        self.persistence = self.persistence * 0.94 + 0.06 * alignment * math.tanh(mag * 2.0)
        self.last_ix, self.last_iy = force_x, force_y

        active_boost = 1.0 + self.persistence * 4.0
        base_speed = 3.2 * active_boost

        target_vx = force_x * base_speed
        target_vy = force_y * base_speed

        t_mag = math.hypot(target_vx, target_vy)
        if t_mag > self.max_speed:
            target_vx = (target_vx / t_mag) * self.max_speed
            target_vy = (target_vy / t_mag) * self.max_speed

        self.vx = self.vx * 0.88 + target_vx * 0.12
        self.vy = self.vy * 0.88 + target_vy * 0.12

        self.x += self.vx * dt
        self.y += self.vy * dt
        return active_boost

def apply_manifold_camera_warp(img_np: np.ndarray, pilot: DynamicPilot, dt: float = 0.016):
    h, w = img_np.shape[:2]
    zoom = 1.0 + (pilot.vy * dt * 0.18)
    dx = -pilot.vx * w * dt * 0.12
    angle = -pilot.wm_curvature * 10.0 * dt

    if abs(zoom - 1.0) < 0.0008 and abs(dx) < 0.15 and abs(angle) < 0.04:
        return img_np

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, zoom)
    M[0, 2] += dx
    warped = cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return warped

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

# =====================================================================
# 2. РЕКУРСИВНЫЙ TREEMAP
# =====================================================================
def calculate_emergent_treemap_recursive(x, y, w, h, weights, parent_map, current_node, depth=0):
    boxes = {}
    boxes[current_node] = (int(x), int(y), int(max(15, w)), int(max(15, h)), 0, depth)
    children = [c for c, p in parent_map.items() if p == current_node and c < len(weights) and weights[c] > 0.02]
    if not children: return boxes

    children = sorted(children, key=lambda idx: weights[idx], reverse=True)
    margin_x, margin_y = w * 0.04, h * 0.08
    inner_x, inner_y = x + margin_x, y + margin_y * 1.5
    inner_w, inner_h = w - 2 * margin_x, h - margin_y * 2.0
    if inner_w < 15 or inner_h < 15: return boxes

    child_weights = np.array([weights[c] for c in children], dtype=np.float32)
    total_cw = np.sum(child_weights)
    proportions = np.ones(len(children)) / len(children) if total_cw <= 1e-6 else child_weights / total_cw

    cur_cx = inner_x
    for i, ch_idx in enumerate(children):
        cw = max(15, inner_w * proportions[i])
        boxes.update(calculate_emergent_treemap_recursive(cur_cx, inner_y, cw, inner_h, weights, parent_map, ch_idx, depth + 1))
        cur_cx += cw
    return boxes

def calculate_emergent_treemap(x, y, w, h, active_weights, count, parent_map=None):
    valid_weights = active_weights[:count]
    if parent_map is None: parent_map = {i: None for i in range(count)}
    active_indices = [i for i, val in enumerate(valid_weights) if val > 0.02]
    if not active_indices: return {0: (x, y, w, h, 0, 0)}, [0]

    roots = [node for node in active_indices if parent_map.get(node) is None or parent_map.get(node) not in active_indices]
    if not roots: roots = [max(active_indices, key=lambda idx: valid_weights[idx])]

    root_weights = np.array([valid_weights[r] for r in roots], dtype=np.float32)
    total_rw = np.sum(root_weights)
    proportions = np.ones(len(roots)) / len(roots) if total_rw <= 1e-6 else root_weights / total_rw

    boxes = {}
    cur_rx = x
    for i, root in enumerate(roots):
        rw = max(20, w * proportions[i])
        boxes.update(calculate_emergent_treemap_recursive(cur_rx, y, rw, h, valid_weights, parent_map, root, depth=0))
        cur_rx += rw
        
    resolved = list(boxes.keys())
    orphans = [i for i in active_indices if i not in resolved]
    if orphans:
        ow = w / len(orphans)
        for i, o_idx in enumerate(orphans):
            boxes[o_idx] = (int(x + i*ow), int(y + h - 25), int(ow), 25, 0, 0)
            resolved.append(o_idx)
    return boxes, resolved

# =====================================================================
# 3. HTM 32-СЛОТОВАЯ ПРОСТРАНСТВЕННО-ВРЕМЕННАЯ КОЛОНКА (100% GPU)
# =====================================================================
class CanonicalHTMColumn(nn.Module):
    def __init__(self, node_id: str = "Node", num_columns: int = 4096, k_active: int = 80, num_slots: int = 32):
        super().__init__()
        self.num_columns, self.k_active, self.num_slots = num_columns, k_active, num_slots

        ex, ey = [], []
        for i in range(16):
            for j in range(i + 1, 16):
                ex.append((COORDS_X[i] + COORDS_X[j]) / 2.0)
                ey.append((COORDS_Y[i] + COORDS_Y[j]) / 2.0)
        self.register_buffer("edge_x", torch.tensor(ex, device=DEVICE, dtype=torch.float32))
        self.register_buffer("edge_y", torch.tensor(ey, device=DEVICE, dtype=torch.float32))

        grid_dim = int(math.isqrt(num_columns))
        cy = torch.linspace(-11.0, 11.0, grid_dim, device=DEVICE).view(grid_dim, 1, 1)
        cx = torch.linspace(-11.0, 11.0, grid_dim, device=DEVICE).view(1, grid_dim, 1)
        d_sq = (cx - self.edge_x.view(1, 1, 120))**2 + (cy - self.edge_y.view(1, 1, 120))**2
        spatial_rf = torch.exp(-d_sq / 40.0).view(-1, 120)[:num_columns]
        self.register_buffer("permanence", spatial_rf)
        self.perm_threshold = 0.25
        
        # КЭШИРОВАНИЕ НА GPU (Исключает постоянные аллокации памяти)
        connected = (spatial_rf >= self.perm_threshold).float()
        self.register_buffer("connected_T", connected.T.contiguous())
        self.register_buffer("synapse_counts", torch.sum(connected, dim=1).clamp(min=1.0).view(1, -1))

    def compute_sdr(self, pac_iplv_32x120: torch.Tensor) -> torch.Tensor:
        x_clean = torch.relu(pac_iplv_32x120)
        if torch.max(x_clean) < 1e-5:
            return torch.full((self.num_slots, self.num_columns), 1e-4, device=DEVICE)

        overlap = torch.matmul(x_clean, self.connected_T) / self.synapse_counts
        _, active_indices = torch.topk(overlap, self.k_active, dim=-1)
        sdr_seq = torch.full((self.num_slots, self.num_columns), 0.01, device=DEVICE)
        sdr_seq.scatter_(1, active_indices, 1.0)
        return sdr_seq

class FrontalExecutiveHeterarchy(nn.Module):
    def __init__(self, num_nodes: int = 1, max_capacity: int = MAX_CONCEPTS_CAPACITY, 
                 num_columns_per_node: int = 4096, k_active_per_node: int = 80, num_slots: int = 32):
        super().__init__()
        self.num_nodes, self.num_slots = max(1, int(num_nodes)), num_slots
        self.total_dim = num_columns_per_node * self.num_nodes
        self.max_capacity, self.total_k_active = max_capacity, k_active_per_node * self.num_nodes
        
        self.nodes = nn.ModuleList([
            CanonicalHTMColumn(node_id=f"Col_{i}", num_columns=num_columns_per_node, 
                               k_active=k_active_per_node, num_slots=num_slots)
            for i in range(self.num_nodes)
        ])
        self.register_buffer("trajectory_weights", torch.full((max_capacity, num_slots, self.total_dim), 0.05, device=DEVICE))
        self.register_buffer("calcium_trajectory", torch.zeros((num_slots, self.total_dim), device=DEVICE))
        self.register_buffer("membrane_potential", torch.zeros(max_capacity, device=DEVICE))

    def get_current_sdr(self, iplv_gamma_nodes: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        sdrs = []
        for i in range(self.num_nodes):
            if i < len(iplv_gamma_nodes):
                sdrs.append(self.nodes[i].compute_sdr(iplv_gamma_nodes[i]))
            else:
                sdrs.append(torch.zeros((self.num_slots, self.total_dim // self.num_nodes), device=DEVICE))
        full_sdr_seq = torch.cat(sdrs, dim=-1)
        return full_sdr_seq, sdrs[0]

    def contrastive_learn(self, target_idx: int, num_active_concepts: int, lr: float = 0.05, ltd_factor: float = 0.5):
        if torch.max(self.calcium_trajectory) > 1e-5 and abs(lr) > 1e-6:
            self.trajectory_weights[target_idx] += lr * self.calcium_trajectory
            self.trajectory_weights[target_idx] = torch.clamp(self.trajectory_weights[target_idx], 0.01, 1.0)
            for other in range(num_active_concepts):
                if other != target_idx:
                    self.trajectory_weights[other] -= (lr * ltd_factor) * self.calcium_trajectory
                    self.trajectory_weights[other] = torch.clamp(self.trajectory_weights[other], 0.01, 1.0)

    def inherit_synapses(self, parent_idx: int, child_idx: int):
        self.trajectory_weights[child_idx] = self.trajectory_weights[parent_idx] * 0.85

    def get_ltm_scores(self, count: int) -> np.ndarray:
        strong = torch.sum(self.trajectory_weights[:count] > 0.40, dim=(1, 2)).float()
        return (torch.clamp(strong / float(self.total_k_active * self.num_slots), 0.05, 1.0) * 100.0).cpu().numpy()

    def predict_evidence(self, cur_sdr_seq: torch.Tensor, active_count: int, dt: float = 0.016, tau: float = 0.250):
        self.calcium_trajectory = torch.max(self.calcium_trajectory * 0.88, cur_sdr_seq)
        ltm_scores = self.get_ltm_scores(active_count)
        if torch.max(self.calcium_trajectory) < 1e-5:
            self.membrane_potential[:active_count] *= (1.0 - dt / tau)
            return np.ones(active_count, dtype=np.float32) / max(1, active_count), np.full(active_count, 5.0), ltm_scores

        w_flat = torch.nn.functional.normalize(self.trajectory_weights[:active_count].view(active_count, -1), p=2, dim=1)
        s_flat = torch.nn.functional.normalize(self.calcium_trajectory.view(-1), p=2, dim=0)
        current = torch.mv(w_flat, s_flat)

        alpha = dt / tau
        self.membrane_potential[:active_count] = (1.0 - alpha) * self.membrane_potential[:active_count] + alpha * current
        wm_scores = torch.clamp(self.membrane_potential[:active_count], 0.0, 1.0) * 100.0
        weights = torch.softmax(self.membrane_potential[:active_count] * 8.0, dim=0).cpu().numpy()
        return weights, wm_scores.cpu().numpy(), ltm_scores

    def compute_contrastive_margin(self, cur_sdr_seq: torch.Tensor, target_idx: int, active_count: int) -> float:
        if active_count <= 1: return 1.0
        with torch.no_grad():
            w_norm = torch.nn.functional.normalize(self.trajectory_weights[:active_count].view(active_count, -1), p=2, dim=1)
            s_norm = torch.nn.functional.normalize(cur_sdr_seq.view(-1), p=2, dim=0)
            scores = torch.mv(w_norm, s_norm)
            target_score = scores[target_idx]
            other_scores = torch.cat([scores[:target_idx], scores[target_idx+1:]])
            return float((target_score - torch.max(other_scores)).item())

    def save_to_file(self, filepath: str, class_names: list):
        torch.save({
            'format_version': '12.2_analog_spatiotemporal',
            'num_nodes': self.num_nodes, 'num_slots': self.num_slots, 'total_dim': self.total_dim,
            'num_concepts': len(class_names), 'class_names': class_names,
            'trajectory_weights': self.trajectory_weights[:len(class_names)].cpu(), 'timestamp': time.time()
        }, filepath)
        print(f"💾 [LTM] Банк весов сохранен: {filepath}")

    def load_from_file(self, filepath: str) -> tuple[bool, int]:
        if not os.path.exists(filepath): return False, 0
        try:
            ckpt = torch.load(filepath, map_location=DEVICE, weights_only=True)
            cnt = min(self.max_capacity, ckpt.get('num_concepts', 0))
            if 'trajectory_weights' in ckpt:
                self.trajectory_weights[:cnt].copy_(ckpt['trajectory_weights'][:cnt].to(DEVICE))
                print(f"📂 [LTM] Загружен банк весов: {filepath} ({cnt} классов)")
                return True, cnt
            return False, 0
        except Exception as e:
            return False, 0

class BrainSubject:
    def __init__(self, subject_id: str, regions_map: dict, max_capacity: int = MAX_CONCEPTS_CAPACITY, **pilot_kwargs):
        self.subject_id = subject_id
        self.regions_map = regions_map
        num_nodes = max(1, len(regions_map))
        
        self.heterarchy = FrontalExecutiveHeterarchy(num_nodes=num_nodes, max_capacity=max_capacity).to(DEVICE)
        self.pilot = DynamicPilot(**pilot_kwargs)
        
        self.decoded_leader_idx = 0
        self.confidence = 0.0
        self.wm_scores = np.zeros(max_capacity, dtype=np.float32)
        self.ltm_scores = np.zeros(max_capacity, dtype=np.float32)
        self.blended_weights = np.zeros(max_capacity, dtype=np.float32)
        self.last_sdr = None
        self.last_sdr_first = None

    def has_region(self, region_name: str) -> bool:
        return region_name in self.regions_map

    def get_region_node(self, region_name: str, frame_nodes: list[NodeState]) -> NodeState | None:
        if region_name in self.regions_map:
            dev_idx = self.regions_map[region_name]
            if 0 <= dev_idx < len(frame_nodes):
                return frame_nodes[dev_idx]
        return None

    def process_neurophysiology(self, frame_nodes: list[NodeState], dt: float, active_count: int):
        my_tensors = []
        for reg_name, dev_idx in self.regions_map.items():
            if 0 <= dev_idx < len(frame_nodes):
                my_tensors.append(torch.tensor(frame_nodes[dev_idx].iplv_gamma, dtype=torch.float32, device=DEVICE))
        if not my_tensors: my_tensors = [torch.zeros((32, 120), device=DEVICE)]

        with torch.no_grad():
            full_sdr_seq, sdr_first = self.heterarchy.get_current_sdr(my_tensors)
            self.last_sdr = full_sdr_seq
            self.last_sdr_first = sdr_first
            weights, wm, ltm = self.heterarchy.predict_evidence(full_sdr_seq, active_count, dt=dt)
            self.blended_weights = weights
            self.wm_scores = wm
            self.ltm_scores = ltm
            self.decoded_leader_idx = int(np.argmax(wm)) if active_count > 0 else 0
            self.confidence = float(np.max(wm))

    def learn_contrastive(self, target_idx: int, num_active: int, lr: float = 0.05, ltd_factor: float = 0.5):
        if self.last_sdr is not None:
            self.heterarchy.contrastive_learn(target_idx, num_active, lr=lr, ltd_factor=ltd_factor)

    def get_cmp_message(self, node_lead: NodeState, smooth_depth: float) -> Message:
        cy = math.cos(self.pilot.wm_curvature * 1.5)
        sy = math.sin(self.pilot.wm_curvature * 1.5)
        rot_matrix = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        current_3d_loc = np.array([self.pilot.x, self.pilot.y, 0.0], dtype=np.float64)
        return Message(
            location=current_3d_loc,
            morphological_features={"pose_vectors": rot_matrix, "pose_fully_defined": bool(smooth_depth >= 1.8), "on_object": 1.0},
            non_morphological_features={
                "object_id": int(self.decoded_leader_idx),
                "torus_u": float(node_lead.torus_u), "torus_v": float(node_lead.torus_v),
                "temporal_bias": float(self.pilot.temporal_bias), "beta_order": float(node_lead.beta_power)
            },
            confidence=float(np.clip(self.confidence / 100.0, 0.0, 1.0)),
            pass_message=True, sender_id=self.subject_id, sender_type="SM", process_features_in_lm=True
        )

class OverBrainCollective:
    def __init__(self, subjects: list[BrainSubject]): self.subjects = subjects
    def resolve_heterarchy(self, active_count: int, worker_parent_map: dict, agent_parent_map: dict):
        parent_map = {i: None for i in range(active_count)}
        for child, parent in worker_parent_map.items():
            if child < active_count and parent is not None and parent < active_count: parent_map[child] = parent
        if agent_parent_map:
            for child, parent in agent_parent_map.items():
                if child < active_count and parent is not None and parent < active_count: parent_map[child] = parent
        return parent_map

# =====================================================================
# 4. ДИФФУЗИОННЫЙ ВОРКЕР И CLIP
# =====================================================================
class VisualCLIPTeacher:
    def __init__(self, text_prompts):
        model_id = "openai/clip-vit-large-patch14"
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        with torch.no_grad():
            inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True).to(DEVICE)
            feat = self.model.get_text_features(**inputs)
            self.text_features = feat / feat.norm(dim=-1, keepdim=True)

    def classify(self, rgb_image_np: np.ndarray, count: int) -> np.ndarray:
        if rgb_image_np is None or np.max(rgb_image_np) == 0 or count == 0: return np.zeros(count, dtype=np.float32)
        inputs = self.processor(images=Image.fromarray(rgb_image_np), return_tensors="pt").to(DEVICE)
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        with torch.no_grad():
            img_feat = self.model.get_image_features(**inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = (img_feat @ self.text_features[:count].T) * 30.0
            return torch.softmax(logits, dim=-1).cpu().numpy()[0]

class EqualPoolChaosWorker:
    def __init__(self, initial_prompts: list, initial_names: list, port: int = 6000, mode: str = "lcm", speed: str = "fast", use_taesd: bool = True, use_color: bool = True):
        self.conn = None
        self.speed, self.use_taesd, self.use_color = speed, use_taesd, use_color
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
        self.c_bases = []
        self.depths = [0] * MAX_CONCEPTS_CAPACITY
        self.energies = [0.0] * MAX_CONCEPTS_CAPACITY
        self.parent_map = {i: None for i in range(MAX_CONCEPTS_CAPACITY)}
        
        self.chaos_enabled = False
        self.latent_active = None
        self.is_sdxl = False
        self.dir_u, self.dir_v = None, None
        self.last_split_time = time.time()
        
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def toggle_chaos(self):
        with self.lock:
            self.chaos_enabled = not self.chaos_enabled
            status = "ВКЛЮЧЕН" if self.chaos_enabled else "ВЫКЛЮЧЕН"
            print(f"\n🌀 [CHAOS TRIGGER] Режим Хаотической Гетерархии: {status}")

    def bifurcate_concept(self, parent_idx: int, torus_u: float, torus_v: float, lead_sign: float):
        if self.active_count >= MAX_CONCEPTS_CAPACITY: return -1
        new_idx = self.active_count
        self.active_count += 1
        p_depth = self.depths[parent_idx]
        self.depths[new_idx] = p_depth + 1
        self.energies[parent_idx], self.energies[new_idx] = 0.0, 0.0
        self.parent_map[new_idx] = parent_idx
        
        drift_scale = 0.55 / (1.35 ** p_depth)
        drift = (self.dir_u * math.cos(torus_u) + self.dir_v * math.sin(torus_v)) * drift_scale
        parent_base = self.c_bases[parent_idx].clone()
        child_base = parent_base + drift
        child_base = child_base / torch.norm(child_base, dim=-1, keepdim=True) * torch.norm(parent_base, dim=-1, keepdim=True)
        self.c_bases.append(child_base)
        
        base_parent_name = self.names[parent_idx].split(".")[0]
        direction_tag = "in" if lead_sign >= 0 else "out"
        new_name = f"{base_parent_name}.{self.depths[new_idx]}{direction_tag}"
        self.names.append(new_name)
        print(f"🌿 [BIFURCATION] {self.names[parent_idx]} -> Рожден: [{new_name}] (Всего: {self.active_count})")
        return new_idx

    def update_cycle(self, leader_idx: int, child_idx: int, beta_order: float, beta_chaos: float, 
                     torus_u: float, torus_v: float, smooth_depth: float, lead_sign: float, 
                     rx_sagitta: float, subjects: list[BrainSubject], resolved_parent_map: dict):
        if not self.initialized or len(self.c_bases) == 0: return

        with torch.inference_mode():
            if self.chaos_enabled:
                chaos_drive = float(np.clip(1.0 - beta_chaos, 0.0, 1.0))
                order_drive = float(np.clip(beta_order, 0.0, 1.0))
                safe_l = max(0, min(leader_idx, len(self.c_bases) - 1))
                self.energies[safe_l] = np.clip(self.energies[safe_l] + chaos_drive * 0.07 - order_drive * 0.03, 0.0, 5.0)
                split_threshold = 0.70 / (FEIGENBAUM_DELTA ** self.depths[safe_l])

                if self.energies[safe_l] > split_threshold and (time.time() - self.last_split_time > 2.0):
                    new_child_idx = self.bifurcate_concept(safe_l, torus_u, torus_v, lead_sign)
                    if new_child_idx != -1:
                        self.last_split_time = time.time()
                        for s in subjects: s.heterarchy.inherit_synapses(safe_l, new_child_idx)

                if order_drive > 0.80 and self.active_count > 2 and (time.time() - self.last_split_time > 3.0):
                    removed_name = self.names.pop()
                    self.c_bases.pop()
                    self.active_count = len(self.c_bases)
                    self.last_split_time = time.time()
                    print(f"🕳️ [CRISIS] Порядок подавил шум: [{removed_name}] схлопнут обратно.")

            current_len = len(self.c_bases)
            if current_len == 0: return
            l_idx = max(0, min(leader_idx, current_len - 1))
            c_idx = max(0, min(child_idx, current_len - 1))
            base_form, base_style = self.c_bases[l_idx], self.c_bases[c_idx]

            target = torch.zeros_like(base_form)
            half_d = target.shape[-1] // 2
            if resolved_parent_map.get(c_idx) == l_idx or resolved_parent_map.get(l_idx) == c_idx:
                target[:, :half_d] = base_form[:, :half_d]
                target[:, half_d:] = base_style[:, half_d:]
            else:
                target = base_form

            alpha_f3 = float(np.clip(1.0 - beta_order * 0.7 + rx_sagitta * 0.2, 0.15, 0.85))
            alpha_f4 = float(np.clip(1.0 - beta_chaos * 0.7 - rx_sagitta * 0.2, 0.15, 0.85))
            mean_alpha = (alpha_f3 + alpha_f4) / 2.0

            with self.lock:
                if self.latent_active is None: self.latent_active = target.clone()
                else: self.latent_active = self.latent_active * (1.0 - mean_alpha) + target * mean_alpha

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
                    
                    gen = torch.Generator(device=DEVICE).manual_seed(42)
                    u = torch.randn_like(self.c_bases[0], generator=gen)
                    v = torch.randn_like(self.c_bases[0], generator=gen)
                    self.dir_u = u / torch.norm(u, dim=-1, keepdim=True)
                    self.dir_v = v / torch.norm(v, dim=-1, keepdim=True)
                    
                    with self.lock:
                        if len(self.c_bases) > 0:
                            self.latent_active = self.c_bases[0].clone()
                    self.initialized = True
                    print(f"✅ [EQUAL POOL] Diffusion Backend активен. Загружено базовых промптов: {len(self.c_bases)}")
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
                    'strength': max(0.20, min(0.85, s_val))
                }
                if self.speed == "fast": req['num_inference_steps'] = 3
                self.conn.send(req)
                resp = self.conn.recv()

                with self.lock:
                    if isinstance(resp, np.ndarray):
                        if resp.shape[:2] != (self.img_h, self.img_w):
                            resp = cv2.resize(resp, (self.img_w, self.img_h))
                        treated = apply_color_surgery(resp, img.astype(np.float32)) if self.use_color else resp
                        self.current_rgb = cv2.addWeighted(self.current_rgb, 0.25, treated, 0.75, 0)
                        self.frame_id += 1

                times.append(time.time() - t0)
                if len(times) > 5: times.pop(0)
                self.fps = 1.0 / (np.mean(times) + 1e-6)
            except Exception:
                self.initialized = False
                time.sleep(0.5)

def parse_user_setup(users_str: str, pilot_kwargs: dict) -> list[BrainSubject]:
    subjects = []
    blocks = users_str.split(";")
    for b in blocks:
        if not b.strip(): continue
        p = b.strip().split(":")
        sub_id = p[0].strip()
        reg_defs = p[1].strip().split(",")
        reg_map = {}
        for rd in reg_defs:
            if not rd.strip(): continue
            r_name, d_idx = rd.strip().split("=")
            reg_map[r_name.strip()] = int(d_idx.strip())
        subjects.append(BrainSubject(subject_id=sub_id, regions_map=reg_map, max_capacity=MAX_CONCEPTS_CAPACITY, **pilot_kwargs))
    if not subjects:
        subjects = [BrainSubject("User1", {"FCz": 0}, max_capacity=MAX_CONCEPTS_CAPACITY, **pilot_kwargs)]
    return subjects

# =====================================================================
# 5. ГЛАВНЫЙ ИСПОЛНИТЕЛЬНЫЙ ЦИКЛ (60 FPS СКОРОСТЬ)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="NeuroCanvas: Real-Time 4-Axis Continuous Flight (60 FPS)")
    parser.add_argument('--config', type=str, default="swarm_config.json")
    parser.add_argument('--sim', action='store_true', default=False)
    parser.add_argument('--concepts', type=int, default=1)
    parser.add_argument('--start-prompt', type=str, default="ancient medieval stone castle fortress towers, daytime, sharp focus, 8k")
    parser.add_argument('--chaos', action='store_true')
    parser.add_argument('--calib-mode', type=str, default="auto", choices=["auto", "motion", "semantic"])
    parser.add_argument('--move-directions', type=int, default=4, choices=[2, 4])
    parser.add_argument('--calib-sigma', type=float, default=3.5)
    parser.add_argument('--calib-seconds', type=float, default=6.0)
    parser.add_argument('--calib-cycles', type=int, default=3)
    parser.add_argument('--users', type=str, default="Dmitry:FCz=0")
    parser.add_argument('--sensory-sub', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default="lcm", choices=["lcm", "turbo", "sdxl-turbo"])
    parser.add_argument('--speed', type=str, default="fast", choices=["fast", "quality"])
    parser.add_argument('--strength-high', type=float, default=0.85)
    parser.add_argument('--strength-low', type=float, default=0.60)
    parser.add_argument('--gamma-100', action='store_true')
    parser.add_argument('--use-kinematics', action='store_true', default=True)
    
    # Настройки физики
    parser.add_argument('--sensitivity', type=float, default=0.05)
    parser.add_argument('--max-speed', type=float, default=9.0)
    parser.add_argument('--fwd-scale', type=float, default=0.2)
    parser.add_argument('--strafe-scale', type=float, default=0.2)
    parser.add_argument('--turn-scale', type=float, default=0.5)
    parser.add_argument('--intent-gain', type=float, default=1.5)
    
    parser.add_argument('--hardcoded-bots', type=int, default=1)
    parser.add_argument('--jepa-bots', type=int, default=0)
    parser.add_argument('--weights', type=str, default="monty_ltm_weights.pt")
    parser.add_argument('--force-recalib', action='store_true')
    parser.add_argument('--no-taesd', action='store_true')
    parser.add_argument('--no-color', action='store_true')
    parser.add_argument('--sps', type=int, default=250, choices=[250, 500])
    args = parser.parse_args()

    pilot_kwargs = {
        'sensitivity': args.sensitivity, 'max_speed': args.max_speed,
        'fwd_scale': args.fwd_scale, 'strafe_scale': args.strafe_scale,
        'turn_scale': args.turn_scale, 'intent_gain': args.intent_gain
    }

    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f: cfg = json.load(f)
        subjects = [
            BrainSubject(u["id"], u["regions"], max_capacity=MAX_CONCEPTS_CAPACITY, **pilot_kwargs)
            for u in cfg["users"]
        ] if "users" in cfg else parse_user_setup(args.users, pilot_kwargs)
    else:
        subjects = parse_user_setup(args.users, pilot_kwargs)

    collective = OverBrainCollective(subjects)
    primary_brain = subjects[0]

    # Проверка наличия регионов по ролям
    has_fcz_global = any(s.has_region("FCz") for s in subjects)
    has_afz_global = any(s.has_region("AFz") for s in subjects)

    if args.calib_mode == "auto":
        is_motion_calib = has_fcz_global and not has_afz_global
    else:
        is_motion_calib = (args.calib_mode == "motion")

    BASE_PROMPTS = [
        "giant snowy mountain peak, rocky cliffs, clear blue sky, sharp focus, 8k",
        "dense lush green tropical jungle, giant trees, vines, sunlight piercing through leaves, 8k",
        "ancient medieval stone castle fortress towers, daytime, sharp focus, 8k",
        "open stormy dark blue ocean, pure water surface, giant ocean waves, sea foam, no land, 8k"
    ]
    
    if is_motion_calib:
        initial_prompts = [args.start_prompt]
        calib_block_names = MOTION_NAMES[:args.move_directions]
        active_memory_classes = args.move_directions
    else:
        num_c = max(2, args.concepts) if args.concepts == 1 else args.concepts
        initial_prompts = [BASE_PROMPTS[i % len(BASE_PROMPTS)] for i in range(num_c)]
        calib_block_names = [ALL_NAMES[i % len(ALL_NAMES)] for i in range(num_c)]
        active_memory_classes = len(calib_block_names)

    calib_desc = "4-AXIS FCz FLIGHT (SMA)" if is_motion_calib else f"AFz СЕМАНТИКА ({active_memory_classes} классов)"
    print(f"👥 [COLLECTIVE] Режим системы: [{calib_desc}] | Участников: {len(subjects)} | Цель: {args.calib_sigma}σ")

    agent = None
    if args.sim:
        agent = SyntheticAutonomousAgent(
            config_path=args.config, num_hardcoded=args.hardcoded_bots, num_jepa=args.jepa_bots, 
            num_concepts=max(2, active_memory_classes), sps=args.sps
        )
        if args.sensory_sub: agent.set_sensory_substitution(True)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("NeuroCanvas: Real-Time 4-Axis Continuous Flight (60 FPS)")
    clock = pygame.time.Clock()
    font_b = pygame.font.SysFont("consolas", 14, bold=True)
    font_s = pygame.font.SysFont("consolas", 11)
    font_large = pygame.font.SysFont("consolas", 20, bold=True)
    font_huge = pygame.font.SysFont("consolas", 24, bold=True)

    gamma_max = 100.0 if args.gamma_100 else 65.0
    engine = HeterarchicalBrainEngine(gamma_max=gamma_max)
    engine.start()

    worker = EqualPoolChaosWorker(
        initial_prompts=initial_prompts, initial_names=[ALL_NAMES[0]] if is_motion_calib else calib_block_names,
        port=6000, mode=args.mode, speed=args.speed, use_taesd=not args.no_taesd, use_color=not args.no_color
    )
    if args.chaos: worker.chaos_enabled = True

    clip_teacher = VisualCLIPTeacher(initial_prompts)

    is_calibrating = not args.chaos
    if not args.force_recalib and os.path.exists(args.weights) and not args.chaos:
        loaded, cnt = primary_brain.heterarchy.load_from_file(args.weights)
        if loaded and cnt == active_memory_classes:
            is_calibrating = False

    calib_step_idx = 0
    calib_cycle_count = 0
    epoch_start_time = time.time()
    calib_data = {'y_fwd': [], 'y_bwd': [], 'x_rgt': [], 'x_lft': []} if is_motion_calib else {i: [] for i in range(active_memory_classes)}
        
    current_d_prime, dp_axis_y, dp_axis_x = 0.0, 0.0, 0.0
    analog_evidence = np.zeros(active_memory_classes, dtype=np.float32)

    svd_spectrum = np.zeros(4)
    lead_causal_sign = 0.0
    cur_probs = np.zeros(MAX_CONCEPTS_CAPACITY, dtype=np.float32)
    clip_lock = threading.Lock()

    # ОПТИМИЗАЦИЯ: CLIP опрашивает кадры 5 раз в секунду (не подвешивая CUDA)
    def async_clip_worker():
        nonlocal cur_probs
        last_fid = -1
        while True:
            try:
                with worker.lock:
                    img_to_eval = worker.current_rgb.copy()
                    c_fid, c_cnt = worker.frame_id, worker.active_count
                if c_fid != last_fid and np.max(img_to_eval) > 0:
                    last_fid = c_fid
                    probs = clip_teacher.classify(img_to_eval, c_cnt)
                    with clip_lock: cur_probs[:c_cnt] = probs
                    if agent: agent.update_visual_state(probs)
                time.sleep(0.20)
            except Exception: time.sleep(0.2)

    threading.Thread(target=async_clip_worker, daemon=True).start()
    FULL_DUPLEX_MODE = args.sensory_sub
    frame_counter = 0

    try:
        while True:
            # Четкий 60 FPS тик, скорость синхронизирована с лабиринтом
            dt = clock.tick(60) / 1000.0
            frame_counter += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT: raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: worker.toggle_chaos()
                    elif event.key == pygame.K_m: 
                        FULL_DUPLEX_MODE = not FULL_DUPLEX_MODE
                        if agent: agent.set_sensory_substitution(FULL_DUPLEX_MODE)
                    elif event.key == pygame.K_r:
                        is_calibrating = True
                        calib_step_idx, calib_cycle_count = 0, 0
                        analog_evidence.fill(0.0)
                        epoch_start_time = time.time()
                        calib_data = {'y_fwd': [], 'y_bwd': [], 'x_rgt': [], 'x_lft': []} if is_motion_calib else {i: [] for i in range(active_memory_classes)}

            frame = engine.get_frame()
            has_live_eeg = (frame.num_live > 0)

            # 1. ОБРАБОТКА НЕЙРОФИЗИОЛОГИИ ВСЕМИ СУБЪЕКТАМИ (32 СЛОТА PAC НА GPU)
            for s in subjects:
                s.process_neurophysiology(frame.nodes, dt, active_memory_classes)

            # 2. РАЗРЕШЕНИЕ ЛИДЕРА И РЕГИОНАЛЬНЫХ НОД
            sorted_subj = sorted(subjects, key=lambda s: s.confidence, reverse=True)
            leader_brain = sorted_subj[0] if sorted_subj else primary_brain
            child_brain = sorted_subj[1] if len(sorted_subj) > 1 else leader_brain

            leader_idx = leader_brain.decoded_leader_idx
            child_idx = child_brain.decoded_leader_idx

            # Прямая маршрутизация нод по анатомическим ролям:
            fcz_node = leader_brain.get_region_node("FCz", frame.nodes)
            afz_node = leader_brain.get_region_node("AFz", frame.nodes)
            f3_node  = leader_brain.get_region_node("F3", frame.nodes)
            f4_node  = child_brain.get_region_node("F4", frame.nodes)
            fpz_node = leader_brain.get_region_node("Fpz", frame.nodes)

            primary_lead_node = fcz_node if fcz_node is not None else (afz_node if afz_node is not None else frame.nodes[0])

            beta_order = float(f3_node.beta_power) if f3_node is not None else float(primary_lead_node.beta_power)
            beta_chaos = float(f4_node.beta_power) if f4_node is not None else (float(frame.nodes[1].beta_power) if len(frame.nodes) > 1 else 0.5)
            torus_u = float(afz_node.torus_u) if afz_node is not None else float(primary_lead_node.torus_u)
            torus_v = float(afz_node.torus_v) if afz_node is not None else float(primary_lead_node.torus_v)

            # =============================================================
            # ИЗВЛЕЧЕНИЕ КИНЕМАТИКИ: ТОЛЬКО FCz УПРАВЛЯЕТ ПОЛЕТОМ КАМЕРЫ!
            # =============================================================
            if fcz_node is not None:
                # Читаем предрассчитанные на GPU оси kinematics (0 мс на CPU!)
                axes = fcz_node.kinematics
                force_x = float(axes.lx)
                force_y = float(-axes.ly)
                wm_curvature = float(axes.rx)
                temporal_bias = float(axes.ry)
                is_motion_active = True
            else:
                # AFz / F3 / F4: Когнитивные регионы НЕ летают от шума!
                force_x = 0.0
                force_y = 0.0
                wm_curvature = 0.0
                temporal_bias = float(primary_lead_node.kinematics.ry)
                is_motion_active = False

            # Ручной override со стрелок
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:    force_y += 1.0
            if keys[pygame.K_DOWN]:  force_y -= 1.0
            if keys[pygame.K_RIGHT]: force_x += 1.0
            if keys[pygame.K_LEFT]:  force_x -= 1.0
            if keys[pygame.K_PERIOD]: wm_curvature += 0.5
            if keys[pygame.K_COMMA]:  wm_curvature -= 0.5

            # Обновление динамического пилота аватара
            leader_brain.pilot.update(dt, force_x, force_y, wm_curvature, temporal_bias)

            # Сенсомоторный варп диффузии (только при реальном движении)
            if abs(force_x) > 0.05 or abs(force_y) > 0.05 or abs(wm_curvature) > 0.04:
                with worker.lock:
                    worker.current_rgb = apply_manifold_camera_warp(
                        worker.current_rgb, leader_brain.pilot, dt
                    )

            # SVD считается раз в 6 кадров (10 Гц), чтобы видеокарта не тормозила основной поток
            if frame_counter % 6 == 0:
                lead_ripple = torch.tensor(primary_lead_node.iplv_human_ripple, dtype=torch.float32, device=DEVICE)
                ripple_centered = lead_ripple - torch.mean(lead_ripple, dim=0, keepdim=True)
                if torch.sum(torch.abs(ripple_centered)) > 1e-4:
                    S_vals = torch.linalg.svdvals(ripple_centered)
                    svd_spectrum = (S_vals[:4] / (S_vals[0] + 1e-6)).cpu().numpy()
                lead_causal_sign = float(np.mean(primary_lead_node.iplv_human_ripple[:, 0]))

            with worker.lock: 
                rgb_m = worker.current_rgb.copy()
                active_pool_size = worker.active_count

            with clip_lock:   
                live_probs = cur_probs[:active_pool_size].copy()

            safe_step_idx = calib_step_idx % active_memory_classes
            target_now = calib_block_names[safe_step_idx]

            # -----------------------------------------------------------------
            # НЕПРЕРЫВНАЯ АНАЛОГОВАЯ ПРОЕКЦИЯ (DRIFT-DIFFUSION MODEL)
            # -----------------------------------------------------------------
            cur_nav_mag = math.hypot(force_x, force_y)

            if is_motion_calib:
                target_vec = TARGET_UNIT_VECS[safe_step_idx]
                analog_projection = float((force_x * target_vec[0] + force_y * target_vec[1]) / (cur_nav_mag + 1e-6)) if cur_nav_mag > 0.01 else 0.0
            else:
                target_score = float(leader_brain.wm_scores[safe_step_idx])
                other_scores = [float(leader_brain.wm_scores[k]) for k in range(active_memory_classes) if k != safe_step_idx]
                max_other = max(other_scores) if other_scores else 0.0
                analog_projection = float(np.tanh((target_score - max_other) / 25.0))

            persistence_gain = leader_brain.pilot.persistence
            active_evidence_flow = analog_projection * (1.0 + 4.0 * persistence_gain)
            analog_evidence[safe_step_idx] = np.clip(analog_evidence[safe_step_idx] * 0.985 + active_evidence_flow * dt * 0.35, 0.0, 5.0)

            # -----------------------------------------------------------------
            # ДИФФУЗИОННЫЙ ЦИКЛ И КАЛИБРОВКА (КАРТИНКИ ГЕНЕРИРУЮТСЯ ВСЕГДА!)
            # -----------------------------------------------------------------
            agent_pmap = agent.get_parent_map() if agent and FULL_DUPLEX_MODE else {}
            parent_map = collective.resolve_heterarchy(worker.active_count, worker.parent_map, agent_pmap)

            if is_calibrating:
                # В семантическом режиме AFz диффузор ОБЯЗАН показывать текущую мишень:
                calib_vis_idx = safe_step_idx if not is_motion_calib else 0
                worker.update_cycle(
                    leader_idx=calib_vis_idx, child_idx=calib_vis_idx,
                    beta_order=0.0, beta_chaos=0.0,
                    torus_u=torus_u, torus_v=torus_v,
                    smooth_depth=1.0, lead_sign=0.0,
                    rx_sagitta=0.0, subjects=subjects,
                    resolved_parent_map=parent_map
                )
                worker.strength = args.strength_high
                if agent: agent.set_calibration_target(True, safe_step_idx)

                if has_live_eeg:
                    plasticity_mod = float(np.tanh(analog_projection * 2.0)) * persistence_gain
                    for s in subjects:
                        s.learn_contrastive(safe_step_idx, active_memory_classes, lr=0.04 * plasticity_mod, ltd_factor=0.6)

                    if is_motion_calib:
                        # Калибровка 4 осей FCz
                        if safe_step_idx == 0:   calib_data['y_fwd'].append(force_y)
                        elif safe_step_idx == 1: calib_data['y_bwd'].append(force_y)
                        elif safe_step_idx == 2: calib_data['x_rgt'].append(force_x)
                        elif safe_step_idx == 3: calib_data['x_lft'].append(force_x)

                        y_f = np.array(calib_data['y_fwd'][-120:]) if len(calib_data['y_fwd']) >= 10 else np.array([0.0])
                        y_b = np.array(calib_data['y_bwd'][-120:]) if len(calib_data['y_bwd']) >= 10 else np.array([0.0])
                        dp_axis_y = (np.mean(y_f) - np.mean(y_b)) / math.sqrt(0.5 * (np.var(y_f) + np.var(y_b)) + 1e-6)

                        x_r = np.array(calib_data['x_rgt'][-120:]) if len(calib_data['x_rgt']) >= 10 else np.array([0.0])
                        x_l = np.array(calib_data['x_lft'][-120:]) if len(calib_data['x_lft']) >= 10 else np.array([0.0])
                        dp_axis_x = (np.mean(x_r) - np.mean(x_l)) / math.sqrt(0.5 * (np.var(x_r) + np.var(x_l)) + 1e-6)

                        current_d_prime = max(0.0, (dp_axis_y + dp_axis_x) / 2.0)
                    else:
                        # Семантическая контрастивная калибровка AFz
                        for s in subjects:
                            if s.last_sdr is not None:
                                margin_val = s.heterarchy.compute_contrastive_margin(
                                    s.last_sdr, safe_step_idx, active_memory_classes
                                )
                                calib_data[safe_step_idx].append(margin_val)

                        means = [np.mean(calib_data[k][-60:]) if len(calib_data[k]) >= 10 else 0.0 for k in range(active_memory_classes)]
                        vars_ = [np.var(calib_data[k][-60:]) if len(calib_data[k]) >= 10 else 1.0 for k in range(active_memory_classes)]
                        current_d_prime = max(0.0, np.mean(means) / math.sqrt(np.mean(vars_) + 1e-6))

                    elapsed = time.time() - epoch_start_time
                    if elapsed >= args.calib_seconds:
                        epoch_start_time = time.time()
                        calib_step_idx += 1
                        if calib_step_idx >= active_memory_classes:
                            calib_step_idx = 0
                            calib_cycle_count += 1
                            if current_d_prime >= args.calib_sigma and calib_cycle_count >= args.calib_cycles:
                                is_calibrating = False
                                leader_brain.heterarchy.save_to_file(args.weights, calib_block_names)
                                if agent: agent.set_calibration_target(False)
                                print(f"🎯 [КАЛИБРОВКА ЗАВЕРШЕНА] d'={current_d_prime:.2f}σ достигнуто!")
            else:
                if agent:
                    if FULL_DUPLEX_MODE: agent.update_swarm_priors(leader_brain.wm_scores)
                    else: agent.update_swarm_priors(np.zeros_like(leader_brain.wm_scores))

                worker.update_cycle(
                    leader_idx=leader_idx, child_idx=child_idx,
                    beta_order=beta_order, beta_chaos=beta_chaos,
                    torus_u=torus_u, torus_v=torus_v,
                    smooth_depth=1.0, lead_sign=lead_causal_sign,
                    rx_sagitta=wm_curvature, subjects=subjects,
                    resolved_parent_map=parent_map
                )

                active_drive = float(np.clip(1.0 - beta_order, 0.0, 1.0))
                base_strength = args.strength_low + (args.strength_high - args.strength_low) * active_drive

                if args.use_kinematics:
                    worker.strength = float(np.clip(base_strength + temporal_bias * 0.15, 0.10, 0.99))
                else:
                    worker.strength = float(np.clip(base_strength, 0.10, 0.99))

            # =============================================================
            # РЕНДЕРИНГ ИНТЕРФЕЙСА (ВСЕ 8 ПАНЕЛЕЙ)
            # =============================================================
            screen.fill((10, 14, 20))

            # 1. Полотно диффузии (Слева)
            img_x, img_y = 370, 40
            surf_diff = pygame.image.frombuffer(rgb_m.tobytes(), (worker.img_w, worker.img_h), 'RGB')
            if (worker.img_w, worker.img_h) != (512, 384):
                surf_diff = pygame.transform.scale(surf_diff, (512, 384))
            screen.blit(surf_diff, (img_x, img_y))
            pygame.draw.rect(screen, (40, 50, 70), (img_x, img_y, 512, 384), 2, border_radius=8)

            # 2. L4 32-SDR Карта
            if leader_brain and leader_brain.last_sdr_first is not None:
                cur_sdr_img = leader_brain.last_sdr_first[-1].view(64, 64).cpu().numpy() * 255.0
                sdr_surf = pygame.surfarray.make_surface(cv2.resize(cur_sdr_img, (140, 140)).astype(np.uint8))
                screen.blit(sdr_surf, (img_x, 445))
            screen.blit(font_s.render(f"L4 32-SDR Trajectory [{leader_brain.subject_id}]", True, (0, 255, 200)), (img_x, 428))

            # 3. НАВИГАЦИОННЫЙ РАДАР (FCz Полет vs AFz Тор)
            dbg_x, dbg_y, dbg_w, dbg_h = img_x + 160, 430, 400, 175
            pygame.draw.rect(screen, (14, 18, 26), (dbg_x, dbg_y, dbg_w, dbg_h), border_radius=6)
            
            radar_color = (0, 255, 180) if analog_projection > 0.4 else (40, 70, 100)
            pygame.draw.rect(screen, radar_color, (dbg_x, dbg_y, dbg_w, dbg_h), 2 if analog_projection > 0.4 else 1, border_radius=6)
            
            rx_c, ry_c, rc = dbg_x + 295, dbg_y + 88, 68
            pygame.draw.circle(screen, (20, 28, 40), (rx_c, ry_c), rc)
            pygame.draw.circle(screen, (40, 60, 85), (rx_c, ry_c), rc, 1)
            pygame.draw.line(screen, (30, 45, 60), (rx_c - rc, ry_c), (rx_c + rc, ry_c), 1)
            pygame.draw.line(screen, (30, 45, 60), (rx_c, ry_c - rc), (rx_c, ry_c + rc), 1)

            if is_motion_active:
                # FCz: Навигация в 4-осевом пространстве
                screen.blit(font_s.render("N", True, (0, 255, 120)), (rx_c - 4, ry_c - rc - 14))
                screen.blit(font_s.render("S", True, (150, 150, 170)), (rx_c - 4, ry_c + rc + 2))
                screen.blit(font_s.render("E", True, (150, 150, 170)), (rx_c + rc + 4, ry_c - 6))
                screen.blit(font_s.render("W", True, (150, 150, 170)), (rx_c - rc - 14, ry_c - 6))

                # 1. Стрелка-мишень (Желтая)
                if is_calibrating and is_motion_calib:
                    t_deg = TARGET_ANGLES_DEG[safe_step_idx]
                    t_rad = math.radians(t_deg)
                    tx_end = rx_c + int((rc - 8) * math.cos(t_rad))
                    ty_end = ry_c - int((rc - 8) * math.sin(t_rad))
                    pygame.draw.line(screen, (255, 220, 50), (rx_c, ry_c), (tx_end, ty_end), 5)
                    pygame.draw.circle(screen, (255, 255, 100), (tx_end, ty_end), 6)

                # 2. Живая стрелка мозга (Бирюзовая)
                if cur_nav_mag > 0.02:
                    norm_len = min(1.0, cur_nav_mag)
                    lx_end = rx_c + int(norm_len * (rc - 8) * (force_x / cur_nav_mag))
                    ly_end = ry_c - int(norm_len * (rc - 8) * (force_y / cur_nav_mag))
                    col_r = int(np.clip((1.0 - analog_projection) * 120, 0, 255))
                    col_g = int(np.clip((analog_projection + 1.0) * 127, 80, 255))
                    col_b = int(np.clip((1.0 - analog_projection) * 255, 60, 255))
                    pygame.draw.line(screen, (col_r, col_g, col_b), (rx_c, ry_c), (lx_end, ly_end), 4)
                    pygame.draw.circle(screen, (255, 255, 255), (lx_end, ly_end), 5)

                screen.blit(font_b.render("FCz НАВИГАТОР (SMA):", True, (0, 255, 200)), (dbg_x + 10, dbg_y + 8))
            else:
                # AFz: Когнитивный фазовый тор Джанаты (u, v)
                screen.blit(font_s.render("u:0", True, (0, 255, 120)), (rx_c - 8, ry_c - rc - 14))
                screen.blit(font_s.render("u:π", True, (150, 150, 170)), (rx_c - 8, ry_c + rc + 2))
                screen.blit(font_s.render("v:π/2", True, (150, 150, 170)), (rx_c + rc + 4, ry_c - 6))
                screen.blit(font_s.render("v:3π/2", True, (150, 150, 170)), (rx_c - rc - 25, ry_c - 6))

                tu_x, tv_y = math.cos(torus_u), math.sin(torus_v)
                pygame.draw.line(screen, (0, 255, 255), (rx_c, ry_c), (rx_c + int(tu_x * (rc - 8)), ry_c - int(tv_y * (rc - 8))), 4)
                pygame.draw.circle(screen, (255, 255, 255), (rx_c + int(tu_x * (rc - 8)), ry_c - int(tv_y * (rc - 8))), 5)

                screen.blit(font_b.render("AFz КОГНИТИВНЫЙ ТОР (u, v):", True, (0, 255, 200)), (dbg_x + 10, dbg_y + 8))

            if is_calibrating:
                screen.blit(font_huge.render(f"МИШЕНЬ: {target_now}", True, (255, 220, 50)), (dbg_x + 10, dbg_y + 34))
                align_pct = int((analog_projection + 1.0) * 50.0)
                screen.blit(font_s.render(f"Выравнивание с целью: {align_pct}%", True, (200, 220, 255)), (dbg_x + 10, dbg_y + 70))
                cur_ev = analog_evidence[safe_step_idx]
                screen.blit(font_s.render(f"Импульс: x{(1.0 + leader_brain.pilot.persistence * 4.0):.1f}", True, (200, 220, 240)), (dbg_x + 10, dbg_y + 95))
                screen.blit(font_b.render(f"Интеграл DDM: {cur_ev:.2f} / 3.00", True, (100, 255, 100)), (dbg_x + 10, dbg_y + 115))
            else:
                for i, name in enumerate(calib_block_names[:min(4, active_memory_classes)]):
                    score = leader_brain.wm_scores[i]
                    col_bar = (0, 255, 180) if score > 15.0 else (80, 80, 90)
                    by = dbg_y + 34 + i * 22
                    screen.blit(font_s.render(f"{name[:8]:8s}", True, (200, 200, 200)), (dbg_x + 10, by))
                    bar_len = int((score / 100.0) * 65)
                    pygame.draw.rect(screen, (30, 35, 45), (dbg_x + 65, by + 2, 65, 10))
                    if bar_len > 0: pygame.draw.rect(screen, col_bar, (dbg_x + 65, by + 2, bar_len, 10))
                    screen.blit(font_s.render(f"{score:4.1f}%", True, col_bar), (dbg_x + 135, by))

            chaos_txt = "CHAOS BIFURCATIONS: [ACTIVE]" if worker.chaos_enabled else "CHAOS BIFURCATIONS: [OFF] (SPACE)"
            col_chaos = (255, 100, 255) if worker.chaos_enabled else (120, 120, 140)
            screen.blit(font_s.render(chaos_txt, True, col_chaos), (dbg_x + 10, dbg_y + 152))

            # 4. Панель Калибровки / LTM (Слева)
            panel_x, panel_y = 20, 40
            pygame.draw.rect(screen, (16, 22, 32), (panel_x, panel_y, 330, 270), border_radius=8)

            if is_calibrating:
                border_col = (255, 120, 40) if has_live_eeg else (180, 50, 50)
                pygame.draw.rect(screen, border_col, (panel_x, panel_y, 330, 270), 2, border_radius=8)
                calib_hdr = "5-SIGMA FCz (2 ОСИ R^2)" if is_motion_calib else "5-SIGMA AFz СЕМАНТИКА"
                screen.blit(font_large.render(calib_hdr, True, (255, 180, 50)), (panel_x + 15, panel_y + 12))
                
                rem_time = max(0.0, args.calib_seconds - (time.time() - epoch_start_time))
                screen.blit(font_b.render(f"МИШЕНЬ: [{target_now}] ({rem_time:.1f}s)", True, (255, 255, 100)), (panel_x + 15, panel_y + 45))
                screen.blit(font_s.render(f"Цикл: {calib_cycle_count}/{args.calib_cycles} | Порог: {args.calib_sigma}σ", True, (200, 200, 200)), (panel_x + 15, panel_y + 70))
                
                col_dp = (100, 255, 100) if current_d_prime >= args.calib_sigma else (255, 180, 50)
                if is_motion_calib:
                    screen.blit(font_b.render(f"Разделимость: Y={dp_axis_y:.2f}σ | X={dp_axis_x:.2f}σ", True, col_dp), (panel_x + 15, panel_y + 95))
                else:
                    screen.blit(font_b.render(f"Маржа Фишера d': {current_d_prime:.2f}σ", True, col_dp), (panel_x + 15, panel_y + 95))
                
                dp_bar_len = int(np.clip(current_d_prime / args.calib_sigma, 0.0, 1.0) * 290)
                pygame.draw.rect(screen, (40, 50, 60), (panel_x + 15, panel_y + 120, 290, 15))
                pygame.draw.rect(screen, col_dp, (panel_x + 15, panel_y + 120, dp_bar_len, 15))
            else:
                pygame.draw.rect(screen, (0, 255, 200), (panel_x, panel_y, 330, 270), 1, border_radius=8)
                screen.blit(font_b.render("LTM RETENTION (ОБУЧЕН)", True, (0, 255, 200)), (panel_x + 12, panel_y + 12))
                for i, name in enumerate(calib_block_names[:min(10, active_memory_classes)]):
                    score = leader_brain.ltm_scores[i]
                    col_s = (100, 255, 100) if score >= 75.0 else (200, 200, 220)
                    screen.blit(font_s.render(f"{name:10s}: {score:4.1f}%", True, col_s), (panel_x + 15, panel_y + 38 + i * 22))

            # 5. Телеметрия осей (Слева внизу)
            c_x, c_y = 20, 330
            pygame.draw.rect(screen, (16, 22, 32), (c_x, c_y, 330, 270), border_radius=8)
            pygame.draw.rect(screen, (100, 180, 255), (c_x, c_y, 330, 270), 1, border_radius=8)
            screen.blit(font_b.render("МУЛЬТИРЕГИОНАЛЬНАЯ ТЕЛЕМЕТРИЯ:", True, (100, 180, 255)), (c_x + 12, c_y + 12))

            lead_name = calib_block_names[leader_idx % len(calib_block_names)] if leader_idx < len(calib_block_names) else "Unknown"
            screen.blit(font_b.render(f"Доминант: [{lead_name}] ({leader_brain.subject_id})", True, (100, 255, 100)), (c_x + 15, c_y + 34))
            
            nav_status = f"X={force_x:+.2f} | Y={force_y:+.2f}" if is_motion_active else "ОТКЛЮЧЕН (0.0)"
            screen.blit(font_s.render(f"• FCz (SMA Полет) : {nav_status}", True, (0, 255, 200) if is_motion_active else (140, 140, 140)), (c_x + 15, c_y + 56))
            screen.blit(font_s.render(f"• AFz Тор (u, v)  : ({torus_u:.2f}, {torus_v:.2f})", True, (255, 220, 50)), (c_x + 15, c_y + 78))
            screen.blit(font_s.render(f"• F3 Порядок (β)  : {beta_order:.2f}", True, (100, 255, 120)), (c_x + 15, c_y + 100))
            screen.blit(font_s.render(f"• F4 Хаос (β)     : {beta_chaos:.2f}", True, (255, 100, 255)), (c_x + 15, c_y + 122))
            fpz_txt = f"АКТИВЕН ({fpz_node.gating_ratio:.2f})" if fpz_node is not None else "НЕТ"
            screen.blit(font_s.render(f"• Fpz Ветвление   : {fpz_txt}", True, (200, 220, 255)), (c_x + 15, c_y + 144))
            screen.blit(font_s.render(f"• Theta={frame.theta_freq:.2f}Hz | Delta={frame.delta_freq:.2f}Hz", True, (200, 220, 240)), (c_x + 15, c_y + 190))
            screen.blit(font_s.render(f"• Потоков LSL: {frame.num_live} (FreeEEG16)", True, (180, 180, 200)), (c_x + 15, c_y + 215))

            # 6. ciPLV Спектр 120 ребер (Снизу слева)
            BY, BH = 620, 300
            pygame.draw.rect(screen, (12, 16, 24), (20, BY, 870, BH), border_radius=8)
            pygame.draw.rect(screen, (30, 45, 65), (20, BY, 870, BH), 1, border_radius=8)
            screen.blit(font_b.render("120-EDGE DIRECTED ciPLV (65–100 Гц РИППЛЫ ДИКИ, ПИК 89.5 Гц)", True, (0, 255, 200)), (35, BY + 15))

            g120 = primary_lead_node.iplv_human_ripple[31] if has_live_eeg else np.zeros(120)
            bw, mid_line = 800.0 / 120.0, BY + 150
            for p in range(120):
                val = g120[p]
                bh = int(abs(val) * 110)
                bx = 35 + p * bw
                col = (255, 80, 80) if val < 0 else (80, 255, 180)
                if val >= 0: pygame.draw.rect(screen, col, (bx, mid_line - bh, max(1, int(bw - 1)), bh))
                else: pygame.draw.rect(screen, col, (bx, mid_line, max(1, int(bw - 1)), bh))

            # 7. Causal Manifold & SVD (Снизу справа)
            rx_p, ry_p = 920, 620
            pygame.draw.rect(screen, (20, 15, 30), (rx_p, ry_p, 840, 300), border_radius=8)
            pygame.draw.rect(screen, (255, 100, 200), (rx_p, ry_p, 840, 300), 1, border_radius=8)
            screen.blit(font_b.render("DECODED CAUSAL MANIFOLD (TBT 2.0)", True, (255, 100, 200)), (rx_p + 20, ry_p + 20))

            dir_str = "A ⊃ B (Вглубь / Child)" if lead_causal_sign >= 0.03 else ("B ⊃ A (Наружу / Parent)" if lead_causal_sign <= -0.03 else "A ∥ B (Вбок / Peer)")
            screen.blit(font_s.render(f"Causal Lead (90 Hz ciPLV): {lead_causal_sign:+.3f} -> {dir_str}", True, (200, 220, 220)), (rx_p + 20, ry_p + 60))
            screen.blit(font_s.render("SVD Спектр Рипплов 65–100 Гц:", True, (150, 150, 150)), (rx_p + 20, ry_p + 85))
            for i, sv in enumerate(svd_spectrum):
                color = (0, 255, 180) if sv > 0.22 else (80, 80, 80)
                bh = int(sv * 35)
                pygame.draw.rect(screen, color, (rx_p + 20 + i*65, ry_p + 105 + (35 - bh), 45, bh))
                screen.blit(font_s.render(f"S{i+1}: {sv:.2f}", True, (200, 200, 200)), (rx_p + 20 + i*65, ry_p + 145))

            # 8. Рекурсивный Treemap (Справа)
            map_x, map_y = 950, 40
            map_w, map_h = 800, 560
            pygame.draw.rect(screen, (10, 10, 10), (map_x, map_y, map_w, map_h))
            pygame.draw.rect(screen, (255, 255, 255), (map_x-2, map_y-2, map_w+4, map_h+4), 2)

            boxes, resolved_hierarchy = calculate_emergent_treemap(
                map_x, map_y, map_w, map_h, leader_brain.blended_weights, active_memory_classes, parent_map=parent_map
            )
            treemap_title = f"EMERGENT TREEMAP ({'FCz ПОЛЁТ' if is_motion_calib else 'AFz СЕМАНТИКА'}: {active_memory_classes})"
            screen.blit(font_b.render(treemap_title, True, (255, 255, 255)), (map_x, map_y - 25))

            for c_idx in resolved_hierarchy:
                if c_idx >= active_memory_classes: continue
                bx, by, bw, bh, rank, depth = boxes[c_idx]
                base_col = FRACTAL_COLORS[c_idx % len(FRACTAL_COLORS)]
                darken = max(0.3, 1.0 - depth * 0.25)
                c_color = (int(base_col[0]*darken), int(base_col[1]*darken), int(base_col[2]*darken))

                pygame.draw.rect(screen, c_color, (bx, by, bw, bh))
                pygame.draw.rect(screen, (200, 200, 200), (bx, by, bw, bh), max(1, 3 - depth))
                c_name = calib_block_names[c_idx % len(calib_block_names)]
                owner_tag = f"[{leader_brain.subject_id}] " if leader_brain.decoded_leader_idx == c_idx and leader_brain.confidence > 20.0 else ""
                screen.blit(font_s.render(f"D{worker.depths[c_idx]} (H:{depth})", True, (255, 255, 255)), (bx + 5, by + 5))
                if bw >= 50 and bh >= 35:
                    screen.blit(font_b.render(f"{owner_tag}{c_name}", True, (255, 255, 255)), (bx + 5, by + 20))

            mode_color = (0, 255, 100) if FULL_DUPLEX_MODE else (160, 160, 160)
            mode_str = "SENSORY SUB: [FULL-DUPLEX (SHARED)]" if FULL_DUPLEX_MODE else "SENSORY SUB: [ISOLATED (SELF)]"
            screen.blit(font_b.render(mode_str, True, mode_color), (1450, 15))

            pygame.display.flip()

    finally:
        worker.running = False
        if primary_brain and hasattr(primary_brain, 'heterarchy'):
            primary_brain.heterarchy.save_to_file(args.weights, calib_block_names)
        if agent: agent.stop()
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
