#!/usr/bin/env python3
"""
🧠 NEUROCANVAS × TBP.MONTY: RIGOROUS 2-AXIS SENSORIMOTOR MANIFOLD (FCz)
- Научное обоснование 2D геометрии рабочей памяти:
  * Fan, Wang, Ding, Luo (2024, Nature Human Behaviour, DOI: 10.1038/s41562-024-02047-8)
  * Chen, Zhang, Hu, Min, Wang (2024, Neuron, DOI: 10.1016/j.neuron.2024.07.024)
  * Xie et al. (2022, Science, DOI: 10.1126/science.abm0204)
  * Miller, Lundqvist, Bastos (2018, Neuron, DOI: 10.1016/j.neuron.2018.09.023)
  * Bieri, Bobbitt, Colgin (2014, Neuron, DOI: 10.1016/j.neuron.2014.03.013)
  * Hawkins, Leadholm, Clay (2025/2026, arXiv:2507.05888)
- 100% реализация динамики FCz из neuro_flexible_maze_app.py (persistence, temp_bias, sagitta).
- 5-Сигма калибровка 2D векторов перемещения из прошлого в будущее (d' >= 4.75σ).
- Строгая упаковка в Cortical Messaging Protocol (tbp.monty.cmp.Message).
- Полная сохранность всех аналитических и графических панелей.
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

from neuro_heterarchy_core import HeterarchicalBrainEngine, DEVICE, NodeState
from tbp.monty.cmp import Message
from synthetic_16d_causal_agent import SyntheticAutonomousAgent, CorticalMontage

WIDTH, HEIGHT = 1800, 960
MAX_CONCEPTS_CAPACITY = 16
FEIGENBAUM_DELTA = 4.669201609

ALL_NAMES = ["ГОРА",  "ДЖУНГЛИ", "ЗАМОК", "ОКЕАН", "КОСМОС", "ПЛАНЕТА", "КИБЕРПАНК", "НЕБОСКРЕБ"]
# 4 ортогональных полюса двух непрерывных осей 2D геометрии (Fan et al., 2024; Chen et al., 2024)
MOTION_NAMES = ["ВПЕРЕД", "НАЗАД", "ВЛЕВО", "ВПРАВО"]
MOTION_TARGET_VECS = [
    np.array([ 0.0,  1.0], dtype=np.float32),  # ВПЕРЕД (+Y, Проспекция)
    np.array([ 0.0, -1.0], dtype=np.float32),  # НАЗАД (-Y, Ретроспекция)
    np.array([-1.0,  0.0], dtype=np.float32),  # ВЛЕВО (-X, Латеральный сдвиг)
    np.array([ 1.0,  0.0], dtype=np.float32)   # ВПРАВО (+X, Латеральный сдвиг)
]

AXIS_NAMES = ["ОСЬ Y (ПРОДОЛЬНАЯ)", "ОСЬ X (БОКОВАЯ)"]

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

# =====================================================================
# ТОЧНАЯ ФИЗИКА НАВИГАЦИИ ИЗ neuro_flexible_maze_app.py
# =====================================================================
class DynamicPilot:
    def __init__(self):
        self.x, self.y = 0.0, 0.0
        self.vx, self.vy = 0.0, 0.0
        self.persistence = 0.0
        self.last_ix, self.last_iy = 0.0, 0.0
        self.wm_curvature = 0.0
        self.temporal_bias = 0.0

    def update(self, dt: float, force_x: float, force_y: float, wm_curvature: float, temp_bias: float):
        self.wm_curvature = wm_curvature
        self.temporal_bias = temp_bias

        mag = math.hypot(force_x, force_y)
        if mag > 0.05:
            last_mag = math.hypot(self.last_ix, self.last_iy) + 1e-6
            dot = (force_x * self.last_ix + force_y * self.last_iy) / (mag * last_mag)
            alignment = max(0.0, dot)
        else:
            alignment = 0.0

        self.persistence = self.persistence * 0.95 + 0.05 * alignment * math.tanh(mag * 2.0)
        self.last_ix, self.last_iy = force_x, force_y

        active_boost = 1.0 + self.persistence * 4.0
        base_speed = 3.2 * active_boost

        target_vx = force_x * base_speed
        target_vy = force_y * base_speed

        MAX_SPEED = 9.0
        t_mag = math.hypot(target_vx, target_vy)
        if t_mag > MAX_SPEED:
            target_vx = (target_vx / t_mag) * MAX_SPEED
            target_vy = (target_vy / t_mag) * MAX_SPEED

        self.vx = self.vx * 0.88 + target_vx * 0.12
        self.vy = self.vy * 0.88 + target_vy * 0.12

        self.x += self.vx * dt
        self.y += self.vy * dt

def apply_manifold_camera_warp(img_np: np.ndarray, pilot: DynamicPilot, dt: float = 0.03):
    h, w = img_np.shape[:2]
    zoom = 1.0 + (pilot.vy * dt * 0.22)
    dx = -pilot.vx * w * dt * 0.15
    angle = -pilot.wm_curvature * 12.0 * dt

    if abs(zoom - 1.0) < 0.0008 and abs(dx) < 0.15 and abs(angle) < 0.04:
        return img_np

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, zoom)
    M[0, 2] += dx
    warped = cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return warped

# =====================================================================
# РЕКУРСИВНЫЙ TREEMAP
# =====================================================================
def calculate_emergent_treemap_recursive(x, y, w, h, weights, parent_map, current_node, depth=0):
    boxes = {}
    boxes[current_node] = (int(x), int(y), int(max(15, w)), int(max(15, h)), 0, depth)

    children = [c for c, p in parent_map.items() if p == current_node and c < len(weights) and weights[c] > 0.02]
    if not children:
        return boxes

    children = sorted(children, key=lambda idx: weights[idx], reverse=True)

    margin_x = w * 0.04
    margin_y = h * 0.08
    inner_x = x + margin_x
    inner_y = y + margin_y * 1.5
    inner_w = w - 2 * margin_x
    inner_h = h - margin_y * 2.0

    if inner_w < 15 or inner_h < 15:
        return boxes

    child_weights = np.array([weights[c] for c in children], dtype=np.float32)
    total_cw = np.sum(child_weights)
    proportions = np.ones(len(children)) / len(children) if total_cw <= 1e-6 else child_weights / total_cw

    cur_cx = inner_x
    for i, ch_idx in enumerate(children):
        cw = max(15, inner_w * proportions[i])
        child_boxes = calculate_emergent_treemap_recursive(
            cur_cx, inner_y, cw, inner_h, weights, parent_map, ch_idx, depth + 1
        )
        boxes.update(child_boxes)
        cur_cx += cw
        
    return boxes

def calculate_emergent_treemap(x, y, w, h, active_weights, count, parent_map=None):
    valid_weights = active_weights[:count]
    if parent_map is None:
        parent_map = {i: None for i in range(count)}

    active_indices = [i for i, val in enumerate(valid_weights) if val > 0.02]
    if not active_indices:
        return {0: (x, y, w, h, 0, 0)}, [0]

    roots = [node for node in active_indices if parent_map.get(node) is None or parent_map.get(node) not in active_indices]
    if not roots:
        roots = [max(active_indices, key=lambda idx: valid_weights[idx])]

    root_weights = np.array([valid_weights[r] for r in roots], dtype=np.float32)
    total_rw = np.sum(root_weights)
    proportions = np.ones(len(roots)) / len(roots) if total_rw <= 1e-6 else root_weights / total_rw

    boxes = {}
    cur_rx = x
    for i, root in enumerate(roots):
        rw = max(20, w * proportions[i])
        b = calculate_emergent_treemap_recursive(cur_rx, y, rw, h, valid_weights, parent_map, root, depth=0)
        boxes.update(b)
        cur_rx += rw
        
    resolved = list(boxes.keys())
    orphans = [i for i in active_indices if i not in resolved]
    if orphans:
        ow = w / len(orphans)
        for i, o_idx in enumerate(orphans):
            boxes[o_idx] = (int(x + i*ow), int(y + h - 25), int(ow), 25, 0, 0)
            resolved.append(o_idx)
            
    return boxes, resolved

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
# HTM-КОЛОНКИ (L4 MACROCOLUMNS)
# =====================================================================
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
    def __init__(self, num_nodes: int = 1, max_capacity: int = MAX_CONCEPTS_CAPACITY, 
                 num_columns_per_node: int = 4096, k_active_per_node: int = 80):
        super().__init__()
        self.num_nodes = max(1, int(num_nodes))
        self.num_columns_per_node = num_columns_per_node
        self.k_active_per_node = k_active_per_node
        self.max_capacity = max_capacity
        
        self.total_dim = self.num_columns_per_node * self.num_nodes
        self.total_k_active = self.k_active_per_node * self.num_nodes 
        
        self.nodes = nn.ModuleList([
            CanonicalHTMColumn(node_id=f"Col_{i}", num_columns=num_columns_per_node, k_active=k_active_per_node)
            for i in range(self.num_nodes)
        ])
        self.register_buffer("calcium_trace", torch.zeros(self.total_dim, device=DEVICE))
        self.register_buffer("synaptic_weights", torch.zeros((max_capacity, self.total_dim), device=DEVICE))
        self.register_buffer("membrane_potential", torch.zeros(max_capacity, device=DEVICE))

    def get_current_sdr(self, iplv_gamma_nodes: list[torch.Tensor]):
        sdrs = []
        for i in range(self.num_nodes):
            if i < len(iplv_gamma_nodes):
                sdrs.append(self.nodes[i].compute_sdr(iplv_gamma_nodes[i]))
            else:
                sdrs.append(torch.zeros(self.num_columns_per_node, device=DEVICE))
        full_sdr = torch.cat(sdrs, dim=0)
        return full_sdr, sdrs[0]

    def reset_calcium(self):
        self.calcium_trace.zero_()

    def contrastive_learn(self, target_idx: int, num_active_concepts: int, lr: float = 0.05, ltd_factor: float = 0.5):
        if torch.max(self.calcium_trace) > 1e-4:
            self.synaptic_weights[target_idx] += lr * self.calcium_trace
            self.synaptic_weights[target_idx] = torch.clamp(self.synaptic_weights[target_idx], 0.0, 1.0)
            
            for other_idx in range(num_active_concepts):
                if other_idx != target_idx:
                    self.synaptic_weights[other_idx] -= (lr * ltd_factor) * self.calcium_trace
                    self.synaptic_weights[other_idx] = torch.clamp(self.synaptic_weights[other_idx], 0.0, 1.0)

    def inherit_synapses(self, parent_idx: int, child_idx: int):
        self.synaptic_weights[child_idx] = self.synaptic_weights[parent_idx] * 0.85

    def get_ltm_scores(self, count: int) -> np.ndarray:
        strong_synapses = torch.sum(self.synaptic_weights[:count] > 0.45, dim=1).float()
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
        weights = torch.softmax(self.membrane_potential[:active_count] * 12.0, dim=0).cpu().numpy()
        return weights, wm_scores.cpu().numpy(), ltm_scores

    def compute_contrastive_margin(self, cur_sdr: torch.Tensor, target_idx: int, active_count: int) -> float:
        if active_count <= 1: return 1.0
        with torch.no_grad():
            w_norm = torch.nn.functional.normalize(self.synaptic_weights[:active_count], p=2, dim=1)
            s_norm = torch.nn.functional.normalize(cur_sdr, p=2, dim=0)
            scores = torch.mv(w_norm, s_norm)
            target_score = scores[target_idx]
            other_scores = torch.cat([scores[:target_idx], scores[target_idx+1:]])
            max_other = torch.max(other_scores)
            return float((target_score - max_other).item())

    def save_to_file(self, filepath: str, class_names: list):
        count = len(class_names)
        payload = {
            'format_version': '9.2_bipolar_axes',
            'num_nodes': self.num_nodes,
            'total_dim': self.total_dim,
            'num_concepts': count,
            'class_names': class_names,
            'synaptic_weights': self.synaptic_weights[:count].cpu(),
            'timestamp': time.time()
        }
        torch.save(payload, filepath)
        print(f"💾 [LTM PERSISTENCE] Сохранен банк весов в {filepath} ({count} классов, dim={self.total_dim})")

    def load_from_file(self, filepath: str) -> tuple[bool, int]:
        if not os.path.exists(filepath): return False, 0
        try:
            ckpt = torch.load(filepath, map_location=DEVICE, weights_only=True)
            cnt = min(self.max_capacity, ckpt.get('num_concepts', 0))
            saved_weights = ckpt['synaptic_weights'][:cnt].to(DEVICE)
            saved_dim = saved_weights.shape[1]
            if saved_dim == self.total_dim:
                self.synaptic_weights[:cnt].copy_(saved_weights)
            elif saved_dim > self.total_dim:
                self.synaptic_weights[:cnt].copy_(saved_weights[:, :self.total_dim])
            else:
                self.synaptic_weights[:cnt, :saved_dim].copy_(saved_weights)
            print(f"📂 [LTM VERIFIED] Загружен банк весов: {filepath} ({cnt} классов, dim={self.total_dim}).")
            return True, cnt
        except Exception as e:
            print(f"⚠️ [LTM LOAD ERROR]: {e}")
            return False, 0

# =====================================================================
# АВТОНОМНЫЙ МОЗГ СУБЪЕКТА (BRAIN SUBJECT)
# =====================================================================
class BrainSubject:
    def __init__(self, subject_id: str, regions_map: dict, max_capacity: int = MAX_CONCEPTS_CAPACITY):
        self.subject_id = subject_id
        self.regions_map = regions_map  
        
        num_nodes = max(1, len(regions_map))
        self.heterarchy = FrontalExecutiveHeterarchy(num_nodes=num_nodes, max_capacity=max_capacity).to(DEVICE)
        
        self.decoded_leader_idx = 0
        self.confidence = 0.0
        self.wm_scores = np.zeros(max_capacity, dtype=np.float32)
        self.ltm_scores = np.zeros(max_capacity, dtype=np.float32)
        self.blended_weights = np.zeros(max_capacity, dtype=np.float32)
        self.last_sdr = None
        self.last_sdr_first = None

        # Физический пилот 2D многообразия
        self.pilot = DynamicPilot()

    def process_neurophysiology(self, frame_nodes: list[NodeState], dt: float, active_count: int):
        my_tensors = []
        for region_name, dev_idx in self.regions_map.items():
            if dev_idx < len(frame_nodes):
                my_tensors.append(torch.tensor(frame_nodes[dev_idx].iplv_gamma, dtype=torch.float32, device=DEVICE))

        if not my_tensors:
            my_tensors = [torch.zeros((32, 120), device=DEVICE)]

        with torch.no_grad():
            full_sdr, sdr_first = self.heterarchy.get_current_sdr(my_tensors)
            self.last_sdr = full_sdr
            self.last_sdr_first = sdr_first
            
            weights, wm, ltm = self.heterarchy.predict_evidence(full_sdr, active_count, dt=dt)
            self.blended_weights = weights
            self.wm_scores = wm
            self.ltm_scores = ltm
            self.decoded_leader_idx = int(np.argmax(wm)) if active_count > 0 else 0
            self.confidence = float(np.max(wm))

    process = process_neurophysiology

    def learn_contrastive(self, target_idx: int, num_active: int, lr: float = 0.05, ltd_factor: float = 0.5):
        if self.last_sdr is not None:
            self.heterarchy.contrastive_learn(target_idx, num_active, lr=lr, ltd_factor=ltd_factor)

    def get_region_beta(self, region_name: str, frame_nodes: list[NodeState], default_val: float = 0.5) -> float:
        if region_name in self.regions_map:
            dev_idx = self.regions_map[region_name]
            if dev_idx < len(frame_nodes):
                return float(frame_nodes[dev_idx].beta_power)
        return default_val

    def get_region_node(self, region_name: str, frame_nodes: list[NodeState]):
        if region_name in self.regions_map:
            dev_idx = self.regions_map[region_name]
            if dev_idx < len(frame_nodes):
                return frame_nodes[dev_idx]
        return frame_nodes[0]

    def get_cmp_message(self, node_lead: NodeState, smooth_depth: float) -> Message:
        cy = math.cos(self.pilot.wm_curvature * 1.5)
        sy = math.sin(self.pilot.wm_curvature * 1.5)
        rot_matrix = np.array([
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        current_3d_loc = np.array([self.pilot.x, self.pilot.y, 0.0], dtype=np.float64)

        return Message(
            location=current_3d_loc,
            morphological_features={
                "pose_vectors": rot_matrix,
                "pose_fully_defined": bool(smooth_depth >= 1.8),
                "on_object": 1.0
            },
            non_morphological_features={
                "object_id": int(self.decoded_leader_idx),
                "torus_u": float(node_lead.torus_u),
                "torus_v": float(node_lead.torus_v),
                "temporal_bias": float(self.pilot.temporal_bias),
                "beta_order": float(node_lead.beta_power)
            },
            confidence=float(np.clip(self.confidence / 100.0, 0.0, 1.0)),
            pass_message=True,
            sender_id=self.subject_id,
            sender_type="SM",
            process_features_in_lm=True
        )

# =====================================================================
# НАДМОЗГ: СБОР ТОЛЬКО РЕАЛЬНЫХ СВЯЗЕЙ
# =====================================================================
class OverBrainCollective:
    def __init__(self, subjects: list[BrainSubject]):
        self.subjects = subjects

    def resolve_heterarchy(self, active_count: int, worker_parent_map: dict, agent_parent_map: dict):
        parent_map = {i: None for i in range(active_count)}
        for child, parent in worker_parent_map.items():
            if child < active_count and parent is not None and parent < active_count:
                parent_map[child] = parent
        if agent_parent_map:
            for child, parent in agent_parent_map.items():
                if child < active_count and parent is not None and parent < active_count:
                    parent_map[child] = parent
        return parent_map

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
        
        self.c_bases = []
        self.pooled_bases = []
        
        self.depths = [0] * MAX_CONCEPTS_CAPACITY
        self.energies = [0.0] * MAX_CONCEPTS_CAPACITY
        self.parent_map = {i: None for i in range(MAX_CONCEPTS_CAPACITY)}
        
        self.chaos_enabled = False
        self.latent_active = None
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
        if self.active_count >= MAX_CONCEPTS_CAPACITY: return -1
        
        new_idx = self.active_count
        self.active_count += 1
        
        p_depth = self.depths[parent_idx]
        self.depths[new_idx] = p_depth + 1
        self.energies[parent_idx] = 0.0
        self.energies[new_idx] = 0.0
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
                
                safe_energy_idx = max(0, min(leader_idx, len(self.c_bases) - 1))
                self.energies[safe_energy_idx] = np.clip(self.energies[safe_energy_idx] + chaos_drive * 0.07 - order_drive * 0.03, 0.0, 5.0)
                split_threshold = 0.70 / (FEIGENBAUM_DELTA ** self.depths[safe_energy_idx])

                if self.energies[safe_energy_idx] > split_threshold and (time.time() - self.last_split_time > 2.0):
                    new_child_idx = self.bifurcate_concept(safe_energy_idx, torus_u, torus_v, lead_sign)
                    if new_child_idx != -1:
                        self.last_split_time = time.time()
                        for s in subjects:
                            s.heterarchy.inherit_synapses(safe_energy_idx, new_child_idx)

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

            base_form = self.c_bases[l_idx]
            base_style = self.c_bases[c_idx]

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
                    
                    gen = torch.Generator(device=DEVICE).manual_seed(42)
                    u = torch.randn_like(self.c_bases[0], generator=gen)
                    v = torch.randn_like(self.c_bases[0], generator=gen)
                    self.dir_u = u / torch.norm(u, dim=-1, keepdim=True)
                    self.dir_v = v / torch.norm(v, dim=-1, keepdim=True)
                    self.initialized = True
                    print(f"✅ [EQUAL POOL] Подключен к Diffusion Backend. Активно базовых промптов: {len(self.c_bases)}")
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

def parse_user_setup(users_str: str) -> list[BrainSubject]:
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
        subjects.append(BrainSubject(subject_id=sub_id, regions_map=reg_map))
    if not subjects:
        subjects = [BrainSubject("User1", {"FCz": 0})]
    return subjects

def main():
    parser = argparse.ArgumentParser(description="NeuroCanvas: 2-Axis Sensorimotor Manifold Flight (FCz)")
    parser.add_argument('--config', type=str, default="swarm_config.json")
    parser.add_argument('--sim', action='store_true', default=False)
    
    parser.add_argument('--concepts', type=int, default=1, help="Количество семантических концептов")
    parser.add_argument('--start-prompt', type=str, default="ancient medieval stone castle fortress towers, daytime, sharp focus, 8k")
    parser.add_argument('--chaos', action='store_true', help="Start with chaos bifurcation active immediately")
    
    parser.add_argument('--calib-mode', type=str, default="auto", choices=["auto", "motion", "semantic"],
                        help="'motion' = калибровка двух осей Y и X на FCz, 'semantic' = калибровка понятий AFz")
    parser.add_argument('--calib-sigma', type=float, default=4.75, help="Критерий статистической разделимости d-prime (4.75 для 5-сигма)")
    parser.add_argument('--calib-seconds', type=float, default=6.0, help="Длительность одной эпохи удержания (сек)")
    parser.add_argument('--calib-cycles', type=int, default=3, help="Минимум чередований перед валидацией")
    
    parser.add_argument('--users', type=str, default="Dmitry:FCz=0", 
                        help="CLI config: e.g. 'Dmitry:FCz=0' or 'Dmitry:AFz=0,FCz=1'")
    parser.add_argument('--sensory-sub', action='store_true', default=False, help="Enable concept broadcast to other agents (Full-Duplex)")

    parser.add_argument('--mode', type=str, default="lcm", choices=["lcm", "turbo", "sdxl-turbo"])
    parser.add_argument('--speed', type=str, default="fast", choices=["fast", "quality"])
    parser.add_argument('--strength-high', type=float, default=0.85)
    parser.add_argument('--strength-low', type=float, default=0.60)
    parser.add_argument('--gamma-100', action='store_true')
    parser.add_argument('--use-kinematics', action='store_true', default=True)
    
    parser.add_argument('--hardcoded-bots', type=int, default=1)
    parser.add_argument('--jepa-bots', type=int, default=0)
    parser.add_argument('--weights', type=str, default="monty_ltm_weights.pt")
    parser.add_argument('--force-recalib', action='store_true')
    parser.add_argument('--no-taesd', action='store_true')
    parser.add_argument('--no-color', action='store_true')
    parser.add_argument('--sps', type=int, default=250, choices=[250, 500])
    args = parser.parse_args()

    # Определение режима калибровки
    if args.calib_mode == "auto":
        is_motion_calib = (args.concepts <= 1) or ("FCz" in args.users and "AFz" not in args.users)
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
        calib_block_names = ["ВПЕРЕД (+Y)", "НАЗАД (-Y)", "ВПРАВО (+X)", "ВЛЕВО (-X)"]
        active_memory_classes = 4
    else:
        initial_prompts = [BASE_PROMPTS[i % len(BASE_PROMPTS)] for i in range(args.concepts)]
        calib_block_names = [ALL_NAMES[i % len(ALL_NAMES)] for i in range(args.concepts)]
        active_memory_classes = len(calib_block_names)

    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f: cfg = json.load(f)
        subjects = [BrainSubject(u["id"], u["regions"], max_capacity=MAX_CONCEPTS_CAPACITY) for u in cfg["users"]] if "users" in cfg else parse_user_setup(args.users)
    else:
        subjects = parse_user_setup(args.users)

    collective = OverBrainCollective(subjects)
    leader_brain = subjects[0] if subjects else None

    calib_desc = "2 НЕПРЕРЫВНЫЕ ОСИ FCz (Y: Вперед/Назад, X: Вправо/Влево)" if is_motion_calib else f"СЕМАНТИКА AFz ({active_memory_classes} классов)"
    print(f"👥 [COLLECTIVE] Мозгов: {len(subjects)}. Режим калибровки: [{calib_desc}] (Цель: {args.calib_sigma}σ)")
    for s in subjects:
        print(f"   👤 Субъект [{s.subject_id}] слушает порты: {s.regions_map} (L4 колонок: {s.heterarchy.total_dim})")

    agent = None
    if args.sim:
        agent = SyntheticAutonomousAgent(
            config_path=args.config, num_hardcoded=args.hardcoded_bots, num_jepa=args.jepa_bots, 
            num_concepts=max(2, active_memory_classes), sps=args.sps
        )
        if args.sensory_sub: agent.set_sensory_substitution(True)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NeuroCanvas: 2-Axis Sensorimotor Manifold & Bipolar 5-Sigma Calibration")
    clock = pygame.time.Clock()
    font_b = pygame.font.SysFont("consolas", 14, bold=True)
    font_s = pygame.font.SysFont("consolas", 11)
    font_large = pygame.font.SysFont("consolas", 20, bold=True)

    gamma_max = 100.0 if args.gamma_100 else 65.0
    engine = HeterarchicalBrainEngine(gamma_max=gamma_max)
    engine.start()

    if agent:
        print("⏳ [SWARM HAL] Ожидание инициализации роя...")
        while not agent.is_ready(): time.sleep(0.05)
        start_wait = time.time()
        while engine.shm['num_live'].value < 1 and (time.time() - start_wait < 10.0): time.sleep(0.05)
    elif not args.sim:
        print("⏳ [CORE HAL] Ожидание подключения LSL-потока FreeEEG16...")
        start_wait = time.time()
        while engine.shm['num_live'].value < 1 and (time.time() - start_wait < 5.0):
            time.sleep(0.1)
        if engine.shm['num_live'].value >= 1:
            print(f"📡 [CORE HAL] Обнаружено активных LSL-узлов: {engine.shm['num_live'].value}")
        else:
            print("ℹ️ [CORE HAL] Поиск LSL-потока продолжается в фоновом режиме.")

    worker = EqualPoolChaosWorker(
        initial_prompts=initial_prompts, initial_names=[ALL_NAMES[0]] if is_motion_calib else calib_block_names,
        port=6000, mode=args.mode, speed=args.speed, use_taesd=not args.no_taesd, use_color=not args.no_color
    )
    if args.chaos: worker.chaos_enabled = True

    clip_teacher = VisualCLIPTeacher(initial_prompts)

    # -----------------------------------------------------------------
    # НАУЧНАЯ КАЛИБРОВКА ПО ОСЯМ
    # -----------------------------------------------------------------
    is_calibrating = not args.chaos
    if not args.force_recalib and os.path.exists(args.weights) and not args.chaos:
        if leader_brain:
            loaded, cnt = leader_brain.heterarchy.load_from_file(args.weights)
            if loaded and cnt == active_memory_classes:
                is_calibrating = False

    calib_step_idx = 0
    calib_cycle_count = 0
    epoch_start_time = time.time()
    
    if is_motion_calib:
        calib_data = {
            'y_fwd': [], 'y_bwd': [],
            'x_rgt': [], 'x_lft': []
        }
    else:
        calib_data = {i: [] for i in range(active_memory_classes)}
        
    current_d_prime = 0.0
    dp_axis_y = 0.0
    dp_axis_x = 0.0

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
    FULL_DUPLEX_MODE = args.sensory_sub

    try:
        while True:
            dt = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT: raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        worker.toggle_chaos()
                    elif event.key == pygame.K_m: 
                        FULL_DUPLEX_MODE = not FULL_DUPLEX_MODE
                        if agent: agent.set_sensory_substitution(FULL_DUPLEX_MODE)

            frame = engine.get_frame()
            has_live_eeg = (frame.num_live > 0)

            # 1. РАСЧЕТ КАЖДОГО МОЗГА В ОТДЕЛЬНОСТИ
            for s in subjects:
                s.process_neurophysiology(frame.nodes, dt, active_memory_classes)

            # 2. СБОР СВЯЗЕЙ
            agent_pmap = agent.get_parent_map() if agent and FULL_DUPLEX_MODE else {}
            parent_map = collective.resolve_heterarchy(worker.active_count, worker.parent_map, agent_pmap)

            sorted_subj = sorted(subjects, key=lambda s: s.confidence, reverse=True)
            leader_brain = sorted_subj[0] if sorted_subj else subjects[0]
            child_brain = sorted_subj[1] if len(sorted_subj) > 1 else leader_brain

            leader_idx = leader_brain.decoded_leader_idx
            child_idx = child_brain.decoded_leader_idx

            beta_order = leader_brain.get_region_beta("F3", frame.nodes, default_val=0.5)
            beta_chaos = child_brain.get_region_beta("F4", frame.nodes, default_val=0.5)

            lead_node = leader_brain.get_region_node("FCz" if "FCz" in leader_brain.regions_map else "AFz", frame.nodes)
            torus_u, torus_v = lead_node.torus_u, lead_node.torus_v

            # -------------------------------------------------------------
            # ТОЧНЫЙ ВЕКТОР НАМЕРЕНИЯ ИЗ neuro_flexible_maze_app.py
            # -------------------------------------------------------------
            fcz_node = leader_brain.get_region_node("FCz", frame.nodes)
            axes = fcz_node.gamepad_axes
            
            force_x = float(axes.lx)
            force_y = float(-axes.ly)
            wm_curvature = float(axes.rx)
            temporal_bias = float(axes.ry)

            # Обновление динамического пилота аватара
            leader_brain.pilot.update(dt, force_x, force_y, wm_curvature, temporal_bias)

            # Сенсомоторное перемещение по генеративному холсту диффузии:
            if abs(force_x) > 0.05 or abs(force_y) > 0.05 or abs(wm_curvature) > 0.04:
                with worker.lock:
                    worker.current_rgb = apply_manifold_camera_warp(
                        worker.current_rgb, leader_brain.pilot, dt
                    )

            lead_ripple = torch.tensor(lead_node.iplv_human_ripple, dtype=torch.float32, device=DEVICE)
            ripple_centered = lead_ripple - torch.mean(lead_ripple, dim=0, keepdim=True)
            if torch.sum(torch.abs(ripple_centered)) > 1e-4:
                S_vals = torch.linalg.svdvals(ripple_centered)
                S_norm = (S_vals[:4] / (S_vals[0] + 1e-6)).cpu().numpy()
                svd_spectrum = S_norm
                live_depth = float(np.sum(S_norm > 0.22))
                smooth_depth = max(1.0, min(4.0, live_depth))
            else:
                smooth_depth = 1.0
                svd_spectrum = np.zeros(4)

            lead_causal_sign = float(np.mean(lead_node.iplv_human_ripple[:, 0]))

            with worker.lock: 
                rgb_m = worker.current_rgb.copy()
                active_pool_size = worker.active_count
                diff_names = list(worker.names)

            with clip_lock:   
                live_probs = cur_probs[:active_pool_size].copy()

            # -----------------------------------------------------------------
            # ЛОГИКА 5-СИГМА КАЛИБРОВКИ
            # -----------------------------------------------------------------
            if is_calibrating:
                worker.update_cycle(
                    leader_idx=0, child_idx=0,
                    beta_order=0.0, beta_chaos=0.0,
                    torus_u=0.0, torus_v=0.0,
                    smooth_depth=1.0, lead_sign=0.0,
                    rx_sagitta=0.0, subjects=subjects,
                    resolved_parent_map=parent_map
                )
                worker.strength = args.strength_high
                if agent: agent.set_calibration_target(True, calib_step_idx)

                if has_live_eeg:
                    for s in subjects:
                        s.learn_contrastive(calib_step_idx, active_memory_classes, lr=0.04, ltd_factor=0.6)
                        
                        if is_motion_calib:
                            if calib_step_idx == 0:   # ВПЕРЕД (+Y)
                                calib_data['y_fwd'].append(force_y)
                            elif calib_step_idx == 1: # НАЗАД (-Y)
                                calib_data['y_bwd'].append(force_y)
                            elif calib_step_idx == 2: # ВПРАВО (+X)
                                calib_data['x_rgt'].append(force_x)
                            elif calib_step_idx == 3: # ВЛЕВО (-X)
                                calib_data['x_lft'].append(force_x)
                        else:
                            if s.last_sdr is not None:
                                margin_val = s.heterarchy.compute_contrastive_margin(
                                    s.last_sdr, calib_step_idx, active_memory_classes
                                )
                                calib_data[calib_step_idx].append(margin_val)

                    elapsed = time.time() - epoch_start_time
                    if elapsed >= args.calib_seconds:
                        epoch_start_time = time.time()
                        calib_step_idx = (calib_step_idx + 1) % active_memory_classes
                        
                        if calib_step_idx == 0:
                            calib_cycle_count += 1
                            
                            if is_motion_calib:
                                y_f = np.array(calib_data['y_fwd'][-120:]) if len(calib_data['y_fwd']) >= 20 else np.array([0.0])
                                y_b = np.array(calib_data['y_bwd'][-120:]) if len(calib_data['y_bwd']) >= 20 else np.array([0.0])
                                mu_yf = np.mean(y_f)
                                mu_yb = np.mean(y_b)
                                std_y = math.sqrt(0.5 * (np.var(y_f) + np.var(y_b)) + 1e-6)
                                dp_axis_y = max(0.0, (mu_yf - mu_yb) / std_y)

                                x_r = np.array(calib_data['x_rgt'][-120:]) if len(calib_data['x_rgt']) >= 20 else np.array([0.0])
                                x_l = np.array(calib_data['x_lft'][-120:]) if len(calib_data['x_lft']) >= 20 else np.array([0.0])
                                mu_xr = np.mean(x_r)
                                mu_xl = np.mean(x_l)
                                std_x = math.sqrt(0.5 * (np.var(x_r) + np.var(x_l)) + 1e-6)
                                dp_axis_x = max(0.0, (mu_xr - mu_xl) / std_x)

                                current_d_prime = min(dp_axis_y, dp_axis_x)
                                print(f"📊 [2-AXIS FCz CALIB] Цикл {calib_cycle_count}: Ось Y (Fwd/Bwd) = {dp_axis_y:.2f}σ | Ось X (Rgt/Lft) = {dp_axis_x:.2f}σ | Общий min = {current_d_prime:.2f}σ (Цель: {args.calib_sigma:.2f}σ)")
                            else:
                                means = [np.mean(calib_data[k][-120:]) if len(calib_data[k]) >= 20 else 0.0 for k in range(active_memory_classes)]
                                vars_ = [np.var(calib_data[k][-120:]) if len(calib_data[k]) >= 20 else 1.0 for k in range(active_memory_classes)]
                                current_d_prime = max(0.0, np.mean(means) / math.sqrt(np.mean(vars_) + 1e-6))
                                print(f"📊 [AFz CALIB] Цикл {calib_cycle_count}: d' = {current_d_prime:.2f} (Цель: {args.calib_sigma:.2f}σ)")

                            if current_d_prime >= args.calib_sigma and calib_cycle_count >= args.calib_cycles:
                                print(f"🎯 [CALIBRATION COMPLETE] Достигнута сверхнаучная разделимость {current_d_prime:.2f}σ по обеим осям!")
                                is_calibrating = False
                                leader_brain.heterarchy.save_to_file(args.weights, calib_block_names)
                                if agent: agent.set_calibration_target(False)
                else:
                    epoch_start_time = time.time()

            else:
                # РАБОЧИЙ РЕЖИМ (ИНФЕРЕНС)
                if agent:
                    if FULL_DUPLEX_MODE:
                        agent.update_swarm_priors(leader_brain.wm_scores)
                    else:
                        agent.update_swarm_priors(np.zeros_like(leader_brain.wm_scores))

                worker.update_cycle(
                    leader_idx=leader_idx, child_idx=child_idx,
                    beta_order=beta_order, beta_chaos=beta_chaos,
                    torus_u=torus_u, torus_v=torus_v,
                    smooth_depth=smooth_depth, lead_sign=lead_causal_sign,
                    rx_sagitta=wm_curvature, subjects=subjects,
                    resolved_parent_map=parent_map
                )

                active_drive = float(np.clip(1.0 - beta_order, 0.0, 1.0))
                base_strength = args.strength_low + (args.strength_high - args.strength_low) * active_drive

                if args.use_kinematics:
                    worker.strength = float(np.clip(base_strength + temporal_bias * 0.15, 0.10, 0.99))
                else:
                    worker.strength = float(np.clip(base_strength, 0.10, 0.99))

            # -------------------------------------------------------------
            # РЕНДЕРИНГ ИНТЕРФЕЙСА (100% СОХРАНЕНИЕ ВСЕХ ПАНЕЛЕЙ)
            # -------------------------------------------------------------
            screen.fill((10, 14, 20))

            # 1. Полотно диффузии
            img_x, img_y = 370, 40
            surf_diff = pygame.image.frombuffer(rgb_m.tobytes(), (worker.img_w, worker.img_h), 'RGB')
            if (worker.img_w, worker.img_h) != (512, 384):
                surf_diff = pygame.transform.scale(surf_diff, (512, 384))
            screen.blit(surf_diff, (img_x, img_y))
            pygame.draw.rect(screen, (40, 50, 70), (img_x, img_y, 512, 384), 2, border_radius=8)

            # 2. L4 SDR Sheet
            if leader_brain and leader_brain.last_sdr_first is not None:
                cur_sdr_img = leader_brain.last_sdr_first.view(64, 64).cpu().numpy() * 255.0
                sdr_surf = pygame.surfarray.make_surface(cv2.resize(cur_sdr_img, (140, 140)).astype(np.uint8))
                screen.blit(sdr_surf, (img_x, 445))
            screen.blit(font_s.render(f"L4 SDR Sheet [{leader_brain.subject_id}]", True, (0, 255, 200)), (img_x, 428))

            # 3. Active Working Memory Pool
            dbg_x, dbg_y, dbg_w, dbg_h = img_x + 160, 430, 400, 170
            pygame.draw.rect(screen, (14, 18, 26), (dbg_x, dbg_y, dbg_w, dbg_h), border_radius=6)
            pygame.draw.rect(screen, (40, 70, 100), (dbg_x, dbg_y, dbg_w, dbg_h), 1, border_radius=6)
            
            pool_title = f"2-AXIS MANIFOLD: {'FCz (Y: Fwd/Bwd, X: Rgt/Lft)' if is_motion_calib else 'AFz (SEMANTICS)'}:"
            screen.blit(font_b.render(pool_title, True, (0, 255, 200)), (dbg_x + 10, dbg_y + 8))

            for i, name in enumerate(calib_block_names[:min(8, active_memory_classes)]):
                score = leader_brain.wm_scores[i]
                col_bar = (0, 255, 180) if score > 15.0 else (80, 80, 90)
                bx = dbg_x + 10 + (i % 2) * 195
                by = dbg_y + 30 + (i // 2) * 22
                
                screen.blit(font_s.render(f"{name[:10]:10s}", True, (200, 200, 200)), (bx, by))
                bar_len = int((score / 100.0) * 80)
                pygame.draw.rect(screen, (30, 35, 45), (bx + 68, by + 2, 70, 10))
                if bar_len > 0: pygame.draw.rect(screen, col_bar, (bx + 68, by + 2, bar_len, 10))
                screen.blit(font_s.render(f"{score:4.1f}%", True, col_bar), (bx + 142, by))

            chaos_txt = "CHAOS BIFURCATIONS: [ACTIVE]" if worker.chaos_enabled else "CHAOS BIFURCATIONS: [OFF] (Press SPACE)"
            col_chaos = (255, 100, 255) if worker.chaos_enabled else (120, 120, 140)
            screen.blit(font_s.render(chaos_txt, True, col_chaos), (dbg_x + 10, dbg_y + 148))

            # 4. LTM / КАЛИБРОВКА (5-СИГМА НАУЧНЫЙ ЭКРАН)
            panel_x, panel_y = 20, 40
            pygame.draw.rect(screen, (16, 22, 32), (panel_x, panel_y, 330, 270), border_radius=8)

            # ГАРАНТИРУЕМ ИНИЦИАЛИЗАЦИЮ target_now ВНЕ ЗАВИСИМОСТИ ОТ СТАТУСА EEG:
            target_now = calib_block_names[calib_step_idx]

            if is_calibrating:
                border_col = (255, 120, 40) if has_live_eeg else (180, 50, 50)
                pygame.draw.rect(screen, border_col, (panel_x, panel_y, 330, 270), 2, border_radius=8)
                
                calib_hdr = "5-SIGMA FCz (2 ОСИ R^2)" if is_motion_calib else "5-SIGMA AFz СЕМАНТИКА"
                screen.blit(font_large.render(calib_hdr, True, (255, 180, 50)), (panel_x + 15, panel_y + 12))
                
                if not has_live_eeg:
                    screen.blit(font_b.render("ОЖИДАНИЕ LSL СИГНАЛА...", True, (255, 80, 80)), (panel_x + 15, panel_y + 45))
                    screen.blit(font_s.render("Подключите FreeEEG / запустите bridge", True, (200, 200, 200)), (panel_x + 15, panel_y + 70))
                else:
                    rem_time = max(0.0, args.calib_seconds - (time.time() - epoch_start_time))
                    screen.blit(font_b.render(f"ФОКУС: [{target_now}] ({rem_time:.1f}s)", True, (255, 255, 100)), (panel_x + 15, panel_y + 45))
                    screen.blit(font_s.render(f"Цикл: {calib_cycle_count}/{args.calib_cycles} | Цель: {args.calib_sigma}σ", True, (200, 200, 200)), (panel_x + 15, panel_y + 70))
                
                col_dp = (100, 255, 100) if current_d_prime >= args.calib_sigma else (255, 100, 100)
                if is_motion_calib:
                    screen.blit(font_b.render(f"Разделимость: Y={dp_axis_y:.1f}σ | X={dp_axis_x:.1f}σ", True, col_dp), (panel_x + 15, panel_y + 95))
                else:
                    screen.blit(font_b.render(f"Маржа Фишера d': {current_d_prime:.2f}σ", True, col_dp), (panel_x + 15, panel_y + 95))
                
                dp_bar_len = int(np.clip(current_d_prime / args.calib_sigma, 0.0, 1.0) * 290)
                pygame.draw.rect(screen, (40, 50, 60), (panel_x + 15, panel_y + 120, 290, 15))
                pygame.draw.rect(screen, col_dp, (panel_x + 15, panel_y + 120, dp_bar_len, 15))

                if is_motion_calib:
                    axis_now = AXIS_NAMES[calib_step_idx // 2]
                    screen.blit(font_s.render(f"Калибровка: {axis_now}", True, (200, 220, 255)), (panel_x + 15, panel_y + 150))
                    screen.blit(font_s.render("Противоположные полюса одной оси в R^2.", True, (180, 180, 180)), (panel_x + 15, panel_y + 168))
                    screen.blit(font_s.render("Georgopoulos (1986); Chen (Neuron 2024).", True, (180, 180, 180)), (panel_x + 15, panel_y + 186))
                else:
                    screen.blit(font_s.render("Удерживайте образ целевого концепта.", True, (180, 180, 180)), (panel_x + 15, panel_y + 150))
                    screen.blit(font_s.render("Chen et al. (Neuron 2024): ортогонализация.", True, (180, 180, 180)), (panel_x + 15, panel_y + 168))
                    screen.blit(font_s.render("Anti-Hebbian LTD разделяет подпространства.", True, (180, 180, 180)), (panel_x + 15, panel_y + 186))
            else:
                pygame.draw.rect(screen, (0, 255, 200), (panel_x, panel_y, 330, 270), 1, border_radius=8)
                screen.blit(font_b.render("LTM RETENTION (КАЛИБРОВАН)", True, (0, 255, 200)), (panel_x + 12, panel_y + 12))
                for i, name in enumerate(calib_block_names[:min(10, active_memory_classes)]):
                    score = leader_brain.ltm_scores[i]
                    col_s = (100, 255, 100) if score >= 75.0 else (200, 200, 220)
                    screen.blit(font_s.render(f"{name:10s}: {score:4.1f}%", True, col_s), (panel_x + 15, panel_y + 38 + i * 22))

            # 5. Cortical Decoder & FCz Flight Telemetry
            c_x, c_y = 20, 330
            pygame.draw.rect(screen, (16, 22, 32), (c_x, c_y, 330, 270), border_radius=8)
            pygame.draw.rect(screen, (100, 180, 255), (c_x, c_y, 330, 270), 1, border_radius=8)
            screen.blit(font_b.render("OVER-BRAIN DECODER (TBT 2.0)", True, (100, 180, 255)), (c_x + 12, c_y + 12))

            lead_name = calib_block_names[leader_idx] if leader_idx < len(calib_block_names) else "Unknown"
            child_name = calib_block_names[child_idx] if child_idx < len(calib_block_names) else "None"
            screen.blit(font_b.render(f"Dominant: [{lead_name}] ({leader_brain.subject_id})", True, (100, 255, 100)), (c_x + 15, c_y + 34))
            screen.blit(font_s.render(f"Subordinate: [{child_name}] ({child_brain.subject_id})", True, (255, 200, 100)), (c_x + 15, c_y + 54))

            screen.blit(font_s.render(f"Torus (Lead): u={torus_u:.2f} | v={torus_v:.2f}", True, (255, 220, 50)), (c_x + 15, c_y + 80))
            screen.blit(font_s.render(f"Drives: Order={beta_order:.2f} | Chaos={beta_chaos:.2f}", True, (150, 255, 200)), (c_x + 15, c_y + 98))

            # FCz НАВИГАЦИОННЫЙ РАДАР (2 НЕПРЕРЫВНЫЕ ОСИ X и Y)
            pygame.draw.rect(screen, (10, 14, 20), (c_x + 10, c_y + 120, 310, 85), border_radius=6)
            pygame.draw.rect(screen, (40, 70, 100), (c_x + 10, c_y + 120, 310, 85), 1, border_radius=6)
            
            nav_radar_cx, nav_radar_cy, n_rad = c_x + 265, c_y + 162, 32
            pygame.draw.circle(screen, (20, 30, 45), (nav_radar_cx, nav_radar_cy), n_rad, 1)
            pygame.draw.line(screen, (30, 45, 60), (nav_radar_cx - n_rad, nav_radar_cy), (nav_radar_cx + n_rad, nav_radar_cy), 1)
            pygame.draw.line(screen, (30, 45, 60), (nav_radar_cx, nav_radar_cy - n_rad), (nav_radar_cx, nav_radar_cy + n_rad), 1)
            
            nav_len = math.hypot(force_x, force_y)
            if nav_len > 0.02:
                nx_p = nav_radar_cx + int(np.clip(force_x, -1.0, 1.0) * (n_rad - 4))
                ny_p = nav_radar_cy - int(np.clip(force_y, -1.0, 1.0) * (n_rad - 4))
                pygame.draw.line(screen, (0, 255, 200), (nav_radar_cx, nav_radar_cy), (nx_p, ny_p), 3)
                pygame.draw.circle(screen, (255, 255, 255), (nx_p, ny_p), 4)

            t_status = "Prospective (Будущее)" if temporal_bias > 0.15 else ("Retrospective (Прошлое)" if temporal_bias < -0.15 else "Equilibrium")
            
            screen.blit(font_b.render("FCz CONTINUOUS 2-AXIS (R^2):", True, (0, 255, 200)), (c_x + 16, c_y + 126))
            screen.blit(font_s.render(f"Ось Y (Fwd/Bwd):  {force_y:+.2f}", True, (200, 220, 240)), (c_x + 16, c_y + 144))
            screen.blit(font_s.render(f"Ось X (Rgt/Lft):  {force_x:+.2f}", True, (200, 220, 240)), (c_x + 16, c_y + 160))
            screen.blit(font_s.render(f"Temp. State (ry): {temporal_bias:+.2f} [{t_status[:4]}]", True, (255, 180, 100)), (c_x + 16, c_y + 176))

            screen.blit(font_s.render(f"Pacing: Theta={frame.theta_freq:.2f}Hz | Delta={frame.delta_freq:.2f}Hz", True, (200, 220, 240)), (c_x + 15, c_y + 215))
            screen.blit(font_s.render(f"Pipeline: {args.mode.upper()} ({worker.fps:.1f} FPS, Str: {worker.strength:.2f})", True, (180, 180, 220)), (c_x + 15, c_y + 235))

            # 6. 120-Edge ciPLV
            BY, BH = 620, 300
            pygame.draw.rect(screen, (12, 16, 24), (20, BY, 870, BH), border_radius=8)
            pygame.draw.rect(screen, (30, 45, 65), (20, BY, 870, BH), 1, border_radius=8)
            screen.blit(font_b.render("120-EDGE DIRECTED ciPLV (65–100 Гц РИППЛЫ ДИКИ, ПИК 89.5 Гц)", True, (0, 255, 200)), (35, BY + 15))

            g120 = lead_node.iplv_human_ripple[31] if has_live_eeg else np.zeros(120)
            bw = 800.0 / 120.0
            mid_line = BY + 150
            for p in range(120):
                val = g120[p]
                bh = int(abs(val) * 110)
                bx = 35 + p * bw
                col = (255, 80, 80) if val < 0 else (80, 255, 180)
                if val >= 0: pygame.draw.rect(screen, col, (bx, mid_line - bh, max(1, int(bw - 1)), bh))
                else: pygame.draw.rect(screen, col, (bx, mid_line, max(1, int(bw - 1)), bh))

            # 7. Causal Manifold & SVD
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

            # 8. Рекурсивный Treemap
            map_x, map_y = 950, 40
            map_w, map_h = 800, 560
            pygame.draw.rect(screen, (10, 10, 10), (map_x, map_y, map_w, map_h))
            pygame.draw.rect(screen, (255, 255, 255), (map_x-2, map_y-2, map_w+4, map_h+4), 2)

            boxes, resolved_hierarchy = calculate_emergent_treemap(
                map_x, map_y, map_w, map_h, 
                leader_brain.blended_weights, active_memory_classes,
                parent_map=parent_map
            )
            
            chaos_indicator = f" [CHAOS ON: δ={FEIGENBAUM_DELTA:.3f}]" if worker.chaos_enabled else ""
            treemap_title = f"EMERGENT TREEMAP ({'FCz 2-AXIS' if is_motion_calib else 'CONCEPTS'}: {active_memory_classes}){chaos_indicator}"
            screen.blit(font_b.render(treemap_title, True, (255, 255, 255)), (map_x, map_y - 25))
            
            for concept_idx in resolved_hierarchy:
                if concept_idx >= active_memory_classes: continue
                bx, by, bw, bh, rank, depth = boxes[concept_idx]
                
                base_col = FRACTAL_COLORS[concept_idx % len(FRACTAL_COLORS)]
                darken = max(0.3, 1.0 - depth * 0.25)
                c_color = (int(base_col[0]*darken), int(base_col[1]*darken), int(base_col[2]*darken))
                
                pygame.draw.rect(screen, c_color, (bx, by, bw, bh))
                pygame.draw.rect(screen, (200, 200, 200), (bx, by, bw, bh), max(1, 3 - depth))
                
                c_name = calib_block_names[concept_idx]
                owner_tag = ""
                if leader_brain.decoded_leader_idx == concept_idx and leader_brain.confidence > 20.0:
                    owner_tag = f"[{leader_brain.subject_id}] "

                txt_line1 = f"L{rank}: D{worker.depths[concept_idx]} (H:{depth})"
                txt_line2 = f"{owner_tag}{c_name}"
                
                surf1 = font_s.render(txt_line1, True, (255, 255, 255))
                surf2 = font_b.render(txt_line2, True, (255, 255, 255))
                
                if bw >= surf2.get_width() + 8 and bh >= 40:
                    screen.blit(surf1, (bx + 5, by + 5))
                    screen.blit(surf2, (bx + 5, by + 20))
                elif bw >= 30 and bh >= 20:
                    screen.blit(font_s.render(txt_line2[:5], True, (255, 255, 255)), (bx + 4, by + 4))

            mode_color = (0, 255, 100) if FULL_DUPLEX_MODE else (160, 160, 160)
            mode_str = "SENSORY SUB: [FULL-DUPLEX (SHARED)]" if FULL_DUPLEX_MODE else "SENSORY SUB: [ISOLATED (SELF)]"
            screen.blit(font_b.render(mode_str, True, mode_color), (1450, 15))

            pygame.display.flip()

    finally:
        worker.running = False
        if leader_brain and hasattr(leader_brain, 'heterarchy'):
            leader_brain.heterarchy.save_to_file(args.weights, calib_block_names)
        if agent: agent.stop()
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
