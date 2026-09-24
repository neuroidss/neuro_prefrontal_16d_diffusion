#!/usr/bin/env python3
"""
🧠 NEUROCANVAS × TBP.MONTY: FULL PREFRONTAL HETERARCHY (F3, F4, AFz, Fpz, FCz)
- F3 (L-PFC): Порядок, синтаксис, категориальные границы, Beta Order.
- F4 (R-PFC): Хаос, дивергентное мышление, бифуркации Фейгенбаума.
- AFz (rmPFC): Когнитивный Тор Джанаты (u, v) и семантическое пространство.
- Fpz (BA 10): Когнитивное ветвление (Cognitive Branching).
- FCz (SMA): Моторная навигация и 4D-действия (Зум/Осмотр, Сагиттальный облет, Сдвиг).
- НИКАКИХ УДАЛЕНИЙ: Все 5 регионов работают одновременно и непрерывно!
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
    np.array([ 0.0,  1.0], dtype=np.float32),
    np.array([ 0.0, -1.0], dtype=np.float32),
    np.array([ 1.0,  0.0], dtype=np.float32),
    np.array([-1.0,  0.0], dtype=np.float32)
]

FRACTAL_COLORS = [
    (20, 20, 60), (30, 80, 120), (100, 100, 110), (150, 100, 50),
    (40, 100, 60), (80, 40, 80), (10, 10, 20), (20, 80, 40),
    (120, 40, 40), (40, 120, 120), (80, 80, 30), (30, 30, 90),
    (90, 40, 90), (40, 90, 40), (110, 70, 30), (50, 50, 50)
]

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
        alignment = 0.5 * (((force_x * self.last_ix + force_y * self.last_iy) / (mag * (math.hypot(self.last_ix, self.last_iy) + 1e-6))) + 1.0) if mag > 0.01 else 0.5
        self.persistence = self.persistence * 0.94 + 0.06 * alignment * math.tanh(mag * 2.0)
        self.last_ix, self.last_iy = force_x, force_y

        active_boost = 1.0 + self.persistence * 4.0
        base_speed = 3.2 * active_boost
        target_vx, target_vy = force_x * base_speed, force_y * base_speed

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

    if abs(zoom - 1.0) < 0.0008 and abs(dx) < 0.15 and abs(angle) < 0.04: return img_np

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, zoom)
    M[0, 2] += dx
    return cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)

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

def calculate_emergent_treemap_recursive(x, y, w, h, weights, parent_map, current_node, depth=0):
    boxes = {}
    boxes[current_node] = (int(x), int(y), int(max(15, w)), int(max(15, h)), 0, depth)
    children = [c for c, p in parent_map.items() if p == current_node and c < len(weights) and weights[c] > 0.01]
    if not children: return boxes

    children = sorted(children, key=lambda idx: weights[idx], reverse=True)
    margin_x, margin_y = w * 0.05, h * 0.08
    inner_x, inner_y = x + margin_x, y + margin_y * 1.5
    inner_w, inner_h = w - 2 * margin_x, h - margin_y * 2.2
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
    active_indices = [i for i, val in enumerate(valid_weights) if val > 0.01]
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
        connected = (spatial_rf >= 0.25).float()
        self.register_buffer("connected_T", connected.T.contiguous())
        self.register_buffer("synapse_counts", torch.sum(connected, dim=1).clamp(min=1.0).view(1, -1))

    def compute_sdr(self, pac_iplv_32x120: torch.Tensor) -> torch.Tensor:
        x_clean = torch.relu(pac_iplv_32x120)
        if torch.max(x_clean) < 1e-5: return torch.full((self.num_slots, self.num_columns), 1e-4, device=DEVICE)
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
            CanonicalHTMColumn(node_id=f"Col_{i}", num_columns=num_columns_per_node, k_active=k_active_per_node, num_slots=num_slots)
            for i in range(self.num_nodes)
        ])
        self.register_buffer("trajectory_weights", torch.full((max_capacity, num_slots, self.total_dim), 0.05, device=DEVICE))
        self.register_buffer("calcium_trajectory", torch.zeros((num_slots, self.total_dim), device=DEVICE))
        self.register_buffer("membrane_potential", torch.zeros(max_capacity, device=DEVICE))
        self.satiation_counter = np.zeros(max_capacity, dtype=np.float32)

    def get_current_sdr(self, iplv_gamma_nodes: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        sdrs = []
        for i in range(self.num_nodes):
            if i < len(iplv_gamma_nodes): sdrs.append(self.nodes[i].compute_sdr(iplv_gamma_nodes[i]))
            else: sdrs.append(torch.zeros((self.num_slots, self.total_dim // self.num_nodes), device=DEVICE))
        return torch.cat(sdrs, dim=-1), sdrs[0]

    def contrastive_learn(self, target_idx: int, num_active: int, lr: float = 0.05, ltd_factor: float = 0.5):
        if torch.max(self.calcium_trajectory) > 1e-5 and abs(lr) > 1e-6:
            self.trajectory_weights[target_idx] += lr * self.calcium_trajectory
            self.trajectory_weights[target_idx] = torch.clamp(self.trajectory_weights[target_idx], 0.01, 1.0)
            for other in range(num_active):
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
        
        # Синаптическая габитуация доминанта (Neural Satiation)
        dom_idx = int(torch.argmax(self.membrane_potential[:active_count]).item())
        self.satiation_counter[dom_idx] += dt
        for i in range(active_count):
            if i != dom_idx: self.satiation_counter[i] = max(0.0, self.satiation_counter[i] - dt * 0.5)

        fatigue = float(np.clip((self.satiation_counter[dom_idx] - 3.0) * 0.12, 0.0, 0.45))
        effective_mp = self.membrane_potential[:active_count].clone()
        effective_mp[dom_idx] -= fatigue

        wm_scores = torch.clamp(effective_mp, 0.0, 1.0) * 100.0
        weights = torch.softmax(effective_mp * 8.0, dim=0).cpu().numpy()
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
            'trajectory_weights': self.trajectory_weights[:len(class_names)].cpu(), 
            'class_names': class_names, 'timestamp': time.time()
        }, filepath)
        print(f"💾 [LTM] Банк весов сохранен: {filepath}")

    def load_from_file(self, filepath: str) -> tuple[bool, int]:
        if not os.path.exists(filepath): return False, 0
        try:
            ckpt = torch.load(filepath, map_location=DEVICE, weights_only=True)
            cnt = min(self.max_capacity, len(ckpt.get('class_names', [])))
            if 'trajectory_weights' in ckpt:
                self.trajectory_weights[:cnt].copy_(ckpt['trajectory_weights'][:cnt].to(DEVICE))
                print(f"📂 [LTM] Загружен банк весов: {filepath} ({cnt} классов)")
                return True, cnt
            return False, 0
        except Exception: return False, 0

class BrainSubject:
    def __init__(self, subject_id: str, regions_map: dict, max_capacity: int = MAX_CONCEPTS_CAPACITY, **pilot_kwargs):
        self.subject_id = subject_id
        self.regions_map = regions_map
        self.heterarchy = FrontalExecutiveHeterarchy(num_nodes=max(1, len(regions_map)), max_capacity=max_capacity).to(DEVICE)
        self.pilot = DynamicPilot(**pilot_kwargs)
        
        self.decoded_leader_idx = 0
        self.confidence = 0.0
        self.wm_scores = np.zeros(max_capacity, dtype=np.float32)
        self.ltm_scores = np.zeros(max_capacity, dtype=np.float32)
        self.blended_weights = np.zeros(max_capacity, dtype=np.float32)
        self.last_sdr = None
        self.last_sdr_first = None

    def has_region(self, region_name: str) -> bool: return region_name in self.regions_map
    def get_region_node(self, region_name: str, frame_nodes: list[NodeState]) -> NodeState | None:
        if region_name in self.regions_map:
            dev_idx = self.regions_map[region_name]
            if 0 <= dev_idx < len(frame_nodes): return frame_nodes[dev_idx]
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

class OverBrainCollective:
    def __init__(self, subjects: list[BrainSubject]): self.subjects = subjects
        
    def resolve_heterarchy(self, active_count: int, worker_parent_map: dict, agent_parent_map: dict, lead_sign: float, leader_idx: int, child_idx: int):
        parent_map = {i: None for i in range(active_count)}
        for child, parent in worker_parent_map.items():
            if child < active_count and parent is not None and parent < active_count: parent_map[child] = parent
                
        if agent_parent_map:
            for child, parent in agent_parent_map.items():
                if child < active_count and parent is not None and parent < active_count and child != parent:
                    parent_map[child] = parent

        if leader_idx < active_count and child_idx < active_count and leader_idx != child_idx:
            if lead_sign >= 0.03:
                if parent_map.get(child_idx) is None: parent_map[child_idx] = leader_idx
            elif lead_sign <= -0.03:
                if parent_map.get(leader_idx) is None: parent_map[leader_idx] = child_idx

        for node in list(parent_map.keys()):
            curr, visited = parent_map[node], {node}
            while curr is not None:
                if curr in visited: parent_map[node] = None; break
                visited.add(curr)
                curr = parent_map.get(curr)
        return parent_map

class VisualCLIPTeacher:
    def __init__(self, text_prompts):
        self.text_prompts = list(text_prompts)
        self.ready = False
        self.model, self.processor, self.text_features = None, None, None
        threading.Thread(target=self._init_bg, daemon=True).start()

    def _init_bg(self):
        try:
            model_id = "openai/clip-vit-large-patch14"
            self.model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE).eval()
            self.processor = CLIPProcessor.from_pretrained(model_id)
            with torch.no_grad():
                inputs = self.processor(text=self.text_prompts, return_tensors="pt", padding=True).to(DEVICE)
                feat = self.model.get_text_features(**inputs)
                self.text_features = feat / feat.norm(dim=-1, keepdim=True)
            self.ready = True
            print("✅ [CLIP] Фоновый учитель активен!")
        except Exception as e:
            print(f"⚠️ [CLIP ERROR]: {e}")

    def classify(self, rgb_image_np: np.ndarray, count: int) -> np.ndarray:
        if not self.ready or rgb_image_np is None or count == 0: return np.zeros(count, dtype=np.float32)
        try:
            inputs = self.processor(images=Image.fromarray(rgb_image_np), return_tensors="pt").to(DEVICE)
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
            with torch.no_grad():
                img_feat = self.model.get_image_features(**inputs)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                logits = (img_feat @ self.text_features[:count].T) * 30.0
                return torch.softmax(logits, dim=-1).cpu().numpy()[0]
        except Exception:
            return np.zeros(count, dtype=np.float32)

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
        self.dir_u, self.dir_v = None, None
        self.last_split_time = time.time()
        
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def toggle_chaos(self):
        with self.lock:
            self.chaos_enabled = not self.chaos_enabled
            print(f"\n🌀 [ХАОС ФЕЙГЕНБАУМА]: {'ВКЛЮЧЕН' if self.chaos_enabled else 'ВЫКЛЮЧЕН'}")

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
        print(f"🌿 [БИФУРКАЦИЯ ДЕРЕВА] {self.names[parent_idx]} -> Потомок: [{new_name}] (D{self.depths[new_idx]})")
        return new_idx

    def update_cycle(self, leader_idx: int, child_idx: int, beta_order: float, beta_chaos: float, 
                     torus_u: float, torus_v: float, smooth_depth: float, lead_sign: float, 
                     rx_sagitta: float, subjects: list, resolved_parent_map: dict):
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

                if order_drive > 0.85 and self.active_count > 2 and (time.time() - self.last_split_time > 3.0):
                    removed_name = self.names.pop()
                    self.c_bases.pop()
                    self.active_count = len(self.c_bases)
                    self.last_split_time = time.time()
                    print(f"🕳️ [КРИЗИС ГРАНИЦ] Схлопнут узел: [{removed_name}]")

            current_len = len(self.c_bases)
            if current_len == 0: return
            l_idx = max(0, min(leader_idx, current_len - 1))
            c_idx = max(0, min(child_idx, current_len - 1))
            
            w_world = self.c_bases[l_idx]
            a_agent = self.c_bases[c_idx]

            # СИНТЕЗ СМЫСЛОВ В ЛАТЕНТНОМ ПРОСТРАНСТВЕ
            if l_idx != c_idx:
                is_child_of_world = (resolved_parent_map.get(c_idx) == l_idx)
                is_parent_of_world = (resolved_parent_map.get(l_idx) == c_idx)
                
                dot_prod = torch.sum(a_agent * w_world, dim=-1, keepdim=True)
                norm_w_sq = torch.sum(w_world * w_world, dim=-1, keepdim=True) + 1e-6
                a_parallel = (dot_prod / norm_w_sq) * w_world
                a_orthogonal = a_agent - a_parallel

                if is_parent_of_world:
                    target = a_agent + 0.40 * w_world
                elif is_child_of_world:
                    gain = float(np.clip(1.1 - beta_order * 0.7 + (1.0 - beta_chaos) * 0.4, 0.3, 1.4))
                    target = w_world + a_orthogonal * gain
                else:
                    # Равноправный синтез (Гора + Джунгли сосуществуют в одном кадре)
                    target = 0.52 * w_world + 0.48 * a_agent
                    
                target = target / torch.norm(target, dim=-1, keepdim=True) * torch.norm(w_world, dim=-1, keepdim=True)
            else:
                target = w_world

            # F3 (Порядок) стабилизирует картинку, F4 (Хаос) раскачивает переход
            alpha = float(np.clip(0.35 - beta_order * 0.2 + beta_chaos * 0.2 + rx_sagitta * 0.15, 0.15, 0.75))
            with self.lock:
                if self.latent_active is None: self.latent_active = target.clone()
                else: self.latent_active = self.latent_active * (1.0 - alpha) + target * alpha

    def _loop(self):
        times = []
        while self.running:
            if not self.initialized:
                try:
                    self.conn = Client(('localhost', 6000), authkey=b'brain')
                    self.conn.send({'cmd': 'init_mode', 'mode': self.mode, 'use_taesd': self.use_taesd})
                    self.conn.recv()
                    self.conn.send({'cmd': 'encode_base_prompts', 'prompts': self.prompts})
                    enc_resp = self.conn.recv()
                    self.c_bases = [torch.tensor(b, dtype=torch.float32, device=DEVICE) for b in enc_resp['c_bases']]
                    
                    gen = torch.Generator(device=DEVICE).manual_seed(42)
                    u = torch.randn_like(self.c_bases[0], generator=gen)
                    v = torch.randn_like(self.c_bases[0], generator=gen)
                    self.dir_u = u / torch.norm(u, dim=-1, keepdim=True)
                    self.dir_v = v / torch.norm(v, dim=-1, keepdim=True)
                    
                    with self.lock:
                        if len(self.c_bases) > 0: self.latent_active = self.c_bases[0].clone()
                    self.initialized = True
                    print(f"✅ [BACKEND] Diffusion активен. Загружено базовых концептов: {len(self.c_bases)}")
                except Exception:
                    time.sleep(0.5)
                    continue

            with self.lock:
                latent = self.latent_active.clone().cpu().numpy() if self.latent_active is not None else None
                img = self.current_rgb.copy()
                s_val = self.strength

            if latent is None: time.sleep(0.04); continue

            try:
                t0 = time.time()
                req = {'cmd': 'generate', 'image_np': img, 'prompt_embeds': latent, 'strength': max(0.20, min(0.85, s_val))}
                if self.speed == "fast": req['num_inference_steps'] = 3
                self.conn.send(req)
                resp = self.conn.recv()

                with self.lock:
                    if isinstance(resp, np.ndarray):
                        if resp.shape[:2] != (self.img_h, self.img_w): resp = cv2.resize(resp, (self.img_w, self.img_h))
                        treated = apply_color_surgery(resp, img.astype(np.float32)) if self.use_color else resp
                        self.current_rgb = cv2.addWeighted(self.current_rgb, 0.25, treated, 0.75, 0)
                        self.frame_id += 1

                times.append(time.time() - t0)
                if len(times) > 5: times.pop(0)
                self.fps = 1.0 / (np.mean(times) + 1e-6)
            except Exception:
                self.initialized = False
                time.sleep(0.5)

# =====================================================================
# 5. УНИВЕРСАЛЬНЫЙ ПАРСЕР ЮЗЕРОВ (СКОЛЬКО УГОДНО РЕГИОНОВ И ЛЮДЕЙ)
# =====================================================================
def parse_user_setup(users_str: str | None, pilot_kwargs: dict, is_sim: bool = False, concepts: int = 1) -> list[BrainSubject]:
    if not users_str:
        if is_sim or concepts > 1:
            # ПОЛНЫЙ АНСАМБЛЬ: F3 (0), F4 (1), AFz (2), Fpz (3), FCz (4)
            return [BrainSubject("User1", {"F3": 0, "F4": 1, "AFz": 2, "Fpz": 3, "FCz": 4}, max_capacity=MAX_CONCEPTS_CAPACITY, **pilot_kwargs)]
        else:
            return [BrainSubject("User1", {"FCz": 0}, max_capacity=MAX_CONCEPTS_CAPACITY, **pilot_kwargs)]

    subjects = []
    for b in users_str.split(";"):
        if not b.strip(): continue
        p = b.strip().split(":")
        sub_id = p[0].strip()
        reg_map = {rd.split("=")[0].strip(): int(rd.split("=")[1].strip()) for rd in p[1].split(",") if "=" in rd}
        if not reg_map: reg_map = {"F3": 0, "F4": 1, "AFz": 2, "Fpz": 3, "FCz": 4} if (is_sim or concepts > 1) else {"FCz": 0}
        subjects.append(BrainSubject(sub_id, reg_map, max_capacity=MAX_CONCEPTS_CAPACITY, **pilot_kwargs))
        
    if not subjects:
        subjects = [BrainSubject("User1", {"F3": 0, "F4": 1, "AFz": 2, "Fpz": 3, "FCz": 4}, max_capacity=MAX_CONCEPTS_CAPACITY, **pilot_kwargs)]
    return subjects

def main():
    parser = argparse.ArgumentParser(description="NeuroCanvas: Full Prefrontal Heterarchy (F3, F4, AFz, Fpz, FCz)")
    parser.add_argument('--config', type=str, default="swarm_config.json")
    parser.add_argument('--sim', action='store_true', default=False)
    parser.add_argument('--concepts', type=int, default=8)
    parser.add_argument('--start-prompt', type=str, default="ancient medieval stone castle fortress towers, daytime, sharp focus, 8k")
    parser.add_argument('--chaos', action='store_true')
    parser.add_argument('--calib-mode', type=str, default="auto", choices=["auto", "motion", "semantic"])
    parser.add_argument('--move-directions', type=int, default=4, choices=[2, 4])
    parser.add_argument('--calib-sigma', type=float, default=3.5)
    parser.add_argument('--calib-seconds', type=float, default=6.0)
    parser.add_argument('--calib-cycles', type=int, default=3)
    parser.add_argument('--users', type=str, default=None)
    parser.add_argument('--sensory-sub', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default="lcm", choices=["lcm", "turbo", "sdxl-turbo"])
    parser.add_argument('--speed', type=str, default="fast", choices=["fast", "quality"])
    parser.add_argument('--strength-high', type=float, default=0.85)
    parser.add_argument('--strength-low', type=float, default=0.50)
    parser.add_argument('--gamma-100', action='store_true')
    parser.add_argument('--use-kinematics', action='store_true', default=True)
    
    # Кинематика действий FCz
    parser.add_argument('--sensitivity', type=float, default=0.05)
    parser.add_argument('--max-speed', type=float, default=9.0)
    parser.add_argument('--fwd-scale', type=float, default=0.2)
    parser.add_argument('--strafe-scale', type=float, default=0.2)
    parser.add_argument('--turn-scale', type=float, default=0.5)
    parser.add_argument('--intent-gain', type=float, default=1.5)
    
    parser.add_argument('--hardcoded-bots', type=int, default=4)
    parser.add_argument('--jepa-bots', type=int, default=4)
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

    if args.users is None:
        args.users = "User1:F3=0,F4=1,AFz=2,Fpz=3,FCz=4" if (args.sim or args.concepts > 1) else "User1:FCz=0"

    subjects = parse_user_setup(args.users, pilot_kwargs, is_sim=args.sim, concepts=args.concepts)
    collective = OverBrainCollective(subjects)
    primary_brain = subjects[0]

    has_fcz_global = any(s.has_region("FCz") for s in subjects)
    has_afz_global = any(s.has_region("AFz") for s in subjects)
    is_motion_calib = False if (args.sim or args.concepts > 1 or has_afz_global) else has_fcz_global

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
    
    if is_motion_calib:
        initial_prompts = [args.start_prompt]
        calib_block_names = MOTION_NAMES[:args.move_directions]
        active_memory_classes = args.move_directions
    else:
        num_c = max(2, args.concepts)
        initial_prompts = [BASE_PROMPTS[i % len(BASE_PROMPTS)] for i in range(num_c)]
        calib_block_names = [ALL_NAMES[i % len(ALL_NAMES)] for i in range(num_c)]
        active_memory_classes = len(calib_block_names)

    print(f"👥 [КОЛЛЕКТИВ]: {len(subjects)} субъект(ов) | Режим: [{'FCz ДЕЙСТВИЯ' if is_motion_calib else 'AFz СЕМАНТИКА'}] | Классов: {active_memory_classes}")

    agent = None
    if args.sim:
        agent = SyntheticAutonomousAgent(
            config_path=args.config, num_hardcoded=args.hardcoded_bots, num_jepa=args.jepa_bots, 
            num_concepts=max(2, active_memory_classes), sps=args.sps
        )
        if args.sensory_sub: agent.set_sensory_substitution(True)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("NeuroCanvas × tbp.monty: Full Prefrontal Suite (F3, F4, AFz, Fpz, FCz)")
    clock = pygame.time.Clock()
    font_b = pygame.font.SysFont("consolas", 14, bold=True)
    font_s = pygame.font.SysFont("consolas", 11)
    font_large = pygame.font.SysFont("consolas", 18, bold=True)
    font_huge = pygame.font.SysFont("consolas", 22, bold=True)

    engine = HeterarchicalBrainEngine(gamma_max=100.0 if args.gamma_100 else 65.0)
    engine.start()

    worker = EqualPoolChaosWorker(
        initial_prompts=initial_prompts, initial_names=calib_block_names,
        port=6000, mode=args.mode, speed=args.speed, use_taesd=not args.no_taesd, use_color=not args.no_color
    )
    if args.chaos: worker.chaos_enabled = True

    clip_teacher = VisualCLIPTeacher(initial_prompts)

    is_calibrating = not args.chaos
    if not args.force_recalib and os.path.exists(args.weights) and not args.chaos:
        loaded, cnt = primary_brain.heterarchy.load_from_file(args.weights)
        if loaded and cnt == active_memory_classes: is_calibrating = False

    calib_step_idx = 0
    calib_cycle_count = 0
    epoch_start_time = time.time()
    calib_data = {'y_fwd': [], 'y_bwd': [], 'x_rgt': [], 'x_lft': []} if is_motion_calib else {i: [] for i in range(active_memory_classes)}
        
    current_d_prime, dp_axis_y, dp_axis_x = 0.0, 0.0, 0.0
    concept_d_primes = [0.0] * active_memory_classes
    all_green = False
    analog_evidence = np.zeros(active_memory_classes, dtype=np.float32)

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
            dt = clock.tick(60) / 1000.0
            frame_counter += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT: raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_SPACE, pygame.K_c]: worker.toggle_chaos()
                    elif event.key == pygame.K_m: 
                        FULL_DUPLEX_MODE = not FULL_DUPLEX_MODE
                        if agent: agent.set_sensory_substitution(FULL_DUPLEX_MODE)
                        print(f"📡 [SENSORY-SUB]: {'ВКЛЮЧЕН (FULL-DUPLEX)' if FULL_DUPLEX_MODE else 'ВЫКЛЮЧЕН (ЧИСТАЯ СТИГМЕРГИЯ)'}")
                    elif event.key == pygame.K_r:
                        is_calibrating = True
                        calib_step_idx, calib_cycle_count = 0, 0
                        analog_evidence.fill(0.0)
                        epoch_start_time = time.time()
                        calib_data = {'y_fwd': [], 'y_bwd': [], 'x_rgt': [], 'x_lft': []} if is_motion_calib else {i: [] for i in range(active_memory_classes)}
                        concept_d_primes = [0.0] * active_memory_classes
                        all_green = False

            frame = engine.get_frame()
            has_live_eeg = (frame.num_live > 0)

            for s in subjects: s.process_neurophysiology(frame.nodes, dt, active_memory_classes)

            sorted_subj = sorted(subjects, key=lambda s: s.confidence, reverse=True)
            leader_brain = sorted_subj[0] if sorted_subj else primary_brain
            child_brain = sorted_subj[1] if len(sorted_subj) > 1 else leader_brain

            leader_idx = leader_brain.decoded_leader_idx % len(calib_block_names)
            child_idx = child_brain.decoded_leader_idx % len(calib_block_names)

            # ВСЕ 5 РЕГИОНОВ ИЗВЛЕКАЮТСЯ ОДНОВРЕМЕННО:
            f3_node  = leader_brain.get_region_node("F3", frame.nodes)
            f4_node  = child_brain.get_region_node("F4", frame.nodes)
            afz_node = leader_brain.get_region_node("AFz", frame.nodes)
            fpz_node = leader_brain.get_region_node("Fpz", frame.nodes)
            fcz_node = leader_brain.get_region_node("FCz", frame.nodes)

            primary_lead_node = afz_node if afz_node is not None else (f3_node if f3_node is not None else frame.nodes[0])

            # F3 управляет Порядком (Beta Order), F4 управляет Хаосом (Beta Chaos)
            beta_order = float(f3_node.beta_power) if f3_node is not None else 0.5
            beta_chaos = float(f4_node.beta_power) if f4_node is not None else 0.5
            torus_u = float(afz_node.torus_u) if afz_node is not None else float(primary_lead_node.torus_u)
            torus_v = float(afz_node.torus_v) if afz_node is not None else float(primary_lead_node.torus_v)

            # FCz управляет Действиями и Кинематикой
            if fcz_node is not None:
                axes = fcz_node.kinematics
                force_x = float(axes.lx)        # Латеральный сдвиг
                force_y = float(-axes.ly)       # Продольное действие (Зум)
                wm_curvature = float(axes.rx)   # Сагитта Yaw (Облет)
                temporal_bias = float(axes.ry)  # Фокус внимания
                is_motion_active = True
            else:
                force_x, force_y, wm_curvature = 0.0, 0.0, 0.0
                temporal_bias = float(primary_lead_node.kinematics.ry)
                is_motion_active = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:    force_y += 1.0
            if keys[pygame.K_DOWN]:  force_y -= 1.0
            if keys[pygame.K_RIGHT]: force_x += 1.0
            if keys[pygame.K_LEFT]:  force_x -= 1.0
            if keys[pygame.K_PERIOD]: wm_curvature += 0.5
            if keys[pygame.K_COMMA]:  wm_curvature -= 0.5

            leader_brain.pilot.update(dt, force_x, force_y, wm_curvature, temporal_bias)

            if abs(force_x) > 0.04 or abs(force_y) > 0.04 or abs(wm_curvature) > 0.03:
                with worker.lock:
                    worker.current_rgb = apply_manifold_camera_warp(worker.current_rgb, leader_brain.pilot, dt)

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

            safe_step_idx = calib_step_idx % active_memory_classes
            target_now = calib_block_names[safe_step_idx]

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

            # Разрешение гетерархии
            agent_pmap = agent.get_parent_map() if agent else {}
            parent_map = collective.resolve_heterarchy(
                worker.active_count, worker.parent_map, agent_pmap, 
                lead_causal_sign, leader_idx, child_idx
            )

            if is_calibrating:
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
                    plasticity_mod = (float(np.tanh(analog_projection * 2.0)) * persistence_gain) if is_motion_calib else 1.0

                    for s in subjects:
                        s.learn_contrastive(safe_step_idx, active_memory_classes, lr=0.04 * plasticity_mod, ltd_factor=0.6)

                    if is_motion_calib:
                        if safe_step_idx == 0:   calib_data['y_fwd'].append(force_y)
                        elif safe_step_idx == 1: calib_data['y_bwd'].append(force_y)
                        elif safe_step_idx == 2: calib_data['x_rgt'].append(force_x)
                        elif safe_step_idx == 3: calib_data['x_lft'].append(force_x)

                        y_f = np.array(calib_data['y_fwd'][-120:]) if len(calib_data['y_fwd']) >= 10 else np.array([0.0])
                        y_b = np.array(calib_data['y_bwd'][-120:]) if len(calib_data['y_bwd']) >= 10 else np.array([0.0])
                        dp_axis_y = max(0.0, float((np.mean(y_f) - np.mean(y_b)) / math.sqrt(0.5 * (np.var(y_f) + np.var(y_b)) + 1e-6)))

                        x_r = np.array(calib_data['x_rgt'][-120:]) if len(calib_data['x_rgt']) >= 10 else np.array([0.0])
                        x_l = np.array(calib_data['x_lft'][-120:]) if len(calib_data['x_lft']) >= 10 else np.array([0.0])
                        dp_axis_x = max(0.0, float((np.mean(x_r) - np.mean(x_l)) / math.sqrt(0.5 * (np.var(x_r) + np.var(x_l)) + 1e-6)))

                        current_d_prime = (dp_axis_y + dp_axis_x) / 2.0
                        all_green = (dp_axis_y >= args.calib_sigma) and (dp_axis_x >= args.calib_sigma)
                    else:
                        for s in subjects:
                            if s.last_sdr is not None:
                                margin_val = s.heterarchy.compute_contrastive_margin(s.last_sdr, safe_step_idx, active_memory_classes)
                                calib_data[safe_step_idx].append(margin_val)

                        concept_d_primes = []
                        for k in range(active_memory_classes):
                            vals_k = calib_data[k][-60:] if len(calib_data[k]) >= 10 else []
                            dp_k = max(0.0, float(np.mean(vals_k) / math.sqrt(np.var(vals_k) + 1e-6))) if len(vals_k) >= 10 else 0.0
                            concept_d_primes.append(dp_k)

                        current_d_prime = float(np.mean(concept_d_primes)) if concept_d_primes else 0.0
                        all_green = all(dp >= args.calib_sigma for dp in concept_d_primes)

                    elapsed = time.time() - epoch_start_time
                    if elapsed >= args.calib_seconds:
                        epoch_start_time = time.time()
                        calib_step_idx += 1
                        if calib_step_idx >= active_memory_classes:
                            calib_step_idx = 0
                            calib_cycle_count += 1
                            if all_green and calib_cycle_count >= args.calib_cycles:
                                is_calibrating = False
                                leader_brain.heterarchy.save_to_file(args.weights, calib_block_names)
                                if agent: agent.set_calibration_target(False)
                                print(f"🎯 [КАЛИБРОВКА ЗАВЕРШЕНА] d'={current_d_prime:.2f}σ!")
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
                worker.strength = float(np.clip(base_strength + temporal_bias * 0.15, 0.10, 0.99))

            # =============================================================
            # РЕНДЕРИНГ ИНТЕРФЕЙСА
            # =============================================================
            screen.fill((10, 14, 20))

            img_x, img_y = 370, 40
            surf_diff = pygame.image.frombuffer(rgb_m.tobytes(), (worker.img_w, worker.img_h), 'RGB')
            if (worker.img_w, worker.img_h) != (512, 384): surf_diff = pygame.transform.scale(surf_diff, (512, 384))
            screen.blit(surf_diff, (img_x, img_y))
            pygame.draw.rect(screen, (40, 50, 70), (img_x, img_y, 512, 384), 2, border_radius=8)

            if leader_brain and leader_brain.last_sdr_first is not None:
                cur_sdr_img = (leader_brain.last_sdr_first[-1].view(64, 64).cpu().numpy() * 255.0).astype(np.uint8)
                resized_sdr = cv2.resize(cur_sdr_img, (140, 140), interpolation=cv2.INTER_NEAREST)
                screen.blit(pygame.surfarray.make_surface(resized_sdr), (img_x, 445))
            screen.blit(font_s.render(f"L4 F3-Sheet [{leader_brain.subject_id}]", True, (0, 255, 200)), (img_x, 428))

            dbg_x, dbg_y, dbg_w, dbg_h = img_x + 160, 430, 400, 175
            pygame.draw.rect(screen, (14, 18, 26), (dbg_x, dbg_y, dbg_w, dbg_h), border_radius=6)
            rx_c, ry_c, rc = dbg_x + 295, dbg_y + 88, 68
            pygame.draw.circle(screen, (20, 28, 40), (rx_c, ry_c), rc)
            pygame.draw.circle(screen, (40, 60, 85), (rx_c, ry_c), rc, 1)

            if is_motion_active:
                if is_calibrating:
                    t_deg = TARGET_ANGLES_DEG[safe_step_idx]
                    tx_end = rx_c + int((rc - 8) * math.cos(math.radians(t_deg)))
                    ty_end = ry_c - int((rc - 8) * math.sin(math.radians(t_deg)))
                    pygame.draw.line(screen, (255, 220, 50), (rx_c, ry_c), (tx_end, ty_end), 5)

                if cur_nav_mag > 0.02:
                    norm_len = min(1.0, cur_nav_mag)
                    lx_end = rx_c + int(norm_len * (rc - 8) * (force_x / cur_nav_mag))
                    ly_end = ry_c - int(norm_len * (rc - 8) * (force_y / cur_nav_mag))
                    pygame.draw.line(screen, (0, 255, 255), (rx_c, ry_c), (lx_end, ly_end), 4)

                screen.blit(font_b.render("FCz МОТОРНЫЙ АФФОРДАНС (SMA):", True, (0, 255, 200)), (dbg_x + 10, dbg_y + 8))
            else:
                tu_x, tv_y = math.cos(torus_u), math.sin(torus_v)
                pygame.draw.line(screen, (0, 255, 255), (rx_c, ry_c), (rx_c + int(tu_x * (rc - 8)), ry_c - int(tv_y * (rc - 8))), 4)
                pygame.draw.circle(screen, (255, 255, 255), (rx_c + int(tu_x * (rc - 8)), ry_c - int(tv_y * (rc - 8))), 5)
                screen.blit(font_b.render("AFz КОГНИТИВНЫЙ ТОР (u, v):", True, (0, 255, 200)), (dbg_x + 10, dbg_y + 8))

            if is_calibrating:
                screen.blit(font_huge.render(f"МИШЕНЬ: {target_now}", True, (255, 220, 50)), (dbg_x + 10, dbg_y + 34))
                cur_ev = analog_evidence[safe_step_idx]
                screen.blit(font_b.render(f"Интеграл DDM: {cur_ev:.2f} / 3.00", True, (100, 255, 100)), (dbg_x + 10, dbg_y + 70))
            else:
                for i, name in enumerate(calib_block_names[:min(4, active_memory_classes)]):
                    score = leader_brain.wm_scores[i]
                    col_bar = (0, 255, 180) if score > 15.0 else (80, 80, 90)
                    by = dbg_y + 34 + i * 22
                    screen.blit(font_s.render(f"{name[:8]:8s}", True, (200, 200, 200)), (dbg_x + 10, by))
                    bar_len = int((score / 100.0) * 65)
                    if bar_len > 0: pygame.draw.rect(screen, col_bar, (dbg_x + 65, by + 2, bar_len, 10))

            chaos_txt = "CHAOS BIFURCATIONS: [ACTIVE]" if worker.chaos_enabled else "CHAOS BIFURCATIONS: [OFF] (SPACE/C)"
            screen.blit(font_s.render(chaos_txt, True, (255, 100, 255) if worker.chaos_enabled else (120, 120, 140)), (dbg_x + 10, dbg_y + 152))

            # Левая верхняя панель (Калибровка)
            panel_x, panel_y = 20, 40
            pygame.draw.rect(screen, (16, 22, 32), (panel_x, panel_y, 330, 270), border_radius=8)
            border_col = (255, 120, 40) if is_calibrating else (0, 255, 200)
            pygame.draw.rect(screen, border_col, (panel_x, panel_y, 330, 270), 2 if is_calibrating else 1, border_radius=8)

            hdr_text = "5-SIGMA FCz (ДЕЙСТВИЯ)" if is_motion_calib else f"5-SIGMA AFz ({active_memory_classes} КОНЦЕПТОВ)"
            screen.blit(font_large.render(hdr_text, True, (255, 180, 50)), (panel_x + 15, panel_y + 10))

            if is_calibrating:
                rem_time = max(0.0, args.calib_seconds - (time.time() - epoch_start_time))
                screen.blit(font_b.render(f"МИШЕНЬ: [{target_now}] ({rem_time:.1f}s)", True, (255, 255, 100)), (panel_x + 15, panel_y + 32))
                status_color = (100, 255, 100) if all_green else (200, 200, 200)
                status_txt = f"Цикл {calib_cycle_count}/{args.calib_cycles} | Ждем ВСЕ ≥ {args.calib_sigma:.1f}σ"
                screen.blit(font_s.render(status_txt, True, status_color), (panel_x + 15, panel_y + 50))
            else:
                screen.blit(font_b.render("LTM ОБУЧЕН (ВСЕ ЗЕЛЁНЫЕ)", True, (0, 255, 200)), (panel_x + 15, panel_y + 32))
                screen.blit(font_s.render(f"Средняя точность: d'={current_d_prime:.2f}σ", True, (100, 255, 100)), (panel_x + 15, panel_y + 50))

            row_h = min(25, int(190 / max(1, active_memory_classes)))
            for k in range(active_memory_classes):
                cy = panel_y + 72 + k * row_h
                dp_val = concept_d_primes[k] if k < len(concept_d_primes) else 0.0
                ltm_val = leader_brain.ltm_scores[k] if k < len(leader_brain.ltm_scores) else 0.0
                name = calib_block_names[k]
                
                is_act = is_calibrating and (k == safe_step_idx)
                is_k_green = dp_val >= args.calib_sigma

                c_col = (100, 255, 100) if is_k_green else ((255, 255, 100) if is_act else (180, 180, 190))
                tag = "▶" if is_act else " "
                screen.blit(font_s.render(f"{tag}{name[:6]:6s} {dp_val:3.1f}σ ({ltm_val:2.0f}%)", True, c_col), (panel_x + 12, cy))

                bar_w, bar_h = 120, max(6, row_h - 10)
                bx, by = panel_x + 195, cy + (row_h - bar_h) // 2 - 2
                fill_w = int(np.clip(dp_val / args.calib_sigma, 0.0, 1.0) * bar_w)
                pygame.draw.rect(screen, (30, 40, 50), (bx, by, bar_w, bar_h), border_radius=3)
                if fill_w > 0: pygame.draw.rect(screen, c_col, (bx, by, fill_w, bar_h), border_radius=3)

            # ПАНЕЛЬ ТЕЛЕМЕТРИИ (ВСЕ 5 РЕГИОНОВ ВЫВОДЯТСЯ В РЕАЛТАЙМЕ)
            c_x, c_y = 20, 330
            pygame.draw.rect(screen, (16, 22, 32), (c_x, c_y, 330, 270), border_radius=8)
            pygame.draw.rect(screen, (100, 180, 255), (c_x, c_y, 330, 270), 1, border_radius=8)
            screen.blit(font_b.render(f"5-ПРЕФРОНТАЛЬНЫЙ АНСАМБЛЬ ({len(subjects)} СУБ.):", True, (100, 180, 255)), (c_x + 12, c_y + 12))

            lead_name = calib_block_names[leader_idx % len(calib_block_names)]
            screen.blit(font_b.render(f"Лидер: [{lead_name}] ({leader_brain.subject_id})", True, (100, 255, 100)), (c_x + 15, c_y + 34))
            
            # F3 (Левая PFC) - Порядок
            screen.blit(font_s.render(f"• F3 Порядок (β)  : {beta_order:.2f} (L-dlPFC Гейт)", True, (100, 255, 120)), (c_x + 15, c_y + 56))
            # F4 (Правая PFC) - Хаос
            screen.blit(font_s.render(f"• F4 Хаос (β)     : {beta_chaos:.2f} (R-dlPFC Драйв)", True, (255, 100, 255)), (c_x + 15, c_y + 78))
            # AFz (Медиальная PFC) - Тор
            screen.blit(font_s.render(f"• AFz Тор (u, v)  : ({torus_u:.2f}, {torus_v:.2f})", True, (255, 220, 50)), (c_x + 15, c_y + 100))
            # Fpz (Фронтополярная) - Ветвление
            fpz_txt = f"АКТИВЕН ({fpz_node.gating_ratio:.2f})" if fpz_node else "НЕТ"
            screen.blit(font_s.render(f"• Fpz Ветвление   : {fpz_txt}", True, (200, 220, 255)), (c_x + 15, c_y + 122))
            # FCz (Моторная) - Действия
            nav_status = f"VX={leader_brain.pilot.vx:+.1f} | VY={leader_brain.pilot.vy:+.1f}" if is_motion_active else "СТАТИЧНЫЙ"
            screen.blit(font_s.render(f"• FCz Действие    : {nav_status} (SMA/pre-SMA)", True, (0, 255, 200)), (c_x + 15, c_y + 144))
            
            screen.blit(font_s.render(f"• Theta={frame.theta_freq:.2f}Hz | Delta={frame.delta_freq:.2f}Hz", True, (200, 220, 240)), (c_x + 15, c_y + 190))
            screen.blit(font_s.render(f"• Потоков LSL: {frame.num_live} (FreeEEG16)", True, (180, 180, 200)), (c_x + 15, c_y + 215))

            # 120-EDGE ciPLV
            BY, BH = 620, 300
            pygame.draw.rect(screen, (12, 16, 24), (20, BY, 870, BH), border_radius=8)
            screen.blit(font_b.render("120-EDGE DIRECTED ciPLV (65–100 Гц РИППЛЫ ДИКИ)", True, (0, 255, 200)), (35, BY + 15))
            g120 = primary_lead_node.iplv_human_ripple[31] if has_live_eeg else np.zeros(120)
            bw, mid_line = 800.0 / 120.0, BY + 150
            for p in range(120):
                val = g120[p]
                bh = int(abs(val) * 110)
                col = (255, 80, 80) if val < 0 else (80, 255, 180)
                pygame.draw.rect(screen, col, (35 + p * bw, mid_line - (bh if val >= 0 else 0), max(1, int(bw - 1)), bh))

            # Causal Manifold
            rx_p, ry_p = 920, 620
            pygame.draw.rect(screen, (20, 15, 30), (rx_p, ry_p, 840, 300), border_radius=8)
            dir_str = "A ⊃ B (Надстройка)" if lead_causal_sign >= 0.03 else ("B ⊃ A (Встраивание)" if lead_causal_sign <= -0.03 else "A ∥ B (Соседство)")
            screen.blit(font_s.render(f"Causal Lead (90 Hz ciPLV): {lead_causal_sign:+.3f} -> {dir_str}", True, (200, 220, 220)), (rx_p + 20, ry_p + 20))
            for i, sv in enumerate(svd_spectrum):
                bh = int(sv * 35)
                pygame.draw.rect(screen, (0, 255, 180) if sv > 0.22 else (80, 80, 80), (rx_p + 20 + i*65, ry_p + 50 + (35 - bh), 45, bh))

            # Рекурсивный Treemap
            map_x, map_y = 950, 40
            map_w, map_h = 800, 560
            pygame.draw.rect(screen, (10, 10, 10), (map_x, map_y, map_w, map_h))
            pygame.draw.rect(screen, (255, 255, 255), (map_x-2, map_y-2, map_w+4, map_h+4), 2)

            boxes, resolved_hierarchy = calculate_emergent_treemap(
                map_x, map_y, map_w, map_h, leader_brain.blended_weights, active_memory_classes, parent_map=parent_map
            )
            treemap_title = f"РЕКУРСИВНАЯ ГЕТЕРАРХИЯ СМЫСЛОВ ({active_memory_classes} КОНЦЕПТОВ)"
            screen.blit(font_b.render(treemap_title, True, (255, 255, 255)), (map_x, map_y - 25))

            for c_idx in resolved_hierarchy:
                if c_idx >= active_memory_classes: continue
                bx, by, bw, bh, rank, depth = boxes[c_idx]
                base_col = FRACTAL_COLORS[c_idx % len(FRACTAL_COLORS)]
                darken = max(0.25, 1.0 - depth * 0.22)
                c_color = (int(base_col[0]*darken), int(base_col[1]*darken), int(base_col[2]*darken))

                pygame.draw.rect(screen, c_color, (bx, by, bw, bh))
                pygame.draw.rect(screen, (220, 220, 220), (bx, by, bw, bh), max(1, 3 - depth))
                c_name = calib_block_names[c_idx % len(calib_block_names)]
                owner_tag = f"[{leader_brain.subject_id}] " if leader_brain.decoded_leader_idx == c_idx and leader_brain.confidence > 20.0 else ""
                
                parent_idx = parent_map.get(c_idx)
                rel_sym = ""
                if parent_idx is not None and parent_idx < len(calib_block_names):
                    rel_sym = f" ➔ {calib_block_names[parent_idx][:4]}"
                
                screen.blit(font_s.render(f"D{worker.depths[c_idx]} (H:{depth}){rel_sym}", True, (255, 255, 255)), (bx + 5, by + 5))
                if bw >= 50 and bh >= 35:
                    screen.blit(font_b.render(f"{owner_tag}{c_name}", True, (255, 255, 255)), (bx + 5, by + 20))

            mode_color = (0, 255, 100) if FULL_DUPLEX_MODE else (160, 160, 160)
            mode_str = "SENSORY SUB: [FULL-DUPLEX (ПРЯМОЙ ОБМЕН)]" if FULL_DUPLEX_MODE else "SENSORY SUB: [ЧИСТАЯ СТИГМЕРГИЯ (ЧЕРЕЗ ХОЛСТ)]"
            screen.blit(font_b.render(mode_str, True, mode_color), (1350, 15))

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
