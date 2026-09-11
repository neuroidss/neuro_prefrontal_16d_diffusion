#!/usr/bin/env python3
"""
🧠 NEUROCANVAS × TBP.MONTY: FULL 3-LEVEL HIERARCHY ENGINE (v320.0)
- 16 384 Macrocolumns (4 Nodes x 64x64 L4 Sheets) running CUDA-accelerated.
- Universal support for LCM, SD-Turbo, and SDXL-Turbo via dynamic server negotiation.
- Complete dimension matching: 768 (SD 1.5), 1024 (SD 2.1), 2048+1280 (SDXL).
- Cortical Messaging Protocol (CMP) passing displacement, SO(3) pose, and scale.
- Active Inference closed loop with Visual CLIP validation and Anti-Trap Denoising.
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
import threading
import numpy as np
import cv2
import pygame
from PIL import Image
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from multiprocessing.connection import Client

# Official CMP Message Interface from Thousand Brains Project
try:
    from tbp.monty.cmp import Message
except ImportError:
    class Message:
        def __init__(self, location, morphological_features, non_morphological_features,
                     confidence, pass_message, sender_id, sender_type, process_features_in_lm):
            self.location = location
            self.morphological_features = morphological_features
            self.non_morphological_features = non_morphological_features
            self.confidence = float(confidence)
            self.pass_message = bool(pass_message)
            self.sender_id = str(sender_id)
            self.sender_type = str(sender_type)
            self.process_features_in_lm = bool(process_features_in_lm)
            self.displacement = {}

        def set_displacement(self, displacement, ppf=None):
            self.displacement = {"displacement": displacement}

from neuro_heterarchy_core import HeterarchicalBrainEngine, DEVICE

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
# 1. LAYER 4 HTM MACROCOLUMN (64x64 CUDA SHEET PER NODE)
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
# 2. STREAMING 16,384-COLUMN EXECUTIVE HETERARCHY
# ==============================================================================
class FrontalExecutiveHeterarchy(nn.Module):
    def __init__(self, num_concepts: int = 8, num_columns_per_node: int = 4096, k_active_per_node: int = 80):
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
# 3. SUPERVISORY CLIP VISION TEACHER
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

# ==============================================================================
# 4. UNIVERSAL TOROIDAL DIFFUSION WORKER (LCM / SD-TURBO / SDXL-TURBO)
# ==============================================================================
class ToroidalDiffusionWorker:
    def __init__(self, prompts, port: int = 6000, mode: str = "sdxl-turbo"):
        self.conn = None
        self.current_rgb = np.zeros((384, 512, 3), dtype=np.uint8)
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
        self.c_pooled_bases = None
        self.is_sdxl = False
        self.latent_active = None
        self.pooled_active = None

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def update_simplex_targets(self, weights_nd: np.ndarray, force_strength: float = 0.50):
        if not self.initialized or self.c_bases is None:
            return

        with torch.inference_mode():
            target = torch.zeros_like(self.c_bases[0])
            for i in range(self.num_concepts):
                target += float(weights_nd[i]) * self.c_bases[i]

            with self.lock:
                self.latent_active = self.latent_active * 0.65 + target * 0.35
                if self.is_sdxl and self.c_pooled_bases is not None:
                    target_pooled = torch.zeros_like(self.c_pooled_bases[0])
                    for i in range(self.num_concepts):
                        target_pooled += float(weights_nd[i]) * self.c_pooled_bases[i]
                    self.pooled_active = self.pooled_active * 0.65 + target_pooled * 0.35
                self.strength = force_strength

    def _loop(self):
        times = []
        while self.running:
            if not self.initialized:
                try:
                    self.conn = Client(('localhost', 6000), authkey=b'brain')
                    
                    # 1. Согласовываем режим работы пайплайна
                    self.conn.send({'cmd': 'init_mode', 'mode': self.mode})
                    init_ack = self.conn.recv()
                    self.is_sdxl = init_ack.get('is_sdxl', False)
                    actual_mode = init_ack.get('mode', self.mode)
                    
                    # 2. Получаем точно сгенерированные сервером эмбеддинги
                    self.conn.send({'cmd': 'encode_base_prompts', 'prompts': self.prompts})
                    enc_resp = self.conn.recv()
                    
                    self.c_bases = [torch.tensor(b, dtype=torch.float32, device=DEVICE) for b in enc_resp['c_bases']]
                    self.latent_active = self.c_bases[0].clone()
                    
                    if self.is_sdxl and enc_resp.get('pooled_bases') is not None:
                        self.c_pooled_bases = [torch.tensor(p, dtype=torch.float32, device=DEVICE) for p in enc_resp['pooled_bases']]
                        self.pooled_active = self.c_pooled_bases[0].clone()
                    else:
                        self.c_pooled_bases = None
                        self.pooled_active = None

                    # 3. Стартовый прогревочный кадр
                    dummy = np.random.randint(100, 150, (384, 512, 3), dtype=np.uint8)
                    init_payload = {
                        'cmd': 'generate',
                        'image_np': dummy,
                        'prompt_embeds': self.latent_active.cpu().numpy(),
                        'strength': 1.0
                    }
                    if self.pooled_active is not None:
                        init_payload['pooled_prompt_embeds'] = self.pooled_active.cpu().numpy()

                    self.conn.send(init_payload)
                    self.current_rgb = apply_surgery(self.conn.recv(), dummy.astype(np.float32))
                    self.initialized = True
                    print(f"✅ [SD-{actual_mode.upper()}] Connected! Base embeds verified (Dim={self.c_bases[0].shape[-1]}, SDXL={self.is_sdxl})")
                except Exception as e:
                    time.sleep(0.5)
                    continue

            try:
                t0 = time.time()
                with self.lock:
                    emb = self.latent_active.cpu().numpy()
                    pooled = self.pooled_active.cpu().numpy() if self.pooled_active is not None else None
                    img = self.current_rgb.copy()
                    s_val = self.strength

                payload = {'cmd': 'generate', 'image_np': img, 'prompt_embeds': emb, 'strength': s_val}
                if pooled is not None:
                    payload['pooled_prompt_embeds'] = pooled

                self.conn.send(payload)
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
# 5. MAIN CLOSED-LOOP CONTROLLER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="NeuroCanvas × tbp.monty: Invasive Laminar Heterarchy")
    parser.add_argument('--sim', action='store_true', help="Enable autonomous invasive agent process")
    parser.add_argument('--concepts', type=int, default=8, choices=[4, 8], help="Number of active concepts in hierarchy")
    parser.add_argument('--online-learn', action='store_true', default=False, help="Enable adaptive ground-truth updates during inference")
    parser.add_argument('--online-lr', type=float, default=0.02, help="Online plasticity rate")
    parser.add_argument('--mode', type=str, default="lcm", choices=["lcm", "turbo", "sdxl-turbo"],
                        help="Active diffusion pipeline mode (Default: sdxl-turbo)")
    parser.add_argument('--turbo', action='store_true', help="Alias for --mode turbo")
    parser.add_argument('--sdxl', action='store_true', help="Alias for --mode sdxl-turbo")
    args = parser.parse_args()

    # Разрешаем псевдонимы аргументов
    active_mode = args.mode
    if args.sdxl:
        active_mode = "sdxl-turbo"
    elif args.turbo:
        active_mode = "turbo"

    ALL_NAMES = [
        "КОСМОС",
        "ПЛАНЕТА",
        "КИБЕРПАНК",
        "НЕБОСКРЕБ",
        "ГОРА",
        "ЗАМОК",
        "ОКЕАН",
        "ДЖУНГЛИ"
    ]

    ALL_PROMPTS = [
        "deep outer space, glowing colorful nebula, bright stars, galaxy, 8k, sharp detailed",
        "spherical alien planet with atmosphere, continents and oceans in space, 8k, sharp detailed",
        "futuristic cyberpunk city street, neon lights, rain, glowing signs, sharp linework, 8k",
        "modern glass skyscraper buildings, downtown city, geometric architecture, sharp focus, 8k",
        "giant snowy mountain peak, rocky cliffs, clear blue sky, sharp focus, 8k",
        "ancient medieval stone castle fortress towers, daytime, sharp focus, 8k",
        "open stormy dark blue ocean, pure water surface, giant ocean waves, sea foam, no land, 8k",
        "dense lush green tropical jungle, giant trees, vines, sunlight piercing through leaves, 8k"
    ]

    NUM_CONCEPTS = args.concepts
    TARGET_NAMES = ALL_NAMES[:NUM_CONCEPTS]
    PROMPTS = ALL_PROMPTS[:NUM_CONCEPTS]

    agent = None
    if args.sim:
        from synthetic_16d_causal_agent import SyntheticAutonomousAgent
        agent = SyntheticAutonomousAgent(num_concepts=NUM_CONCEPTS)
        print(f"🤖 [BOOT] Synthetic Invasive Agent active ({NUM_CONCEPTS} concepts, 3-level tree)...")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"NeuroCanvas × tbp.monty: Invasive Laminar Heterarchy [{active_mode.upper()}]")
    clock = pygame.time.Clock()
    font_b = pygame.font.SysFont("consolas", 13, bold=True)
    font_s = pygame.font.SysFont("consolas", 11)

    engine = HeterarchicalBrainEngine()
    engine.start()

    worker = ToroidalDiffusionWorker(PROMPTS, port=6000, mode=active_mode)
    clip_teacher = VisualCLIPTeacher(TARGET_NAMES, PROMPTS)
    heterarchy = FrontalExecutiveHeterarchy(num_concepts=NUM_CONCEPTS, num_columns_per_node=4096, k_active_per_node=80).to(DEVICE)

    cx, cy = WIDTH // 2 + 140, HEIGHT // 2
    current_weights = np.ones(NUM_CONCEPTS, dtype=np.float32) / NUM_CONCEPTS

    last_frame_id = -1
    cur_probs = np.ones(NUM_CONCEPTS, dtype=np.float32) / NUM_CONCEPTS

    is_calibrating = True
    learn_idx = 0
    clean_steps_accumulated = 0
    CLEAN_STEPS_REQUIRED = 15
    CLIP_HONEST_THRESHOLD = 0.65

    concept_snapshots = {}
    concept_scores = [0.0] * NUM_CONCEPTS
    curriculum_epoch = 1

    monty_location = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    print(f"🧠 [SYSTEM] Pipeline Active [{active_mode.upper()}]. Honest Calibration Gate running...")

    try:
        while True:
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            frame = engine.get_frame()
            node_tensors = {
                "F3":  torch.tensor(frame.nodes[0].iplv_32, dtype=torch.float32, device=DEVICE),
                "F4":  torch.tensor(frame.nodes[1].iplv_32, dtype=torch.float32, device=DEVICE),
                "AFz": torch.tensor(frame.nodes[2].iplv_32, dtype=torch.float32, device=DEVICE),
                "Fpz": torch.tensor(frame.nodes[3].iplv_32, dtype=torch.float32, device=DEVICE)
            }

            with worker.lock:
                rgb_m, fid = worker.current_rgb.copy(), worker.frame_id

            if fid != last_frame_id:
                last_frame_id = fid
                cur_probs = clip_teacher.classify(rgb_m)

            if agent:
                agent.update_visual_state(cur_probs)

            with torch.no_grad():
                full_sdr, sdr_f3 = heterarchy.get_current_sdr(node_tensors)

            lead_node = frame.nodes[0]
            monty_location += lead_node.disp_xyz
            monty_location = np.clip(monty_location, -2.0, 2.0)

            cmp_message = Message(
                location=monty_location.copy(),
                morphological_features={
                    "pose_vectors": lead_node.pose_matrix,
                    "pose_fully_defined": True,
                    "on_object": True
                },
                non_morphological_features={
                    "theta_hz": frame.theta_freq,
                    "delta_hz": frame.delta_freq
                },
                confidence=max(0.0, min(1.0, (lead_node.beta_stability + 1.0) / 2.0)),
                pass_message=True,
                sender_id="F3_Macrocolumn",
                sender_type="SM",
                process_features_in_lm=True
            )
            cmp_message.set_displacement(lead_node.disp_xyz)

            # ==================================================================
            # HONEST HARD-GATE CALIBRATION
            # ==================================================================
            if is_calibrating:
                vis_conf = float(cur_probs[learn_idx])

                target_sim = np.zeros(NUM_CONCEPTS, dtype=np.float32)
                target_sim[learn_idx] = 1.0
                worker.update_simplex_targets(target_sim, force_strength=1.0)
                if agent:
                    agent.set_calibration_target(True, learn_idx)

                if vis_conf >= CLIP_HONEST_THRESHOLD:
                    clip_status = f"CLIP VERIFIED ({vis_conf*100:.1f}% >= 65%) -> IMPRINTING"
                    heterarchy.stream_learn_accumulate(learn_idx, full_sdr)
                    clean_steps_accumulated += 1
                    concept_snapshots[learn_idx] = full_sdr.clone()
                else:
                    clip_status = f"WAITING FOR CLIP CONVERGENCE... ({vis_conf*100:.1f}% < 65%)"

                if clean_steps_accumulated >= CLEAN_STEPS_REQUIRED:
                    heterarchy.finalize_slot(learn_idx)

                    with torch.no_grad():
                        for chk_i in range(NUM_CONCEPTS):
                            if chk_i in concept_snapshots and heterarchy.concept_trained[chk_i] > 0:
                                p_w, _, _ = heterarchy.predict_evidence(concept_snapshots[chk_i])
                                concept_scores[chk_i] = float(p_w[chk_i]) * 100.0

                    learn_idx += 1
                    clean_steps_accumulated = 0

                    if learn_idx >= NUM_CONCEPTS:
                        min_score = min(concept_scores)
                        if min_score >= 85.0 and all(heterarchy.concept_trained > 0):
                            is_calibrating = False
                            if agent:
                                agent.set_calibration_target(False)
                            print("🏆 [HARD GATE PASSED] All hierarchical slots locked in LTM >= 85%!")
                        else:
                            curriculum_epoch += 1
                            learn_idx = int(np.argmin(concept_scores))
                            heterarchy.reset_accumulator(learn_idx)
                            print(f"⚠️ [RE-LEARNING] Retraining weakest node: [{TARGET_NAMES[learn_idx]}] ({concept_scores[learn_idx]:.1f}% < 85%)")

                current_weights = target_sim
                switched = False

            # ==================================================================
            # ACTIVE INFERENCE SURFING & LEVEL TRANSITION
            # ==================================================================
            else:
                with torch.no_grad():
                    new_w, switched, raw_sims = heterarchy.predict_evidence(full_sdr)

                top_vis_idx = int(np.argmax(cur_probs))
                vis_weight = float(cur_probs[top_vis_idx])
                fru_val = agent.shm['frustration'].value if agent else 0.0

                if args.online_learn:
                    pred_winner = int(np.argmax(new_w))
                    if (vis_weight >= 0.75) and (pred_winner == top_vis_idx) and (fru_val <= 0.25):
                        heterarchy.online_grounding_update(top_vis_idx, full_sdr, lr=args.online_lr)

                current_weights = current_weights * 0.75 + new_w * 0.25
                current_weights = current_weights / np.sum(current_weights)

                current_target_idx = int(np.argmax(current_weights))
                clip_error = 1.0 - float(cur_probs[current_target_idx])

                if switched or fru_val > 0.40 or clip_error > 0.70:
                    force_str = 0.92
                else:
                    force_str = float(np.clip(0.48 + (clip_error * 0.28) + (1.0 - lead_node.beta_stability) * 0.12, 0.48, 0.78))

                worker.update_simplex_targets(current_weights, force_strength=force_str)

            # ==================================================================
            # PYGAME RENDERING
            # ==================================================================
            screen.fill((10, 14, 20))

            # Center Canvas
            screen.blit(pygame.image.frombuffer(rgb_m.tobytes(), (512, 384), 'RGB'), (cx - 256, cy - 192))
            pygame.draw.rect(screen, (40, 50, 70), (cx - 256, cy - 192, 512, 384), 2, border_radius=8)

            # L4 F3 Column Sheet (64x64)
            cur_sdr_img = sdr_f3[:4096].view(64, 64).cpu().numpy() * 255.0
            sdr_surf = pygame.surfarray.make_surface(cv2.resize(cur_sdr_img, (140, 140)).astype(np.uint8))
            screen.blit(sdr_surf, (cx + 256 + 20, cy - 192))
            screen.blit(font_s.render("L4 F3 Sheet (64x64)", True, (0, 255, 200)), (cx + 256 + 20, cy - 212))

            # Left Panel 1: Learning & Benchmark
            panel_x, panel_y = 30, 40
            pygame.draw.rect(screen, (16, 22, 32), (panel_x, panel_y, 380, 270), border_radius=8)
            pygame.draw.rect(screen, (0, 255, 200), (panel_x, panel_y, 380, 270), 1, border_radius=8)

            if is_calibrating:
                screen.blit(font_b.render(f"HONEST CALIBRATION [EPOCH {curriculum_epoch}]", True, (255, 180, 50)), (panel_x + 12, panel_y + 12))
                screen.blit(font_s.render(f"Target Concept    : [{TARGET_NAMES[learn_idx]}] ({learn_idx+1}/{NUM_CONCEPTS})", True, (255, 255, 100)), (panel_x + 15, panel_y + 36))

                col_status = (0, 255, 180) if "VERIFIED" in clip_status else (255, 200, 50)
                screen.blit(font_s.render(clip_status, True, col_status), (panel_x + 15, panel_y + 56))

                prog_val = min(1.0, clean_steps_accumulated / float(CLEAN_STEPS_REQUIRED))
                screen.blit(font_s.render(f"Clean Steps (>=65%): {clean_steps_accumulated}/{CLEAN_STEPS_REQUIRED}", True, (200, 220, 255)), (panel_x + 15, panel_y + 78))
                pygame.draw.rect(screen, (30, 40, 50), (panel_x + 15, panel_y + 96, 350, 8), border_radius=2)
                pygame.draw.rect(screen, (0, 255, 180), (panel_x + 15, panel_y + 96, int(prog_val * 350), 8), border_radius=2)

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
                    screen.blit(font_s.render(txt_s, True, col_s), (panel_x + 15, panel_y + 118 + i * 18))

            else:
                screen.blit(font_b.render("FROZEN RETENTION BENCHMARK (PASSED)", True, (0, 255, 200)), (panel_x + 12, panel_y + 12))
                for i, name in enumerate(TARGET_NAMES):
                    screen.blit(font_s.render(f"{name:10s}: {concept_scores[i]:4.1f}% [LOCKED IN LTM]", True, (100, 255, 100)), (panel_x + 15, panel_y + 38 + i * 22))

            # Left Panel 2: Telemetry & Active Inference
            c_x, c_y = 30, 330
            pygame.draw.rect(screen, (16, 22, 32), (c_x, c_y, 380, 270), border_radius=8)
            pygame.draw.rect(screen, (100, 180, 255), (c_x, c_y, 380, 270), 1, border_radius=8)
            screen.blit(font_b.render("ACTIVE INFERENCE & CMP HIERARCHY", True, (100, 180, 255)), (c_x + 12, c_y + 12))

            top_idx = int(np.argmax(current_weights))
            top_conf = current_weights[top_idx] * 100.0
            col_intent = (100, 255, 100) if top_conf >= 70.0 else (255, 220, 50)
            screen.blit(font_b.render(f"Decoded Intent : [{TARGET_NAMES[top_idx]}] ({top_conf:4.1f}%)", True, col_intent), (c_x + 15, c_y + 36))

            top_vis = int(np.argmax(cur_probs))
            screen.blit(font_s.render(f"Visual Reality : {TARGET_NAMES[top_vis]} ({cur_probs[top_vis]*100:.1f}% CLIP)", True, (200, 220, 255)), (c_x + 15, c_y + 58))

            if agent:
                mode, desc, mood, t_idx, sat, bor, fru = agent.get_telemetry()
                quest_col = (0, 255, 200) if t_idx == top_idx else (255, 100, 100)
                screen.blit(font_b.render(f"Hierarchy State: [{TARGET_NAMES[t_idx]}]", True, quest_col), (c_x + 15, c_y + 88))
                screen.blit(font_s.render(f"Transition Mode: {desc}", True, quest_col), (c_x + 15, c_y + 110))

                screen.blit(font_s.render(f"Convergence    : {sat*100:4.1f}%", True, (100, 255, 100)), (c_x + 15, c_y + 135))
                pygame.draw.rect(screen, (30, 40, 50), (c_x + 15, c_y + 150, 350, 6), border_radius=2)
                pygame.draw.rect(screen, (0, 255, 180), (c_x + 15, c_y + 150, int(sat * 350), 6), border_radius=2)

                screen.blit(font_s.render(f"dACC Error (AFz): {fru*100:4.1f}% | Satiation: {bor*100:4.1f}%", True, (255, 80, 80)), (c_x + 15, c_y + 168))
                screen.blit(font_s.render(f"Beta Stability : {lead_node.beta_stability:+.2f} | Conf: {cmp_message.confidence*100:.0f}%", True, (255, 200, 50)), (c_x + 15, c_y + 190))
                screen.blit(font_s.render(f"Clock dPhi/dt  : Theta={frame.theta_freq:.2f}Hz | Delta={frame.delta_freq:.2f}Hz", True, (200, 220, 240)), (c_x + 15, c_y + 212))
                screen.blit(font_s.render(f"Denoise Power  : {worker.strength:.2f} | FPS: {worker.fps:.1f} ({worker.mode.upper()})", True, (160, 180, 200)), (c_x + 15, c_y + 235))

            # Bottom Spectrum: 120-Edge Signed iPLV
            BY, BH = 620, 300
            pygame.draw.rect(screen, (12, 16, 24), (30, BY, WIDTH - 60, BH), border_radius=8)
            pygame.draw.rect(screen, (30, 45, 65), (30, BY, WIDTH - 60, BH), 1, border_radius=8)
            screen.blit(font_b.render("120-EDGE DIRECTED iPLV SPECTRUM [STRICT SIGNED sin(Δφ) ∈ [-1.0, +1.0]]", True, (0, 255, 200)), (45, BY + 15))

            g120 = lead_node.iplv_32[31]
            bw = (WIDTH - 120) / 120.0
            mid_line = BY + 150
            for p in range(120):
                val = g120[p]
                bh = int(abs(val) * 110)
                bx = 45 + p * bw
                col = (255, 80, 80) if val < 0 else (80, 255, 180)
                if val >= 0:
                    pygame.draw.rect(screen, col, (bx, mid_line - bh, bw - 1, bh))
                else:
                    pygame.draw.rect(screen, col, (bx, mid_line, bw - 1, bh))

            pygame.draw.line(screen, (60, 80, 100), (45, mid_line), (WIDTH - 75, mid_line), 1)

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
