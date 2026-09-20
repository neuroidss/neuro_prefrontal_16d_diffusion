#!/usr/bin/env python3
"""
🤖 SYNTHETIC ACTIVE INFERENCE SWARM v2200.4 (TOPOLOGICAL + CLI COMPATIBLE)
- ПОЛНОЕ ВОССТАНОВЛЕНИЕ: Включены FullJepaVideoAgent, парсинг JSON, CorticalMontage.
- СТРОГИЙ ЗАПРЕТ МЕТАГЕЙМИНГА: Расчеты универсальны для N-агентов, никаких проверок на кол-во.
- Поддержка Sensory Substitution (отключена по умолчанию).
- N-уровневая интеграция через минимизацию Свободной Энергии (Active Inference).
"""

import time
import math
import json
import os
import traceback
import threading
import multiprocessing as mp
import numpy as np
from pylsl import StreamInfo, StreamOutlet

FS = 500.0
CHUNK_SIZE = 20
NUM_CHANNELS = 16
NUM_DEVICES = 4
TWO_PI = 2.0 * math.pi
VIDEO_BUFFER_LEN = 16

COORDS_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], dtype=np.float32)
COORDS_Y = np.array([-2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], dtype=np.float32)
COORDS_Z = np.sqrt(np.maximum(100.0 - COORDS_X**2 - COORDS_Y**2, 0.0)).astype(np.float32)

ALL_NAMES = ["ГОРА",  "ДЖУНГЛИ", "ЗАМОК", "ОКЕАН", "КОСМОС", "ПЛАНЕТА", "КИБЕРПАНК", "НЕБОСКРЕБ"]

def torus_geodesic_distance(u1, v1, u2, v2):
    du = abs(u1 - u2) % TWO_PI
    if du > math.pi: du = TWO_PI - du
    dv = abs(v1 - v2) % TWO_PI
    if dv > math.pi: dv = TWO_PI - dv
    return math.sqrt(du**2 + dv**2)

class CorticalMontage:
    def __init__(self, config_dict):
        self.devices = config_dict.get('montage', {}).get('devices', [])

    def get_nodes_by_names(self, names: list) -> list[int]:
        matched_ids = []
        for dev in self.devices:
            if any(dev['name'].startswith(n) for n in names):
                matched_ids.append(dev['id'])
        return list(set(matched_ids))

    def get_nodes_by_radius(self, anchor_xyz: list, radius: float) -> list[int]:
        matched_ids = []
        anchor = np.array(anchor_xyz)
        for dev in self.devices:
            dist = np.linalg.norm(np.array(dev['coords']) - anchor)
            if dist <= radius:
                matched_ids.append(dev['id'])
        return list(set(matched_ids))
        
    def check_feature(self, active_nodes: list, feature_name: str) -> bool:
        for dev in self.devices:
            if dev['id'] in active_nodes and feature_name in dev['name']:
                return True
        return False

class BaseActiveAgent:
    def __init__(self, bot_id: int, initial_idx: int, active_nodes: list, montage: CorticalMontage, num_concepts: int = 8):
        self.bot_id = bot_id
        self.current_idx = initial_idx
        self.original_idx = initial_idx 
        self.target_name = ALL_NAMES[initial_idx]
        self.num_concepts = num_concepts
        self.active_nodes = active_nodes
        
        self.has_afz = montage.check_feature(active_nodes, "AFz")
        self.has_fpz = montage.check_feature(active_nodes, "Fpz")
        
        all_other = [i for i in range(num_concepts) if i != initial_idx]
        np.random.seed(bot_id * 101)
        self.plan_b_queue = list(np.random.permutation(all_other))
        
        self.role = "LEADER"
        self.parent_id = None 
        self.sprt_log_evidence = 0.0
        self.u_torus = (initial_idx * TWO_PI / num_concepts)
        self.v_torus = ((initial_idx * 3) * TWO_PI / num_concepts) % TWO_PI
        self.steps_in_role = 0

    def attempt_heterarchical_integration(self, dominant_idx: int):
        if not self.has_fpz or dominant_idx == self.original_idx:
            return False
            
        self.parent_id = dominant_idx
        self.role = "SUB_CHILD"
        self.target_name = f"{ALL_NAMES[self.original_idx]} ⊂ {ALL_NAMES[dominant_idx]}"
        
        dom_u = (dominant_idx * TWO_PI / self.num_concepts)
        dom_v = ((dominant_idx * 3) * TWO_PI / self.num_concepts) % TWO_PI
        self.u_torus = (dom_u + math.pi/4) % TWO_PI
        self.v_torus = (dom_v + math.pi/4) % TWO_PI
        
        self.sprt_log_evidence = 1.0 
        self.steps_in_role = 0
        return True

    def trigger_fpz_cognitive_branch(self):
        if not self.has_fpz: return False 
        self.sprt_log_evidence = 0.0
        old_idx = self.current_idx
        self.plan_b_queue.append(old_idx)
        self.current_idx = self.plan_b_queue.pop(0)
        self.target_name = ALL_NAMES[self.current_idx]
        self.parent_id = None 
        self.u_torus = (self.current_idx * TWO_PI / self.num_concepts)
        self.v_torus = ((self.current_idx * 3) * TWO_PI / self.num_concepts) % TWO_PI
        self.role = "LEADER"
        self.steps_in_role = 0
        return True

    def generate_waves(self, role: str, dp_so3: np.ndarray | None, t_vec: np.ndarray, theta_norm: np.ndarray):
        freq = 35.0 + (self.current_idx % 8) * 3.5
        offset_ch = np.linspace(-1.0, 1.0, NUM_CHANNELS, dtype=np.float32)
        spatial_phase_ch = (COORDS_X * math.cos(self.u_torus) + COORDS_Y * math.sin(self.v_torus)) * 0.18 + offset_ch

        target_phase = 0.25 if role in ["LEADER", "SUPER_PARENT"] else (0.75 if role == "SUB_CHILD" else 0.50)
        amp = 5.0 if role in ["LEADER", "SUPER_PARENT"] else (3.8 if role == "SUB_CHILD" else 4.2)

        w_theta = np.exp(-((theta_norm - target_phase)**2) / 0.025)
        low_gamma = np.sin(TWO_PI * freq * t_vec[None, :] + spatial_phase_ch[:, None]) * w_theta[None, :] * amp
        high_ripple = np.zeros_like(low_gamma)
        
        if dp_so3 is not None:
            w_ripple = np.exp(-((theta_norm - ((target_phase + 0.15) % 1.0))**2) / 0.015)
            phase_ripple = (COORDS_X * dp_so3[0] + COORDS_Y * dp_so3[1] + COORDS_Z * dp_so3[2]) * 0.25
            high_ripple = np.sin(TWO_PI * 89.5 * t_vec[None, :] + phase_ripple[:, None]) * w_ripple[None, :] * 4.5
        return low_gamma, high_ripple

class FullDynamicHardcodedBot(BaseActiveAgent):
    def step(self, effective_probs: np.ndarray):
        """Расчет опирается СТРОГО на поданный эффективный массив (картинка или картинка+монти)."""
        target_p = float(effective_probs[self.current_idx]) if self.current_idx < len(effective_probs) else 0.0
        dominant_idx = int(np.argmax(effective_probs))
        dominant_p = float(effective_probs[dominant_idx])

        if target_p >= 0.28:
            self.sprt_log_evidence = max(0.0, self.sprt_log_evidence - 0.20)
            self.role = "LEADER"
            self.parent_id = None
            self.steps_in_role = 0
            return self.role, None, self.sprt_log_evidence

        if dominant_idx != self.current_idx and dominant_p >= 0.25:
            dom_u = (dominant_idx * TWO_PI / self.num_concepts)
            dom_v = ((dominant_idx * 3) * TWO_PI / self.num_concepts) % TWO_PI
            d_torus = torus_geodesic_distance(self.u_torus, self.v_torus, dom_u, dom_v) if self.has_afz else math.pi

            evidence_step = math.log((dominant_p + 1e-4) / (target_p + 1e-4)) * (0.5 + d_torus / math.pi)
            self.sprt_log_evidence += 0.04 * evidence_step
            self.steps_in_role += 1

            if self.sprt_log_evidence > 1.35:
                if self.attempt_heterarchical_integration(dominant_idx):
                    return "CHILD_INTEGRATION", None, self.sprt_log_evidence
                elif self.trigger_fpz_cognitive_branch():
                    return "BRANCH_HOP", None, self.sprt_log_evidence
                else:
                    self.sprt_log_evidence = 1.35

            du = self.u_torus - dom_u
            dv = self.v_torus - dom_v
            dp_so3 = np.array([math.sin(du), math.cos(dv), math.sin(du + dv)], dtype=np.float32)
            if self.role != "SUB_CHILD": dp_so3 = -dp_so3
            return self.role, dp_so3, self.sprt_log_evidence

        self.sprt_log_evidence = max(0.0, self.sprt_log_evidence - 0.05)
        return self.role, None, self.sprt_log_evidence

class FullJepaVideoAgent(BaseActiveAgent):
    def __init__(self, bot_id: int, initial_idx: int, active_nodes: list, montage: CorticalMontage, jepa_wrapper, num_concepts: int = 8):
        super().__init__(bot_id, initial_idx, active_nodes, montage, num_concepts)
        self.jepa = jepa_wrapper
        self.video_buffer = []
        self.last_jepa_energy = 1.0

    def push_frame(self, frame_rgb: np.ndarray):
        self.video_buffer.append(frame_rgb)
        if len(self.video_buffer) > VIDEO_BUFFER_LEN:
            self.video_buffer.pop(0)

    def step(self, effective_probs: np.ndarray):
        if len(self.video_buffer) < 2: return self.role, None, self.sprt_log_evidence

        try:
            target_p = float(effective_probs[self.current_idx]) if self.current_idx < len(effective_probs) else 0.0
            dominant_idx = int(np.argmax(effective_probs))
            dominant_p = float(effective_probs[dominant_idx])

            if self.jepa is not None and len(self.video_buffer) >= 2:
                z_cur = self.jepa.encode_world_state(self.video_buffer[-1])
                norm_val = float(z_cur.abs().mean().item()) if hasattr(z_cur, 'abs') else 0.5
                self.last_jepa_energy = norm_val

            if target_p >= 0.28:
                self.sprt_log_evidence = max(0.0, self.sprt_log_evidence - 0.25)
                self.role = "LEADER"
                self.parent_id = None
                self.steps_in_role = 0
                return self.role, None, self.sprt_log_evidence

            if dominant_idx != self.current_idx and dominant_p >= 0.25:
                dom_u = (dominant_idx * TWO_PI / self.num_concepts)
                dom_v = ((dominant_idx * 3) * TWO_PI / self.num_concepts) % TWO_PI
                d_torus = torus_geodesic_distance(self.u_torus, self.v_torus, dom_u, dom_v) if self.has_afz else math.pi

                evidence_step = math.log((dominant_p + 1e-4) / (target_p + 1e-4)) * (0.6 + 0.4 * self.last_jepa_energy)
                self.sprt_log_evidence += 0.045 * evidence_step
                self.steps_in_role += 1

                if self.sprt_log_evidence > 1.35:
                    if self.attempt_heterarchical_integration(dominant_idx):
                        return "CHILD_INTEGRATION", None, self.sprt_log_evidence
                    elif self.trigger_fpz_cognitive_branch():
                        return "BRANCH_HOP", None, self.sprt_log_evidence
                    else:
                        self.sprt_log_evidence = 1.35

                du = self.u_torus - dom_u
                dv = self.v_torus - dom_v
                dp_so3 = np.array([math.sin(du), math.cos(dv), math.sin(du + dv)], dtype=np.float32)
                if self.role != "SUB_CHILD": dp_so3 = -dp_so3

                return self.role, dp_so3, self.sprt_log_evidence

            self.sprt_log_evidence = max(0.0, self.sprt_log_evidence - 0.05)
            return self.role, None, self.sprt_log_evidence

        except Exception:
            return self.role, None, 0.0

class AutonomousSwarmProcess(mp.Process):
    def __init__(self, shm, config_path: str, num_hardcoded: int, num_jepa: int, num_concepts: int = 8, sps: float = 250.0):
        super().__init__()
        self.daemon = True
        self.shm = shm
        self.config_path = config_path
        self.num_hardcoded = num_hardcoded
        self.num_jepa = num_jepa
        self.num_concepts = num_concepts
        self.sps = float(sps)

    def run(self):
        FS = self.sps
        try:
            config_dict = {}
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
            else:
                config_dict = {
                    "montage": {
                        "system": "10-20-extended",
                        "devices": [
                            {"id": 0, "name": "F3", "coords": [-4.0, 3.0, 5.0]},
                            {"id": 1, "name": "F4", "coords": [4.0, 3.0, 5.0]},
                            {"id": 2, "name": "AFz", "coords": [0.0, 6.0, 4.0]},
                            {"id": 3, "name": "Fpz", "coords": [0.0, 8.0, 2.0]}
                        ]
                    },
                    "agents": []
                }
                if self.num_hardcoded > 0:
                    config_dict["agents"].append({
                        "type": "hardcoded", "count": self.num_hardcoded,
                        "profile": "CLI_Hardcoded_Bot", "bind_mode": "names",
                        "bind_target": ["F3", "F4", "AFz", "Fpz"]
                    })
                if self.num_jepa > 0:
                    config_dict["agents"].append({
                        "type": "jepa", "count": self.num_jepa,
                        "profile": "CLI_JEPA_Bot", "bind_mode": "names",
                        "bind_target": ["F3", "F4", "AFz", "Fpz"]
                    })

            montage = CorticalMontage(config_dict)

            outlets = [
                StreamOutlet(StreamInfo(f'FreeEEG_Node{i}', 'EEG', NUM_CHANNELS, FS, 'float32', f'sim_node_{i}'))
                for i in range(NUM_DEVICES)
            ]

            jepa_wrapper = None
            if any(a.get('type') == 'jepa' for a in config_dict.get('agents', [])):
                try:
                    from vla_jepa_wrapper import VLA_JEPA_Wrapper
                    jepa_wrapper = VLA_JEPA_Wrapper(port=6001)
                except Exception as e:
                    print(f"⚠️ [SWARM] VLA-JEPA недоступна: {e}")

            rng = np.random.RandomState(42)
            available_slots = list(rng.permutation(self.num_concepts))
            self.bots = []
            bot_idx = 1

            for agent_cfg in config_dict.get('agents', []):
                count = agent_cfg.get('count', 1)
                for _ in range(count):
                    c_idx = available_slots[(bot_idx - 1) % len(available_slots)]
                    mode = agent_cfg.get('bind_mode', 'names')
                    if mode == 'radius':
                        tgt = agent_cfg.get('bind_target', {})
                        active_nodes = montage.get_nodes_by_radius(tgt.get('anchor', [0,0,0]), tgt.get('radius', 5.0))
                    else:
                        active_nodes = montage.get_nodes_by_names(agent_cfg.get('bind_target', []))
                    
                    if not active_nodes: active_nodes = [0]
                    
                    if agent_cfg.get('type') == 'jepa':
                        bot = FullJepaVideoAgent(bot_idx, c_idx, active_nodes, montage, jepa_wrapper, self.num_concepts)
                    else:
                        bot = FullDynamicHardcodedBot(bot_idx, c_idx, active_nodes, montage, self.num_concepts)
                        
                    self.bots.append(bot)
                    bot_idx += 1

            start_time = time.time()
            regional_delays = [0.0, 0.035, 0.070, 0.105]
            eval_lock = threading.Lock()

            def async_jepa_worker():
                while self.shm['is_running'].value:
                    try:
                        if not self.shm['is_calibrating'].value:
                            raw_buf = np.frombuffer(self.shm['raw_rgb_frame'].get_obj(), dtype=np.uint8)
                            if np.any(raw_buf > 0):
                                frame_copy = raw_buf.reshape(384, 512, 3).copy()
                                for bot in self.bots:
                                    if isinstance(bot, FullJepaVideoAgent):
                                        bot.push_frame(frame_copy)
                    except Exception:
                        pass
                    time.sleep(0.08)

            if any(isinstance(b, FullJepaVideoAgent) for b in self.bots):
                threading.Thread(target=async_jepa_worker, daemon=True).start()

            self.shm['is_swarm_ready'].value = True

            while self.shm['is_running'].value:
                dt = CHUNK_SIZE / FS
                t_now = time.time() - start_time
                t_vec = np.linspace(t_now, t_now + dt, CHUNK_SIZE, endpoint=False, dtype=np.float32)
                theta_phase_norm = ((TWO_PI * 6.0 * t_vec) % TWO_PI) / TWO_PI

                is_calib = self.shm['is_calibrating'].value
                
                # Изолированное вычисление вероятностей (Без метагейминга!)
                world_probs = np.array(self.shm['clip_probs'][:self.num_concepts], dtype=np.float32)
                
                if self.shm['sensory_sub'].value:
                    priors = np.array(self.shm['swarm_priors'][:self.num_concepts], dtype=np.float32)
                    effective_probs = world_probs * 0.5 + priors * 0.5
                else:
                    effective_probs = world_probs

                device_lg = [np.zeros((NUM_CHANNELS, CHUNK_SIZE), dtype=np.float32) for _ in range(NUM_DEVICES)]
                device_hg = [np.zeros((NUM_CHANNELS, CHUNK_SIZE), dtype=np.float32) for _ in range(NUM_DEVICES)]
                owners = [-1] * 8
                status_str = "SWARM:" if not is_calib else f"CALIBRATING: [{ALL_NAMES[self.shm['calib_target_idx'].value]}]"

                if not is_calib:
                    for b_idx, bot in enumerate(self.bots):
                        role, dp, sprt_val = bot.step(effective_probs)
                        owners[bot.current_idx] = bot.bot_id
                        
                        self.shm['parent_map'][bot.original_idx] = bot.parent_id if bot.parent_id is not None else -1

                        lg, hg = bot.generate_waves(role, dp, t_vec, theta_phase_norm)
                        for node_i in bot.active_nodes:
                            if node_i < NUM_DEVICES:
                                device_lg[node_i] += lg
                                device_hg[node_i] += hg

                        prefix = "H" if isinstance(bot, FullDynamicHardcodedBot) else "J"
                        nodes_str = "".join([str(n) for n in bot.active_nodes])
                        status_str += f" [{prefix}{bot.bot_id}({nodes_str})-{bot.target_name[:4]}:{role[:3]}|E:{sprt_val:.2f}]"

                for i in range(NUM_DEVICES):
                    device_lg[i] = np.clip(device_lg[i], -6.0, 6.0)
                    device_hg[i] = np.clip(device_hg[i], -6.0, 6.0)

                for i in range(8): self.shm['concept_owners'][i] = owners[i]
                self.shm['agent_desc'].value = status_str.encode('utf-8', errors='replace')[:250]

                for node_i in range(NUM_DEVICES):
                    if is_calib and not self.bots:
                        raw_sig = np.zeros((NUM_CHANNELS, CHUNK_SIZE), dtype=np.float32)
                    else:
                        delay = regional_delays[node_i]
                        noise = np.random.normal(0, 0.035, (NUM_CHANNELS, CHUNK_SIZE))
                        raw_sig = (
                            3.0 * np.sin(TWO_PI * 6.0 * t_vec + delay)[None, :] +
                            2.0 * np.sin(TWO_PI * 1.5 * t_vec + delay)[None, :] +
                            device_lg[node_i] +
                            device_hg[node_i] +
                            np.sin(TWO_PI * 22.0 * t_vec)[None, :] * 1.8 +
                            noise
                        )
                        if node_i == 2 and not is_calib and np.any(device_hg[node_i]):
                            raw_sig += device_hg[node_i] * 1.5

                    outlets[node_i].push_chunk(raw_sig.T.tolist())

                time.sleep(dt)

        except Exception as err:
            print(f"❌ [SWARM CRASH]: {err}")
            traceback.print_exc()

class SyntheticAutonomousAgent:
    def __init__(self, config_path: str = "swarm_config.json", num_hardcoded: int = 1, num_jepa: int = 0, num_concepts: int = 8, sps: float = 250.0):
        ctx = mp.get_context('spawn')
        self.num_concepts = num_concepts
        self.shm = {
            'is_running': ctx.Value('b', True),
            'is_calibrating': ctx.Value('b', True),
            'is_swarm_ready': ctx.Value('b', False),
            'sensory_sub': ctx.Value('b', False), 
            'calib_target_idx': ctx.Value('i', 0),
            'clip_probs': ctx.Array('d', [0.0] * 256),
            'swarm_priors': ctx.Array('d', [0.0] * 256), 
            'raw_rgb_frame': ctx.Array('B', 384 * 512 * 3),
            'agent_desc': ctx.Array('c', 256),
            'concept_owners': ctx.Array('i', [-1] * 8),
            'parent_map': ctx.Array('i', [-1] * 16) 
        }
        self.process = AutonomousSwarmProcess(
            self.shm, config_path, num_hardcoded=num_hardcoded, num_jepa=num_jepa, num_concepts=num_concepts, sps=sps
        )
        self.process.start()

    def is_ready(self) -> bool: return bool(self.shm['is_swarm_ready'].value)
    def is_alive(self) -> bool: return self.process.is_alive()
    
    def set_sensory_substitution(self, state: bool):
        self.shm['sensory_sub'].value = state
        
    def update_swarm_priors(self, priors: np.ndarray):
        for i in range(min(self.num_concepts, len(priors))):
            self.shm['swarm_priors'][i] = float(priors[i])
            
    def update_visual_state(self, visual_data):
        if isinstance(visual_data, np.ndarray):
            if visual_data.ndim == 3:
                np.copyto(np.frombuffer(self.shm['raw_rgb_frame'].get_obj(), dtype=np.uint8), visual_data.reshape(-1))
            elif visual_data.ndim == 1:
                for i in range(min(self.num_concepts, len(visual_data))):
                    self.shm['clip_probs'][i] = float(visual_data[i])

    def get_parent_map(self):
        pmap = list(self.shm['parent_map'][:self.num_concepts])
        return {i: (p if p != -1 else None) for i, p in enumerate(pmap)}

    def set_calibration_target(self, active: bool, tgt_idx: int = 0):
        self.shm['is_calibrating'].value = bool(active)
        self.shm['calib_target_idx'].value = int(tgt_idx)

    def get_telemetry(self):
        return self.shm['agent_desc'].value.decode('utf-8', errors='replace').replace('\x00', '').strip()

    def get_owners(self):
        return list(self.shm['concept_owners'][:8])

    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=1.0)
