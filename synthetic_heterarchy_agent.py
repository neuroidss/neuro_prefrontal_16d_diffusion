#!/usr/bin/env python3
"""
🤖 SYNTHETIC HETERARCHY AGENT: 4-NODE INTER-REGIONAL PAC LSL STREAMER
- 4 независимых LSL потока: FreeEEG_Node0 (F3), Node1 (F4), Node2 (AFz), Node3 (Fpz).
- Мультичастотный синтаксис: Delta (эпоха), Theta (часы), Alpha (маска), Beta/Gamma (push-pull).
- Иерархические фазовые задержки между узлами: Fpz -> AFz -> F3/F4.
"""

import time
import math
import multiprocessing as mp
import numpy as np
from pylsl import StreamInfo, StreamOutlet

FS = 250.0
CHUNK_SIZE = 10
NUM_CHANNELS = 16
NUM_DEVICES = 4
TWO_PI = 2.0 * math.pi

COORDS_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14,
                    -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], dtype=np.float32)
COORDS_Y = np.array([-2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42, -2.73,
                     2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], dtype=np.float32)

TARGET_CONCEPTS = ["ГОРА", "ЗАМОК", "НЕБОСКРЕБ", "ОКЕАН"]
CONCEPT_VECTORS = [
    np.array([-1.0, -1.0]),
    np.array([ 1.0, -1.0]),
    np.array([ 1.0,  1.0]),
    np.array([-1.0,  1.0])
]

class HeterarchicalAgentProcess(mp.Process):
    def __init__(self, is_running):
        super().__init__()
        self.daemon = True
        self.is_running = is_running

    def run(self):
        outlets = []
        for i in range(NUM_DEVICES):
            info = StreamInfo(f'FreeEEG_Node{i}', 'EEG', NUM_CHANNELS, FS, 'float32', f'sim_node_{i}')
            outlets.append(StreamOutlet(info))

        print("🤖 [AGENT] Агент активен: стриминг 4-х узлов гетерархии (F3, F4, AFz, Fpz) в LSL...")

        tgt_idx = 0
        state_timer = 0.0
        start_time = time.time()
        
        # Межрегиональные фазовые сдвиги
        node_delays = [0.0, 0.08, 0.16, 0.24]

        while self.is_running.value:
            dt = CHUNK_SIZE / FS
            t_now = time.time() - start_time
            t_vec = np.linspace(t_now, t_now + dt, CHUNK_SIZE, endpoint=False)
            state_timer += dt

            # Смена цели каждые 7 секунд
            if state_timer >= 7.0:
                state_timer = 0.0
                tgt_idx = (tgt_idx + 1) % len(TARGET_CONCEPTS)
                print(f"🎯 [AGENT QUEST] Переключение фокуса на концепт: [{TARGET_CONCEPTS[tgt_idx]}]")

            wave_vec = CONCEPT_VECTORS[tgt_idx]

            # 1. Delta (2.5 Гц) и Theta (6.0 Гц) фазовые часы
            phi_delta = TWO_PI * 2.5 * t_vec
            phi_theta = TWO_PI * 6.0 * t_vec
            env_theta = (np.clip(np.cos(phi_theta), 0, 1) ** 2)

            # 2. Моделирование Beta-Gamma Push-Pull
            # При смене цели (первые 2 секунды) Бета падает, Гамма вспыхивает
            if state_timer < 2.0:
                beta_strength = state_timer / 2.0
                gamma_strength = 1.0 - beta_strength
            else:
                beta_strength = 1.0 # Правило удерживается
                gamma_strength = 0.25

            beta_wave = np.sin(TWO_PI * 22.0 * t_vec) * beta_strength
            gamma_wave = np.sin(TWO_PI * 55.0 * t_vec) * gamma_strength * env_theta
            alpha_wave = np.sin(TWO_PI * 10.0 * t_vec) * 0.5

            for i in range(NUM_DEVICES):
                # Направленный сдвиг фазы по геометрии электродов
                delay = node_delays[i]
                spatial_phase = (COORDS_X * (wave_vec[0] * 0.35) + COORDS_Y * (wave_vec[1] * 0.35))[:, None]

                sig = (
                    6.0 * np.sin(phi_theta + delay)[None, :] +
                    3.0 * np.sin(phi_delta + delay)[None, :] +
                    3.5 * beta_wave[None, :] +
                    5.0 * gamma_wave[None, :] * np.sin(spatial_phase) +
                    2.0 * alpha_wave[None, :] +
                    np.random.normal(0, 0.015, (NUM_CHANNELS, CHUNK_SIZE))
                )
                outlets[i].push_chunk(sig.T.tolist())

            time.sleep(dt)

class SyntheticHeterarchyAgent:
    def __init__(self):
        self.is_running = mp.Value('b', True)
        self.process = HeterarchicalAgentProcess(self.is_running)
        self.process.start()

    def stop(self):
        self.is_running.value = False
        self.process.join(timeout=1.0)
