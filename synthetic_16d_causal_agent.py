#!/usr/bin/env python3
"""
🧠 NEUROCANVAS: ROBUST SENSORIMOTOR AGENT (v255.0)
- Четкие ортогональные волновые векторы для концептов (без уплывания фаз).
- Квесты агента строго согласованы с фазовыми отпечатками 120 ребер.
- Поддержка 4 концептов (по умолчанию) и 8 концептов.
- 4 LSL-аутлета по 16 каналов (250 Hz) с тета-гамма PAC.
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

COORDS_X = np.array([
    10.14,  7.43,  2.75,  2.72, -2.72, -2.75, -7.42, -10.14,
   -10.14, -7.43, -2.75, -2.72,  2.72,  2.75,  7.43,  10.14
], dtype=np.float32)

COORDS_Y = np.array([
    -2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42,  -2.73,
     2.72,  7.43,  4.76,  10.14, 10.15,  4.77,  7.42,   2.71
], dtype=np.float32)

IS_CORE = np.hypot(COORDS_X, COORDS_Y) < 8.0

BASE_CONCEPTS_4 = ["ГОРА", "ЗАМОК", "НЕБОСКРЕБ", "ОКЕАН"]
BASE_VECTORS_4 = [
    np.array([-1.0, -1.0], dtype=np.float32),  # ГОРА (Северо-Запад)
    np.array([ 1.0, -1.0], dtype=np.float32),  # ЗАМОК (Северо-Восток)
    np.array([ 1.0,  1.0], dtype=np.float32),  # НЕБОСКРЕБ (Юго-Восток)
    np.array([-1.0,  1.0], dtype=np.float32)   # ОКЕАН (Юго-Запад)
]

EXTRA_CONCEPTS_4 = ["КИБЕРПАНК", "ПУСТЫНЯ", "КОСМОС", "ДЖУНГЛИ"]
EXTRA_VECTORS_4 = [
    np.array([ 0.0, -1.4], dtype=np.float32),
    np.array([ 1.4,  0.0], dtype=np.float32),
    np.array([ 0.0,  1.4], dtype=np.float32),
    np.array([-1.4,  0.0], dtype=np.float32)
]


class AutonomousAgentProcess(mp.Process):
    def __init__(self, shm, num_concepts=4):
        super().__init__()
        self.daemon = True
        self.shm = shm
        self.num_concepts = num_concepts
        if num_concepts == 8:
            self.names = BASE_CONCEPTS_4 + EXTRA_CONCEPTS_4
            self.vectors = BASE_VECTORS_4 + EXTRA_VECTORS_4
        else:
            self.names = BASE_CONCEPTS_4
            self.vectors = BASE_VECTORS_4

    def run(self):
        outlets = []
        for i in range(NUM_DEVICES):
            info = StreamInfo(f'FreeEEG_Node{i}', 'EEG', NUM_CHANNELS, FS, 'float32', f'sim_node_{i}')
            outlets.append(StreamOutlet(info))

        print(f"🤖 [AGENT] Агент запущен на {self.num_concepts} концептах (LSL 250 Hz)...")

        start_time = time.time()
        tgt_idx = 0
        boredom = 0.0
        frustration = 0.0
        saccade_cooldown = 0.0

        while self.shm['is_running'].value:
            dt = CHUNK_SIZE / FS
            t_now = time.time() - start_time
            t_vec = np.linspace(t_now, t_now + dt, CHUNK_SIZE, endpoint=False)

            is_calib = self.shm['is_calibrating'].value
            clip_probs = np.array(self.shm['clip_probs'][:self.num_concepts], dtype=np.float32)

            if is_calib:
                tgt_idx = int(self.shm['calib_target_idx'].value)
                cur_target_name = self.names[tgt_idx]
                mode = "CALIBRATION"
                state_desc = f"Калибровка: [{cur_target_name}]"
                satisfaction = 1.0
                boredom = 0.0
                frustration = 0.0
                mood = "ОБУЧЕНИЕ"
            else:
                cur_target_name = self.names[tgt_idx]
                target_presence = clip_probs[tgt_idx]
                satisfaction = float(target_presence)

                if saccade_cooldown > 0.0:
                    saccade_cooldown = max(0.0, saccade_cooldown - dt)
                    mode = "SACCADE"
                    mood = "САККАДА"
                    state_desc = f"Смена цели на [{cur_target_name}]"
                elif satisfaction >= 0.50:
                    mode = f"SATISFIED ({int(satisfaction*100)}%)"
                    boredom += dt
                    frustration = max(0.0, frustration - dt * 2.5)
                    mood = "УСПЕХ"
                    state_desc = f"Вижу [{cur_target_name}] ({int(satisfaction*100)}%). Скука: {int(boredom/4.0*100)}%"
                else:
                    mode = "SEEKING"
                    frustration += dt
                    boredom = max(0.0, boredom - dt * 2.0)
                    mood = "ПОИСК" if frustration < 6.0 else "ФРУСТРАЦИЯ"
                    state_desc = f"Ищу [{cur_target_name}] (CLIP: {int(satisfaction*100)}%). Застой: {int(frustration/8.0*100)}%"

                if (boredom >= 4.0 or frustration >= 8.0) and saccade_cooldown <= 0.0:
                    tgt_idx = (tgt_idx + 1) % self.num_concepts
                    boredom = 0.0
                    frustration = 0.0
                    saccade_cooldown = 0.8
                    mode = "SACCADE"
                    state_desc = f"Саккада! Переход на [{self.names[tgt_idx]}]."

            self.shm['agent_mode'].value = mode.encode('utf-8')[:127].ljust(128, b'\x00')
            self.shm['agent_desc'].value = state_desc.encode('utf-8')[:255].ljust(256, b'\x00')
            self.shm['agent_mood'].value = mood.encode('utf-8')[:127].ljust(128, b'\x00')
            self.shm['target_idx'].value = int(tgt_idx)
            self.shm['satisfaction'].value = float(satisfaction)
            self.shm['boredom'].value = float(boredom / 4.0)
            self.shm['frustration'].value = float(frustration / 8.0)

            # Генерация бегущей волны по вектору цели
            wave_vec = self.vectors[tgt_idx]
            cmd_lx, cmd_ly = wave_vec[0], wave_vec[1]
            cmd_rx = min(1.0, max(boredom / 4.0, frustration / 8.0))

            theta_phase = TWO_PI * 6.0 * t_vec
            gamma_phase = TWO_PI * 55.0 * t_vec
            env_nucleus = (np.clip(np.cos(theta_phase), 0, 1) ** 2)
            noise = np.random.normal(0, 0.005, (NUM_CHANNELS, len(t_vec)))

            theta_sweep = np.sin(theta_phase)[None, :]
            spatial_base = (COORDS_X * (cmd_lx * 0.45) + COORDS_Y * (cmd_ly * 0.45))[:, None]
            spatial_phase = spatial_base * (1.2 + 0.8 * theta_sweep)
            curl_mod = np.where(IS_CORE, -cmd_rx * 0.7, cmd_rx * 0.7)[:, None]

            raw_sig = 10.0 * np.sin(theta_phase)[None, :] + \
                      4.5 * env_nucleus[None, :] * np.sin(gamma_phase[None, :] + spatial_phase + curl_mod) + noise

            for node_i in range(NUM_DEVICES):
                outlets[node_i].push_chunk(raw_sig.T.tolist())

            time.sleep(dt)


class SyntheticAutonomousAgent:
    def __init__(self, num_concepts=4):
        ctx = mp.get_context('spawn')
        self.num_concepts = num_concepts
        self.shm = {
            'is_running': ctx.Value('b', True),
            'is_calibrating': ctx.Value('b', True),
            'calib_target_idx': ctx.Value('i', 0),
            'clip_probs': ctx.Array('d', [1.0 / num_concepts] * num_concepts + [0.0] * (256 - num_concepts)),
            'target_idx': ctx.Value('i', 0),
            'agent_mode': ctx.Array('c', 128),
            'agent_desc': ctx.Array('c', 256),
            'agent_mood': ctx.Array('c', 128),
            'satisfaction': ctx.Value('d', 0.0),
            'boredom': ctx.Value('d', 0.0),
            'frustration': ctx.Value('d', 0.0)
        }
        self.process = AutonomousAgentProcess(self.shm, num_concepts=num_concepts)
        self.process.start()

    def update_visual_state(self, probs):
        for i in range(min(self.num_concepts, len(probs))):
            self.shm['clip_probs'][i] = float(probs[i])

    def set_calibration_target(self, active: bool, tgt_idx: int = 0):
        self.shm['is_calibrating'].value = bool(active)
        self.shm['calib_target_idx'].value = int(tgt_idx)

    def get_telemetry(self):
        return (
            self.shm['agent_mode'].value.decode('utf-8').replace('\x00', '').strip(),
            self.shm['agent_desc'].value.decode('utf-8').replace('\x00', '').strip(),
            self.shm['agent_mood'].value.decode('utf-8').replace('\x00', '').strip(),
            self.shm['target_idx'].value,
            self.shm['satisfaction'].value,
            self.shm['boredom'].value,
            self.shm['frustration'].value
        )

    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=1.0)
