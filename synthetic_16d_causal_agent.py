#!/usr/bin/env python3
"""
🧠 AUTONOMOUS PREFRONTAL AGENT (BIOLOGICAL TRAVELING WAVE ENGINE) - v248.1
- Исправлен бродкастинг размерностей (16 каналов x 10 сэмплов чанка).
- Честная бегущая фазовая волна, модулированная тета-циклом (Muller et al., 2018).
- 4 ортогональных волновых вектора для 4 концептов (NW, NE, SE, SW).
- 16-канальный ЭЭГ LSL поток (250 Hz, 4 узла).
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

COORDS_X = np.array([10.14,  7.43,  2.75,  2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72,  2.72,  2.75,  7.43,  10.14], dtype=np.float32)
COORDS_Y = np.array([-2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42,  -2.73,   2.72,  7.43,  4.76,  10.14, 10.15,  4.77,  7.42,   2.71], dtype=np.float32)
IS_CORE = np.hypot(COORDS_X, COORDS_Y) < 8.0

TARGET_NAMES = ["ГОРА", "ЗАМОК", "НЕБОСКРЕБ", "ОКЕАН"]

# 4 Ортогональных волновых вектора направления бегущей волны:
CONCEPT_WAVE_VECTORS = [
    np.array([-1.0, -1.0], dtype=np.float32), # ГОРА (Северо-Запад)
    np.array([ 1.0, -1.0], dtype=np.float32), # ЗАМОК (Северо-Восток)
    np.array([ 1.0,  1.0], dtype=np.float32), # НЕБОСКРЕБ (Юго-Восток)
    np.array([-1.0,  1.0], dtype=np.float32)  # ОКЕАН (Юго-Запад)
]

class AutonomousAgentProcess(mp.Process):
    def __init__(self, shm):
        super().__init__()
        self.daemon = True
        self.shm = shm

    def run(self):
        outlets = []
        for i in range(NUM_DEVICES):
            info = StreamInfo(f'FreeEEG_Node{i}', 'EEG', NUM_CHANNELS, FS, 'float32', f'sim_diff_node_{i}')
            outlets.append(StreamOutlet(info))

        print("🤖 [AUTONOMOUS BRAIN AGENT] Бегущая фазовая волна запущена (LSL 250 Hz)...")
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
            clip_probs = np.array(self.shm['clip_probs'][:], dtype=np.float32)
            
            sorted_order = np.argsort(clip_probs)[::-1]
            active_tokens = [f"{TARGET_NAMES[i]} {int(clip_probs[i] * 100)}%" for i in sorted_order if clip_probs[i] >= 0.04]
            mixture_str = " | ".join(active_tokens) if active_tokens else "СКАНИРОВАНИЕ..."

            if is_calib:
                tgt_idx = int(self.shm['calib_target_idx'].value)
                cur_target_name = TARGET_NAMES[tgt_idx]
                mode = "CALIBRATION"
                state_desc = f"Калибровка эталона [{cur_target_name}]..."
                satisfaction = 1.0
                boredom = 0.0
                frustration = 0.0
                mood = "РЕЖИМ ОБУЧЕНИЯ"
            else:
                cur_target_name = TARGET_NAMES[tgt_idx]
                target_presence = clip_probs[tgt_idx]
                satisfaction = float(target_presence)

                if saccade_cooldown > 0.0:
                    saccade_cooldown = max(0.0, saccade_cooldown - dt)
                    mode = "SACCADE"
                    mood = "САККАДА: Смена фокуса"
                    state_desc = f"Саккада -> ищу [{cur_target_name}]"
                elif satisfaction >= 0.60:
                    mode = f"SATISFIED ({int(satisfaction*100)}%)"
                    boredom += dt
                    frustration = max(0.0, frustration - dt * 2.5)
                    mood = f"УСПЕХ: Наслаждаюсь [{cur_target_name}]!"
                    state_desc = f"Вижу [{cur_target_name}] ({mixture_str}). Скука: {int(boredom/3.5*100)}%"
                else:
                    mode = "SEEKING"
                    frustration += dt
                    boredom = max(0.0, boredom - dt * 2.0)
                    mood = f"ПОИСК: Ожидаю [{cur_target_name}]" if frustration < 4.0 else "ФРУСТРАЦИЯ: Застой холста!"
                    state_desc = f"Ищу [{cur_target_name}] (холст: {mixture_str}). Застой: {int(frustration/7.0*100)}%"

                if (boredom >= 3.5 or frustration >= 7.0) and saccade_cooldown <= 0.0:
                    tgt_idx = (tgt_idx + 1) % 4
                    boredom = 0.0
                    frustration = 0.0
                    saccade_cooldown = 0.6
                    mode = "SACCADE"
                    state_desc = f"Саккада! Переключаюсь на [{TARGET_NAMES[tgt_idx]}]."

            self.shm['agent_mode'].value = mode.encode('utf-8')[:127].ljust(128, b'\x00')
            self.shm['agent_desc'].value = state_desc.encode('utf-8')[:255].ljust(256, b'\x00')
            self.shm['agent_mood'].value = mood.encode('utf-8')[:127].ljust(128, b'\x00')
            self.shm['target_idx'].value = int(tgt_idx)
            self.shm['satisfaction'].value = float(satisfaction)
            self.shm['boredom'].value = float(boredom / 3.5)
            self.shm['frustration'].value = float(frustration / 7.0)

            # ГЕНЕРАЦИЯ БЕГУЩЕЙ ФАЗОВОЙ ВОЛНЫ С ТЕТА-МОДУЛЯЦИЕЙ
            wave_vec = CONCEPT_WAVE_VECTORS[tgt_idx]
            cmd_lx, cmd_ly = wave_vec[0], wave_vec[1]
            cmd_rx = min(1.0, max(boredom / 3.5, frustration / 7.0))

            theta_phase = TWO_PI * 6.0 * t_vec
            gamma_phase = TWO_PI * 55.0 * t_vec
            env_nucleus = (np.clip(np.cos(theta_phase), 0, 1) ** 2)
            noise = np.random.normal(0, 0.005, (NUM_CHANNELS, len(t_vec)))

            # Векторы времени приводятся к форме (1, CHUNK_SIZE) для корректного бродкастинга
            theta_sweep = np.sin(theta_phase)[None, :] # shape (1, 10)
            spatial_base = (COORDS_X * (cmd_lx * 0.45) + COORDS_Y * (cmd_ly * 0.45))[:, None] # shape (16, 1)
            spatial_phase = spatial_base * (1.2 + 0.8 * theta_sweep) # shape (16, 10)

            curl_mod = np.where(IS_CORE, -cmd_rx * 0.7, cmd_rx * 0.7)[:, None] # shape (16, 1)
            
            # shape (16, 10)
            raw_sig = 10.0 * np.sin(theta_phase)[None, :] + \
                      4.5 * env_nucleus[None, :] * np.sin(gamma_phase[None, :] + spatial_phase + curl_mod) + noise

            for node_i in range(NUM_DEVICES):
                outlets[node_i].push_chunk(raw_sig.T.tolist()) # (10, 16)

            time.sleep(dt)

class SyntheticAutonomousAgent:
    def __init__(self):
        ctx = mp.get_context('spawn')
        self.shm = {
            'is_running': ctx.Value('b', True),
            'is_calibrating': ctx.Value('b', True),
            'calib_target_idx': ctx.Value('i', 0),
            'clip_probs': ctx.Array('d', [0.25, 0.25, 0.25, 0.25]),
            'target_idx': ctx.Value('i', 0),
            'agent_mode': ctx.Array('c', 128),
            'agent_desc': ctx.Array('c', 256),
            'agent_mood': ctx.Array('c', 128),
            'satisfaction': ctx.Value('d', 0.0),
            'boredom': ctx.Value('d', 0.0),
            'frustration': ctx.Value('d', 0.0)
        }
        self.process = AutonomousAgentProcess(self.shm)
        self.process.start()

    def update_visual_state(self, probs_4):
        for i in range(4):
            self.shm['clip_probs'][i] = float(probs_4[i])

    def set_calibration_target(self, active: bool, tgt_idx: int = 0):
        self.shm['is_calibrating'].value = bool(active)
        self.shm['calib_target_idx'].value = int(tgt_idx)

    def get_telemetry(self):
        mode = self.shm['agent_mode'].value.decode('utf-8').replace('\x00', '').strip()
        desc = self.shm['agent_desc'].value.decode('utf-8').replace('\x00', '').strip()
        mood = self.shm['agent_mood'].value.decode('utf-8').replace('\x00', '').strip()
        t_idx = self.shm['target_idx'].value
        sat = self.shm['satisfaction'].value
        bor = self.shm['boredom'].value
        fru = self.shm['frustration'].value
        return mode, desc, mood, t_idx, sat, bor, fru

    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=1.0)
