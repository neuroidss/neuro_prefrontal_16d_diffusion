#!/usr/bin/env python3
"""
🧠 NEURO-HETERARCHY CORE v127.0 (DUAL-RATE AUTO-SPS & SINC3 RIPPLE COMPENSATED)
- Автоматическое определение частоты потока (250 SPS или 500 SPS) из LSL.
- Zero-Padding 1024 для Дельты и Теты: шаг сетки 0.24 Гц (прощай, залипание на 1.95 Гц!).
- Аппаратная компенсация sinc3-спада АЦП для рипплов Дики (89.5 Гц) при 250 SPS.
- Защита от алиасинга для Fast Ripples при 250 SPS (ограничение до Найквиста 120 Гц).
"""

import os
import time
import math
import ctypes
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
import torch
from pylsl import StreamInlet, resolve_streams

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

DEFAULT_FS = 250.0
BUF_SIZE = 256
FFT_SLOW = 1024  # Zero-padding для непрерывного спектрального разрешения
NUM_CHANNELS = 16
NUM_MAX_DEVICES = 4
NUM_SLOTS = 32
NUM_PAIRS = 120
TWO_PI = 2.0 * math.pi

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Геометрия 26-мм датчика FreeEEG16-alpha2
COORDS_X = np.array([
    10.14,  7.43,  2.75,  2.72, -2.72, -2.75, -7.42, -10.14,
   -10.14, -7.43, -2.75, -2.72,  2.72,  2.75,  7.43,  10.14
], dtype=np.float32)
COORDS_Y = np.array([
    -2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42,  -2.73,
     2.72,  7.43,  4.76,  10.14, 10.15,  4.77,  7.42,   2.71
], dtype=np.float32)
I_IDX, J_IDX = np.triu_indices(NUM_CHANNELS, k=1)

DX_PAIR = (COORDS_X[J_IDX] - COORDS_X[I_IDX]).astype(np.float32)
DY_PAIR = (COORDS_Y[J_IDX] - COORDS_Y[I_IDX]).astype(np.float32)

DX_GPU = torch.from_numpy(DX_PAIR).to(DEVICE)
DY_GPU = torch.from_numpy(DY_PAIR).to(DEVICE)
I_GPU  = torch.from_numpy(I_IDX).to(DEVICE, dtype=torch.long)
J_GPU  = torch.from_numpy(J_IDX).to(DEVICE, dtype=torch.long)

@dataclass
class Kinematics4D:
    lx: float
    ly: float
    rx: float
    ry: float

@dataclass
class NodeState:
    device_id: int
    name: str
    source_id: str
    is_connected: bool
    phase_theta: float
    phase_delta: float
    beta_stability: float
    beta_power: float
    gamma_power: float
    gating_ratio: float
    torus_u: float
    torus_v: float
    disp_xyz: np.ndarray
    pose_matrix: np.ndarray
    kinematics: Kinematics4D
    iplv_gamma: np.ndarray        
    iplv_human_ripple: np.ndarray 
    iplv_fast_ripple: np.ndarray  

    @property
    def gamepad_axes(self) -> Kinematics4D:
        return self.kinematics

@dataclass
class UniversalFrame:
    nodes: list[NodeState]
    theta_freq: float
    delta_freq: float
    theta_sync: float
    theta_phase: float
    delta_phase: float
    is_real: bool
    num_live: int

def make_lie_so3_generator(omega: float, axis=np.array([0.0, 0.0, 1.0], dtype=np.float32)):
    norm = np.linalg.norm(axis) + 1e-7
    kx, ky, kz = axis / norm
    M_skew = np.array([
        [ 0.0, -kz,   ky],
        [ kz,   0.0, -kx],
        [-ky,   kx,   0.0]
    ], dtype=np.float32) * float(omega)
    return torch.from_numpy(M_skew).to(DEVICE)

class GPU_Daemon_Process(mp.Process):
    def __init__(self, shared_mem, gamma_max: float = 65.0):
        super().__init__()
        self.daemon = True
        self.shm = shared_mem
        self.gamma_max = gamma_max

    def build_spectral_filters(self, fs: float):
        """Строит спектральные фильтры под текущую частоту дискретизации fs."""
        freqs = torch.fft.fftfreq(BUF_SIZE, d=1.0/fs).to(DEVICE)
        freqs_slow = torch.fft.rfftfreq(FFT_SLOW, d=1.0/fs).to(DEVICE)

        # Нотч-фильтры 50/100 Гц
        notch = torch.ones_like(freqs)
        notch[(torch.abs(freqs) >= 48.5) & (torch.abs(freqs) <= 51.5)] = 0.0
        if fs > 220.0:
            notch[(torch.abs(freqs) >= 98.5) & (torch.abs(freqs) <= 101.5)] = 0.0
        notch = notch.view(1, 1, BUF_SIZE)

        # Макро-ритмы Гильберта для извлечения фазы
        f_delta = (torch.exp(-0.5 * ((freqs - 1.5) / 0.6)**2) * 2.0).view(1, 1, BUF_SIZE)
        f_delta[:, :, freqs < 0] = 0.0

        f_theta = (torch.exp(-0.5 * ((freqs - 6.0) / 1.5)**2) * 2.0).view(1, 1, BUF_SIZE)
        f_theta[:, :, freqs < 0] = 0.0

        f_beta = (torch.exp(-0.5 * ((freqs - 22.0) / 5.0)**2) * 2.0).view(1, 1, BUF_SIZE)
        f_beta[:, :, freqs < 0] = 0.0

        freqs_4d = freqs.view(1, 1, 1, BUF_SIZE)

        # Контур А: Гамма 30 - gamma_max Гц
        max_g = min(self.gamma_max, fs * 0.46)
        gamma_centers = torch.linspace(30.0, max_g, NUM_SLOTS, device=DEVICE).view(1, NUM_SLOTS, 1, 1)
        bandwidth = 3.5 if max_g <= 65.0 else 4.5
        gamma_filters = torch.exp(-0.5 * ((freqs_4d - gamma_centers) / bandwidth)**2) * 2.0
        gamma_filters[:, :, :, freqs < 0] = 0.0

        # Контур Б: Рипплы Дики (70-100 Гц, пик 89.5 Гц)
        h_rip_max = min(100.0, fs * 0.46)
        h_rip_min = min(70.0, h_rip_max - 15.0)
        h_ripple_centers = torch.linspace(h_rip_min, h_rip_max, NUM_SLOTS, device=DEVICE).view(1, NUM_SLOTS, 1, 1)
        h_ripple_filters = torch.exp(-0.5 * ((freqs_4d - h_ripple_centers) / 4.0)**2) * 2.0
        h_ripple_filters[:, :, :, freqs < 0] = 0.0

        # АППАРАТНАЯ КОМПЕНСАЦИЯ SINC3-СПАДА АЦП ДЛЯ РИППЛОВ
        # H_sinc3(f) = (sinc(f / fs))^3. Домножаем на обратную функцию!
        sinc_ratio = torch.sinc(h_ripple_centers / fs)
        sinc_comp = (1.0 / (sinc_ratio + 1e-4)) ** 3
        h_ripple_filters = h_ripple_filters * sinc_comp

        # Контур В: Fast Ripples (с защитой от Найквиста)
        if fs <= 260.0:
            # При 250 SPS верхний предел - 120 Гц
            f_ripple_centers = torch.linspace(95.0, 120.0, NUM_SLOTS, device=DEVICE).view(1, NUM_SLOTS, 1, 1)
            f_bandwidth = 3.0
        else:
            # При 500 SPS полный диапазон до 200 Гц
            f_ripple_centers = torch.linspace(100.0, 200.0, NUM_SLOTS, device=DEVICE).view(1, NUM_SLOTS, 1, 1)
            f_bandwidth = 7.0

        f_ripple_filters = torch.exp(-0.5 * ((freqs_4d - f_ripple_centers) / f_bandwidth)**2) * 2.0
        f_ripple_filters[:, :, :, freqs < 0] = 0.0

        return freqs, freqs_slow, notch, f_delta, f_theta, f_beta, gamma_filters, h_ripple_filters, f_ripple_filters

    def run(self):
        current_fs = DEFAULT_FS
        freqs, freqs_slow, notch, f_delta, f_theta, f_beta, gamma_filters, h_ripple_filters, f_ripple_filters = self.build_spectral_filters(current_fs)

        slot_angles = (-math.pi + (TWO_PI / NUM_SLOTS) * (torch.arange(NUM_SLOTS, device=DEVICE) + 0.5)).view(1, NUM_SLOTS, 1, 1)

        inlets = [None] * NUM_MAX_DEVICES
        stream_names = ["Empty"] * NUM_MAX_DEVICES
        stream_uids = [""] * NUM_MAX_DEVICES
        connected_uids = set()

        raw_buffers = np.zeros((NUM_MAX_DEVICES, NUM_CHANNELS, BUF_SIZE), dtype=np.float32)
        raw_buf_gpu = torch.zeros((NUM_MAX_DEVICES, NUM_CHANNELS, BUF_SIZE), device=DEVICE, dtype=torch.float32)

        # Привязка Shared Memory
        sh_dev_th_phase   = np.frombuffer(self.shm['dev_th_phase'].get_obj(), dtype=np.float64)
        sh_dev_dl_phase   = np.frombuffer(self.shm['dev_dl_phase'].get_obj(), dtype=np.float64)
        sh_beta_power     = np.frombuffer(self.shm['beta_power'].get_obj(), dtype=np.float64)
        sh_gamma_power    = np.frombuffer(self.shm['gamma_power'].get_obj(), dtype=np.float64)
        sh_gating_ratio   = np.frombuffer(self.shm['gating_ratio'].get_obj(), dtype=np.float64)
        sh_torus_coords   = np.frombuffer(self.shm['torus_coords'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 2)
        sh_kinematics     = np.frombuffer(self.shm['kinematics'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 4)
        sh_disp           = np.frombuffer(self.shm['disp'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 3)
        sh_pose           = np.frombuffer(self.shm['pose'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 9)
        sh_beta_stab      = np.frombuffer(self.shm['beta_stab'].get_obj(), dtype=np.float64)
        sh_iplv_gamma     = np.frombuffer(self.shm['iplv_gamma'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS, NUM_PAIRS)
        sh_iplv_h_ripple  = np.frombuffer(self.shm['iplv_h_ripple'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS, NUM_PAIRS)
        sh_iplv_f_ripple  = np.frombuffer(self.shm['iplv_f_ripple'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS, NUM_PAIRS)

        prev_beta_vecs = torch.zeros((NUM_MAX_DEVICES, 2), device=DEVICE)
        last_resolve_time = 0.0

        while self.shm['is_running'].value:
            now = time.time()

            # Сканирование и подключение LSL потоков
            if (None in inlets) and (now - last_resolve_time > 1.2):
                last_resolve_time = now
                try:
                    streams = resolve_streams(wait_time=0.1)
                    streams = sorted(streams, key=lambda s: s.name())
                    for s in streams:
                        s_uid = s.uid()
                        if s_uid not in connected_uids and s.channel_count() == NUM_CHANNELS:
                            for slot_i in range(NUM_MAX_DEVICES):
                                if inlets[slot_i] is None:
                                    try:
                                        inlets[slot_i] = StreamInlet(s, max_buflen=1, max_chunklen=32, recover=True)
                                        connected_uids.add(s_uid)
                                        stream_names[slot_i] = s.name()
                                        stream_uids[slot_i] = s.source_id()
                                        
                                        # АВТО-ОПРЕДЕЛЕНИЕ SPS ИЗ МЕТАДАННЫХ ПОТОКА
                                        inlet_srate = inlets[slot_i].info().nominal_srate()
                                        detected_sps = 500.0 if inlet_srate > 350.0 else 250.0
                                        if detected_sps != current_fs:
                                            current_fs = detected_sps
                                            freqs, freqs_slow, notch, f_delta, f_theta, f_beta, gamma_filters, h_ripple_filters, f_ripple_filters = self.build_spectral_filters(current_fs)
                                            print(f"🔄 [AUTO-SPS] Спектральные фильтры перестроены под {current_fs:.0f} SPS!")

                                        print(f"✅ [CORE HAL] Подключен узел [{slot_i}] <- '{s.name()}' (@ {current_fs:.0f} Hz)")
                                        break
                                    except Exception:
                                        pass
                except Exception:
                    pass

            num_live = sum(1 for inl in inlets if inl is not None)
            is_real = (num_live > 0)
            self.shm['is_real'].value = is_real
            self.shm['num_live'].value = num_live

            max_n_pulled = 0

            if is_real:
                for i in range(NUM_MAX_DEVICES):
                    if inlets[i] is not None:
                        try:
                            chunk, _ = inlets[i].pull_chunk(timeout=0.0, max_samples=BUF_SIZE)
                            if chunk:
                                arr = np.array(chunk, dtype=np.float32).T
                                n = arr.shape[1]
                                if n > max_n_pulled:
                                    max_n_pulled = n
                                    
                                if n >= BUF_SIZE:
                                    raw_buffers[i] = arr[:NUM_CHANNELS, -BUF_SIZE:]
                                else:
                                    raw_buffers[i] = np.roll(raw_buffers[i], -n, axis=1)
                                    raw_buffers[i][:, -n:] = arr[:NUM_CHANNELS, :]
                        except Exception:
                            if stream_uids[i] in connected_uids:
                                connected_uids.remove(stream_uids[i])
                            inlets[i] = None
                            stream_names[i] = "Empty"
                            
                if max_n_pulled == 0:
                    time.sleep(0.001)
                    continue
                    
                raw_buf_gpu.copy_(torch.from_numpy(raw_buffers))
            else:
                time.sleep(0.001)
                continue

            dt_data = max_n_pulled / current_fs

            with torch.inference_mode():
                centered = raw_buf_gpu - torch.mean(raw_buf_gpu, dim=2, keepdim=True)
                fft_clean = torch.fft.fft(centered, dim=-1) * notch

                first_active = 0
                for idx in range(NUM_MAX_DEVICES):
                    if inlets[idx] is not None:
                        first_active = idx
                        break

                # =========================================================================
                # 1. МАКРО-РИТМЫ: ZERO-PADDING 1024 ДЛЯ НЕПРЕРЫВНОЙ ТОЧНОСТИ ЧАСТОТ
                # =========================================================================
                fft_1024 = torch.fft.rfft(centered[first_active], n=FFT_SLOW, dim=-1) # [16, 513]

                # --- ДЕЛЬТА (0.5 - 3.2 Гц): Спектральный Центроид по 1024 точкам ---
                dl_mask = (freqs_slow >= 0.5) & (freqs_slow <= 3.2)
                dl_pwr = torch.sum(torch.abs(fft_1024[:, dl_mask])**2, dim=0) # [N_dl_bins]
                total_dl = torch.sum(dl_pwr)
                if total_dl > 1e-6:
                    raw_delta_hz = (torch.sum(dl_pwr * freqs_slow[dl_mask]) / total_dl).item()
                else:
                    raw_delta_hz = 1.5
                inst_delta_freq = float(np.clip(raw_delta_hz, 0.5, 3.2))

                # --- ТЕТА (3.8 - 8.5 Гц): Спектральный Центроид по 1024 точкам ---
                th_mask = (freqs_slow >= 3.8) & (freqs_slow <= 8.5)
                th_pwr = torch.sum(torch.abs(fft_1024[:, th_mask])**2, dim=0) # [N_th_bins]
                total_th = torch.sum(th_pwr)
                if total_th > 1e-6:
                    raw_theta_hz = (torch.sum(th_pwr * freqs_slow[th_mask]) / total_th).item()
                else:
                    raw_theta_hz = 6.0
                inst_theta_freq = float(np.clip(raw_theta_hz, 3.5, 9.0))

                # Извлечение фаз для Тора и PAC через аналитический Гильберт (256 точек)
                Z_theta = torch.fft.ifft(fft_clean * f_theta, dim=-1)
                P_theta = Z_theta / (torch.abs(Z_theta) + 1e-12)
                mean_th_phasors = torch.mean(P_theta, dim=1)
                phi_theta_all = torch.angle(mean_th_phasors)

                Z_delta = torch.fft.ifft(fft_clean * f_delta, dim=-1)
                P_delta = Z_delta / (torch.abs(Z_delta) + 1e-12)
                mean_dl_phasor = torch.mean(P_delta, dim=1)
                phi_delta_all = torch.angle(mean_dl_phasor)

                self.shm['theta_phase'].value = float(phi_theta_all[first_active, -1].item())
                self.shm['theta_freq'].value = inst_theta_freq
                self.shm['theta_sync'].value = float(torch.mean(torch.abs(mean_th_phasors[first_active])).item())

                self.shm['delta_phase'].value = float(phi_delta_all[first_active, -1].item())
                self.shm['delta_freq'].value = inst_delta_freq

                # =========================================================
                # 2. Тор Джанаты
                # =========================================================
                torus_u_all = (phi_theta_all[:, -1] + math.pi).cpu().numpy()
                torus_v_all = (phi_delta_all[:, -1] + math.pi).cpu().numpy()
                torus_coords = np.stack([torus_u_all, torus_v_all], axis=-1)

                # =========================================================
                # 3. Бета-ритм (L5/L6 тормозной гейт)
                # =========================================================
                Z_beta = torch.fft.ifft(fft_clean * f_beta, dim=-1)
                b_power = torch.mean(torch.abs(Z_beta)**2, dim=(1, 2)).cpu().numpy()
                
                P_beta = Z_beta / (torch.abs(Z_beta) + 1e-12)
                cg_beta = P_beta[:, I_GPU, :] * torch.conj(P_beta[:, J_GPU, :])
                iplv_beta = torch.mean(torch.imag(cg_beta), dim=-1)

                v_bx = torch.sum(iplv_beta * DX_GPU, dim=-1) / 120.0
                v_by = torch.sum(iplv_beta * DY_GPU, dim=-1) / 120.0
                cur_beta_vecs = torch.stack([v_bx, v_by], dim=-1)

                dot_b = torch.sum(cur_beta_vecs * prev_beta_vecs, dim=-1)
                norm_b = torch.norm(cur_beta_vecs, dim=-1) * torch.norm(prev_beta_vecs, dim=-1) + 1e-6
                beta_stabs = torch.clamp(dot_b / norm_b, -1.0, 1.0)
                prev_beta_vecs.copy_(cur_beta_vecs)

                # =========================================================
                # 4. Фазовые веса тета-цикла
                # =========================================================
                p_diff = phi_theta_all[first_active:first_active+1].view(1, 1, 1, BUF_SIZE) - slot_angles
                w = torch.exp(3.2 * torch.cos(p_diff))
                w = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-6)

                fft_exp = fft_clean.unsqueeze(1)

                # =========================================================
                # 5. Контур А: Локальная Гамма
                # =========================================================
                Z_gamma = torch.fft.ifft(fft_exp * gamma_filters, dim=-1)
                g_power = torch.mean(torch.abs(Z_gamma)**2, dim=(1, 2, 3)).cpu().numpy()
                P_gamma = Z_gamma / (torch.abs(Z_gamma) + 1e-12)
                cg_gamma = P_gamma[:, :, I_GPU, :] * torch.conj(P_gamma[:, :, J_GPU, :])
                psi_gamma = torch.sum(cg_gamma * w, dim=-1)
                
                im_gamma = torch.imag(psi_gamma)
                re_gamma = torch.real(psi_gamma)
                gamma_120 = im_gamma / torch.sqrt(torch.clamp(1.0 - re_gamma**2, min=1e-5))

                gating_ratios = g_power / (g_power + b_power + 1e-6)

                # =========================================================
                # 6. Контур Б: Кортикальные рипплы человека ~89.5 Гц (sinc3 compensated)
                # =========================================================
                Z_h_ripple = torch.fft.ifft(fft_exp * h_ripple_filters, dim=-1)
                P_h_ripple = Z_h_ripple / (torch.abs(Z_h_ripple) + 1e-12)
                cg_h_ripple = P_h_ripple[:, :, I_GPU, :] * torch.conj(P_h_ripple[:, :, J_GPU, :])
                psi_h_ripple = torch.sum(cg_h_ripple * w, dim=-1)
                im_hr = torch.imag(psi_h_ripple)
                re_hr = torch.real(psi_h_ripple)
                h_ripple_120 = im_hr / torch.sqrt(torch.clamp(1.0 - re_hr**2, min=1e-5))

                # =========================================================
                # 7. Контур В: Fast Ripples
                # =========================================================
                Z_f_ripple = torch.fft.ifft(fft_exp * f_ripple_filters, dim=-1)
                P_f_ripple = Z_f_ripple / (torch.abs(Z_f_ripple) + 1e-12)
                cg_f_ripple = P_f_ripple[:, :, I_GPU, :] * torch.conj(P_f_ripple[:, :, J_GPU, :])
                psi_f_ripple = torch.sum(cg_f_ripple * w, dim=-1)
                im_fr = torch.imag(psi_f_ripple)
                re_fr = torch.real(psi_f_ripple)
                f_ripple_120 = im_fr / torch.sqrt(torch.clamp(1.0 - re_fr**2, min=1e-5))

                # =========================================================
                # 8. КИНЕМАТИКА ФАЗОВОГО ПОЛЯ
                # =========================================================
                M_skew = make_lie_so3_generator(omega=TWO_PI * inst_theta_freq)

                v_gx = torch.sum(gamma_120[:, 31] * DX_GPU, dim=-1) / 120.0
                v_gy = torch.sum(gamma_120[:, 31] * DY_GPU, dim=-1) / 120.0
                g_vecs = torch.stack([v_gx, v_gy, v_gx * 0.5], dim=-1)

                disp_all = torch.matmul(g_vecs, M_skew.T) * dt_data * 0.05

                u1 = disp_all / (torch.norm(disp_all, dim=-1, keepdim=True) + 1e-6)
                up_ref = torch.tensor([0.0, 0.0, 1.0], device=DEVICE).view(1, 3).expand(NUM_MAX_DEVICES, 3)
                u2 = torch.cross(u1, up_ref, dim=-1)
                u2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + 1e-6)
                u3 = torch.cross(u1, u2, dim=-1)
                pose_matrices = torch.stack([u1, u2, u3], dim=1)

                kinematics_gpu = torch.stack([v_gx, v_gy, beta_stabs, torch.tensor([inst_theta_freq / 10.0]*4, device=DEVICE)], dim=-1)

                # =========================================================
                # Обновление Shared Memory
                # =========================================================
                np.copyto(sh_dev_th_phase, phi_theta_all[:, -1].cpu().numpy())
                np.copyto(sh_dev_dl_phase, phi_delta_all[:, -1].cpu().numpy())
                np.copyto(sh_beta_power, b_power)
                np.copyto(sh_gamma_power, g_power)
                np.copyto(sh_gating_ratio, gating_ratios)
                np.copyto(sh_torus_coords, torus_coords)
                np.copyto(sh_kinematics, kinematics_gpu.cpu().numpy())
                np.copyto(sh_disp, disp_all.cpu().numpy())
                np.copyto(sh_pose, pose_matrices.contiguous().view(NUM_MAX_DEVICES, 9).cpu().numpy())
                np.copyto(sh_beta_stab, beta_stabs.cpu().numpy())
                np.copyto(sh_iplv_gamma, gamma_120.cpu().numpy())
                np.copyto(sh_iplv_h_ripple, h_ripple_120.cpu().numpy())
                np.copyto(sh_iplv_f_ripple, f_ripple_120.cpu().numpy())

class HeterarchicalBrainEngine:
    def __init__(self, gamma_max: float = 65.0):
        self.gamma_max = gamma_max
        self.roles = ["F3 (Left/Form)", "F4 (Right/Style)", "AFz (Janata Torus)", "Fpz (BA10/Plan B)"]
        self.shm = {
            'is_running': mp.Value(ctypes.c_bool, True),
            'is_real': mp.Value(ctypes.c_bool, False),
            'num_live': mp.Value('i', 0),
            'theta_sync': mp.Value('d', 0.8),
            'theta_freq': mp.Value('d', 6.0),
            'delta_freq': mp.Value('d', 1.5),
            'theta_phase': mp.Value('d', 0.0),
            'delta_phase': mp.Value('d', 0.0),
            'dev_th_phase': mp.Array('d', NUM_MAX_DEVICES),
            'dev_dl_phase': mp.Array('d', NUM_MAX_DEVICES),
            'beta_power': mp.Array('d', NUM_MAX_DEVICES),
            'gamma_power': mp.Array('d', NUM_MAX_DEVICES),
            'gating_ratio': mp.Array('d', NUM_MAX_DEVICES),
            'torus_coords': mp.Array('d', NUM_MAX_DEVICES * 2),
            'kinematics': mp.Array('d', NUM_MAX_DEVICES * 4),
            'disp': mp.Array('d', NUM_MAX_DEVICES * 3),
            'pose': mp.Array('d', NUM_MAX_DEVICES * 9),
            'beta_stab': mp.Array('d', NUM_MAX_DEVICES),
            'iplv_gamma': mp.Array('d', NUM_MAX_DEVICES * NUM_SLOTS * NUM_PAIRS),
            'iplv_h_ripple': mp.Array('d', NUM_MAX_DEVICES * NUM_SLOTS * NUM_PAIRS),
            'iplv_f_ripple': mp.Array('d', NUM_MAX_DEVICES * NUM_SLOTS * NUM_PAIRS)
        }
        self._dev_th_phase   = np.frombuffer(self.shm['dev_th_phase'].get_obj(), dtype=np.float64)
        self._dev_dl_phase   = np.frombuffer(self.shm['dev_dl_phase'].get_obj(), dtype=np.float64)
        self._beta_power     = np.frombuffer(self.shm['beta_power'].get_obj(), dtype=np.float64)
        self._gamma_power    = np.frombuffer(self.shm['gamma_power'].get_obj(), dtype=np.float64)
        self._gating_ratio   = np.frombuffer(self.shm['gating_ratio'].get_obj(), dtype=np.float64)
        self._torus_coords   = np.frombuffer(self.shm['torus_coords'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 2)
        self._kinematics     = np.frombuffer(self.shm['kinematics'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 4)
        self._disp           = np.frombuffer(self.shm['disp'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 3)
        self._pose           = np.frombuffer(self.shm['pose'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 3, 3)
        self._beta_stab      = np.frombuffer(self.shm['beta_stab'].get_obj(), dtype=np.float64)
        self._iplv_gamma     = np.frombuffer(self.shm['iplv_gamma'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS, NUM_PAIRS)
        self._iplv_h_ripple  = np.frombuffer(self.shm['iplv_h_ripple'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS, NUM_PAIRS)
        self._iplv_f_ripple  = np.frombuffer(self.shm['iplv_f_ripple'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS, NUM_PAIRS)
        
        self.process = GPU_Daemon_Process(self.shm, gamma_max=self.gamma_max)

    def start(self): self.process.start()
    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=2.0)

    def get_frame(self) -> UniversalFrame:
        nodes = []
        for i in range(NUM_MAX_DEVICES):
            k = self._kinematics[i]
            tc = self._torus_coords[i]
            nodes.append(NodeState(
                device_id=i,
                name=self.roles[i],
                source_id=f"Slot_{i}",
                is_connected=bool(self._dev_th_phase[i] != 0.0),
                phase_theta=float(self._dev_th_phase[i]),
                phase_delta=float(self._dev_dl_phase[i]),
                beta_stability=float(self._beta_stab[i]),
                beta_power=float(self._beta_power[i]),
                gamma_power=float(self._gamma_power[i]),
                gating_ratio=float(self._gating_ratio[i]),
                torus_u=float(tc[0]),
                torus_v=float(tc[1]),
                disp_xyz=self._disp[i].copy(),
                pose_matrix=self._pose[i].copy(),
                kinematics=Kinematics4D(lx=float(k[0]), ly=float(k[1]), rx=float(k[2]), ry=float(k[3])),
                iplv_gamma=self._iplv_gamma[i].copy(),
                iplv_human_ripple=self._iplv_h_ripple[i].copy(),
                iplv_fast_ripple=self._iplv_f_ripple[i].copy()
            ))
        return UniversalFrame(
            nodes=nodes,
            theta_freq=self.shm['theta_freq'].value,
            delta_freq=self.shm['delta_freq'].value,
            theta_sync=self.shm['theta_sync'].value,
            theta_phase=self.shm['theta_phase'].value,
            delta_phase=self.shm['delta_phase'].value,
            is_real=self.shm['is_real'].value,
            num_live=self.shm['num_live'].value
        )
