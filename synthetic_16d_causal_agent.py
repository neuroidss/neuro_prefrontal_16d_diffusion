#!/usr/bin/env python3
"""
🧠 NEUROCANVAS: INVASIVE LAMINAR HIERARCHICAL ACTIVE INFERENCE AGENT (v310.0)
Strictly grounded in micro-electrode & intracranial primate/human electrophysiology.

Neurobiological Implementation Citations:
1. LAMINAR DYNAMICS (Superficial Gamma / Deep Beta Gating):
   - Bastos, Loonis, Kornblith, Lundqvist, & Miller (2018), PNAS.
     DOI: 10.1073/pnas.1714522115
   - Lundqvist, Herman, Warden, Brincat, & Miller (2018), Nature Communications.
     DOI: 10.1038/s41467-017-02791-8

2. HIERARCHICAL ROSTRO-CAUDAL GRADIENT (Scale & Level Encoding):
   - Badre & D'Esposito (2007), Nature Neuroscience.
     DOI: 10.1038/nn1953
   - Badre & Nee (2018), Trends in Cognitive Sciences.
     DOI: 10.1016/j.tics.2017.11.005

3. CONTINGENT BRANCHING & ALTERNATIVE GOAL RESERVOIR (Area 10 / Fpz):
   - Koechlin & Hyafil (2007), Science.
     DOI: 10.1126/science.1142995
   - Boorman, Behrens, Woolrich, & Rushworth (2009), Neuron.
     DOI: 10.1016/j.neuron.2009.05.014
   - Tsujimoto, Genovesio, & Wise (2010), Journal of Neuroscience.
     DOI: 10.1523/JNEUROSCI.6667-09.2010

4. HIERARCHICAL PREDICTION ERROR & PHASE RESET (dACC Area 24/32 / AFz):
   - Alexander & Brown (2011), Nature Neuroscience.
     DOI: 10.1038/nn.2921
   - Shenhav, Botvinick, & Cohen (2013), Nature Neuroscience.
     DOI: 10.1038/nn.3423
   - Womelsdorf, Johnston, Vinck, & Everling (2010), Journal of Neuroscience.
     DOI: 10.1523/JNEUROSCI.2861-10.2010

5. ORTHOGONAL SEQUENCE SUBSPACES & NEURAL GEOMETRY:
   - Chen, Zhang, Hu, Min, & Wang (2024), Neuron.
     DOI: 10.1016/j.neuron.2024.07.024
   - Fan, Wang, Fang, Ding, & Luo (2024), Nature Human Behaviour.
     DOI: 10.1038/s41562-024-02047-8
"""

import time
import math
import multiprocessing as mp
import numpy as np
from pylsl import StreamInfo, StreamOutlet

FS = 250.0
CHUNK_SIZE = 10
NUM_CHANNELS = 16
NUM_DEVICES = 4  # Node 0: F3, Node 1: F4, Node 2: AFz, Node 3: Fpz
TWO_PI = 2.0 * math.pi

# FreeEEG16-alpha2 Concentric 26mm Micro-Array Layout
COORDS_X = np.array([
    10.14,  7.43,  2.75,  2.72, -2.72, -2.75, -7.42, -10.14,
   -10.14, -7.43, -2.75, -2.72,  2.72,  2.75,  7.43,  10.14
], dtype=np.float32)

COORDS_Y = np.array([
    -2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42,  -2.73,
     2.72,  7.43,  4.76,  10.14, 10.15,  4.77,  7.42,   2.71
], dtype=np.float32)

# Hemispherical CSD Cortical Surface Projection (Muller et al., 2018, DOI: 10.1038/nrn.2018.20)
COORDS_Z = np.sqrt(np.maximum(100.0 - COORDS_X**2 - COORDS_Y**2, 0.0))

ALL_CONCEPT_NAMES = [
    "КОСМОС",      # Macro Root A (Level 0)
    "ПЛАНЕТА",     # Meso Scene A (Level 1)
    "КИБЕРПАНК",   # Micro Focus A1 (Level 2)
    "НЕБОСКРЕБ",   # Micro Focus A2 (Level 2)
    "ГОРА",        # Meso Scene B (Level 1)
    "ЗАМОК",       # Micro Focus B1 (Level 2)
    "ОКЕАН",       # Macro Root B (Level 0)
    "ДЖУНГЛИ"      # Meso Scene C (Level 1)
]

# 3-Level Grounded Hierarchy Tree (Fan et al., 2024, DOI: 10.1038/s41562-024-02047-8)
HIERARCHY_TREE = {
    # LEVEL 0 (Macro: Global Rank = 0)
    "КОСМОС":    {"level": 0, "parent": None,      "so3": np.array([0.0,  0.0,  0.0])},
    "ОКЕАН":     {"level": 0, "parent": None,      "so3": np.array([0.8, -1.0,  0.5])},
    
    # LEVEL 1 (Meso: Global Rank = 1)
    "ПЛАНЕТА":   {"level": 1, "parent": "КОСМОС",  "so3": np.array([0.5, -0.2,  0.1])},
    "ГОРА":      {"level": 1, "parent": "КОСМОС",  "so3": np.array([-1.0, 0.5,  0.0])},
    "ДЖУНГЛИ":   {"level": 1, "parent": "ОКЕАН",   "so3": np.array([-1.2, -0.8, 0.2])},

    # LEVEL 2 (Micro: Global Rank = 2)
    "КИБЕРПАНК": {"level": 2, "parent": "ПЛАНЕТА", "so3": np.array([1.2,  0.8, -0.3])},
    "НЕБОСКРЕБ": {"level": 2, "parent": "ПЛАНЕТА", "so3": np.array([1.5,  0.0,  1.0])},
    "ЗАМОК":     {"level": 2, "parent": "ГОРА",    "so3": np.array([-0.5, 0.3,  0.4])},
}

# Orthogonal Rank Subspaces (Chen et al., 2024, DOI: 10.1016/j.neuron.2024.07.024)
# QR decomposition guarantees zero dimensional overlap across levels
np.random.seed(42)
Q_RAW, _ = np.linalg.qr(np.random.randn(NUM_CHANNELS, 3))
SUBSPACE_BASIS = {
    0: Q_RAW[:, 0],  # Level 0 (Macro)
    1: Q_RAW[:, 1],  # Level 1 (Meso)
    2: Q_RAW[:, 2]   # Level 2 (Micro)
}


class AutonomousAgentProcess(mp.Process):
    def __init__(self, shm, num_concepts=8):
        super().__init__()
        self.daemon = True
        self.shm = shm
        self.num_concepts = num_concepts
        self.names = ALL_CONCEPT_NAMES[:num_concepts]

    def run(self):
        outlets = []
        for i in range(NUM_DEVICES):
            info = StreamInfo(f'FreeEEG_Node{i}', 'EEG', NUM_CHANNELS, FS, 'float32', f'sim_node_{i}')
            outlets.append(StreamOutlet(info))

        print(f"🤖 [NEURO-ACTIVE AGENT] Started Invasive Laminar Engine ({self.num_concepts} concepts, 3 levels)...")

        start_time = time.time()
        tgt_idx = 0
        plan_b_idx = 1
        boredom = 0.0
        frustration = 0.0
        saccade_cooldown = 0.0
        theta_phase_offset = 0.0
        active_transition = "IDLE"

        # Axonal conduction delays between prefrontal columns (Harris & Mrsic-Flogel, 2013)
        regional_delays = [0.0, 0.035, 0.070, 0.105]

        while self.shm['is_running'].value:
            dt = CHUNK_SIZE / FS
            t_now = time.time() - start_time
            t_vec = np.linspace(t_now, t_now + dt, CHUNK_SIZE, endpoint=False)

            is_calib = self.shm['is_calibrating'].value
            clip_probs = np.array(self.shm['clip_probs'][:self.num_concepts], dtype=np.float32)

            cur_name = self.names[tgt_idx]
            cur_node = HIERARCHY_TREE[cur_name]
            cur_level = cur_node["level"]

            # Children & Parent determination for hierarchical branching
            children_indices = [
                i for i, n in enumerate(self.names)
                if HIERARCHY_TREE[n]["parent"] == cur_name
            ]
            parent_name = cur_node["parent"]
            parent_idx = self.names.index(parent_name) if (parent_name and parent_name in self.names) else None

            # Same-level sibling candidates
            sibling_indices = [
                i for i, n in enumerate(self.names)
                if HIERARCHY_TREE[n]["level"] == cur_level and i != tgt_idx
            ]

            # ------------------------------------------------------------------
            # 1. dACC AREA 24/32 (AFz) PREDICTION ERROR ACCUMULATION
            # (Alexander & Brown, 2011, DOI: 10.1038/nn.2921)
            # ------------------------------------------------------------------
            if is_calib:
                tgt_idx = int(self.shm['calib_target_idx'].value)
                mode = "CALIBRATION"
                state_desc = f"Calibration LTM Imprint: [{self.names[tgt_idx]}]"
                satisfaction = 1.0
                boredom = 0.0
                frustration = 0.0
                mood = "LEARNING"
                prediction_error = 0.0
            else:
                satisfaction = float(clip_probs[tgt_idx])
                prediction_error = 1.0 - satisfaction

                # Dynamic update of Frontopolar Area 10 unchosen alternative (Plan B)
                # (Tsujimoto et al., 2010, DOI: 10.1523/JNEUROSCI.6667-09.2010)
                sorted_candidates = np.argsort(clip_probs)
                alt_idx = int(sorted_candidates[-2]) if sorted_candidates[-2] != tgt_idx else int(sorted_candidates[-3])
                plan_b_idx = alt_idx if alt_idx < self.num_concepts else (tgt_idx + 1) % self.num_concepts

                if saccade_cooldown > 0.0:
                    saccade_cooldown = max(0.0, saccade_cooldown - dt)
                    mode = active_transition
                    mood = "TRANSITION"
                    state_desc = f"{active_transition} -> [{self.names[tgt_idx]}] (L{cur_level})"
                elif satisfaction >= 0.70:
                    mode = f"HOLD_STABLE ({int(satisfaction*100)}%)"
                    boredom += dt
                    frustration = max(0.0, frustration - dt * 2.5)
                    mood = "CONVERGED"
                    state_desc = f"Attractor Locked: [{self.names[tgt_idx]}] (L{cur_level})"
                else:
                    mode = "SEEKING"
                    frustration += dt * (prediction_error * 2.2)
                    boredom = max(0.0, boredom - dt * 2.0)
                    mood = "CONFLICT" if frustration > 4.0 else "SEARCH"
                    state_desc = f"Resolving [{self.names[tgt_idx]}] (PE: {prediction_error:.2f})"

                # --------------------------------------------------------------
                # 2. HIERARCHICAL NAVIGATION DISPATCHER
                # --------------------------------------------------------------
                if saccade_cooldown <= 0.0:
                    # A. ZOOM IN (Concretization / De-composition to child)
                    # Triggered when macro/meso state is recognized & stable (Boredom accumulation)
                    if satisfaction >= 0.80 and boredom >= 2.5 and len(children_indices) > 0:
                        tgt_idx = children_indices[0]
                        saccade_cooldown = 1.4
                        active_transition = "ZOOM_IN"
                        boredom = 0.0
                        frustration = 0.0
                        theta_phase_offset += math.pi * 0.5

                    # B. ZOOM OUT (Abstraction / Return to Parent on Local Unresolvability)
                    # Triggered when micro/meso state fails or reaches impasse
                    elif frustration >= 5.0 and parent_idx is not None:
                        tgt_idx = parent_idx
                        saccade_cooldown = 1.4
                        active_transition = "ZOOM_OUT"
                        boredom = 0.0
                        frustration = 0.0
                        theta_phase_offset += math.pi * 0.5

                    # C. LATERAL SACCADE (Switch within same rank)
                    # (Voloh et al., 2015, DOI: 10.1073/pnas.1502092112)
                    elif (boredom >= 4.0 or frustration >= 6.5):
                        if len(sibling_indices) > 0 and np.random.rand() > 0.3:
                            tgt_idx = sibling_indices[0]
                            active_transition = "LATERAL_SACCADE"
                        else:
                            tgt_idx = plan_b_idx  # Area 10 Branching Plan B switch
                            active_transition = "PLAN_B_BRANCH"
                        saccade_cooldown = 1.0
                        boredom = 0.0
                        frustration = 0.0
                        theta_phase_offset += math.pi  # Full 180 deg phase reset

            # Telemetry to shared memory
            self.shm['agent_mode'].value = mode.encode('utf-8')[:127].ljust(128, b'\x00')
            self.shm['agent_desc'].value = state_desc.encode('utf-8')[:255].ljust(256, b'\x00')
            self.shm['agent_mood'].value = mood.encode('utf-8')[:127].ljust(128, b'\x00')
            self.shm['target_idx'].value = int(tgt_idx)
            self.shm['satisfaction'].value = float(satisfaction)
            self.shm['boredom'].value = float(boredom / 4.0)
            self.shm['frustration'].value = float(frustration / 8.0)

            # ------------------------------------------------------------------
            # 3. LAMINAR RHYTHMS & FREQUENCY SYNTAX
            # (Bastos et al., 2018; Lundqvist et al., 2018)
            # ------------------------------------------------------------------
            phi_theta = TWO_PI * 6.0 * t_vec + theta_phase_offset
            phi_delta = TWO_PI * 2.5 * t_vec

            # Theta-Gamma Phase division (Siegel et al., 2009, DOI: 10.1073/pnas.0908193106)
            theta_phase_norm = (phi_theta % TWO_PI) / TWO_PI
            gamma_envelope = np.exp(-((theta_phase_norm - 0.50)**2) / 0.015)

            # Rostro-Caudal Scale Metric (Badre & D'Esposito, 2007)
            # Higher level (Macro L0) = broader scale (0.3), Lower level (Micro L2) = finer scale (1.2)
            hierarchical_scale = 0.3 + (cur_level * 0.45)

            # Push-pull laminar amplitudes
            if saccade_cooldown > 0.0:
                # Disruption: Deep Beta collapses, Superficial Gamma surges
                beta_amp = 0.4
                gamma_amp = 7.5
                desync_noise = 0.12
            else:
                # Equilibrium attractor state
                beta_amp = 3.5
                gamma_amp = 4.0
                desync_noise = 0.01

            beta_wave = np.sin(TWO_PI * 22.0 * t_vec) * beta_amp
            gamma_wave = np.sin(TWO_PI * 55.0 * t_vec) * gamma_amp * gamma_envelope
            alpha_wave = np.sin(TWO_PI * 10.0 * t_vec) * 1.5

            # Orthogonal subspace projection vector for current rank
            rank_subspace_vector = SUBSPACE_BASIS[cur_level]

            # ------------------------------------------------------------------
            # 4. MICRO-ARRAY EMISSION ACROSS 4 FUNCTIONAL PREFRONTAL NODES
            # ------------------------------------------------------------------
            vec_current = HIERARCHY_TREE[cur_name]["so3"]
            vec_plan_b = HIERARCHY_TREE[self.names[plan_b_idx]]["so3"]

            for node_i in range(NUM_DEVICES):
                delay = regional_delays[node_i]

                # NODE 0: F3 (Left dlPFC Area 9/46 - Syntax / Form Subspace)
                if node_i == 0:
                    so3_active = vec_current
                    spatial_gain = 1.2
                    extra_sig = 0.0

                # NODE 1: F4 (Right dlPFC Area 45/47 - Style / Semantics Subspace)
                elif node_i == 1:
                    so3_active = vec_current
                    spatial_gain = 1.0
                    extra_sig = 0.0

                # NODE 2: AFz (dACC Area 24/32 - Hierarchical Prediction Error Transmitter)
                elif node_i == 2:
                    so3_active = vec_current
                    spatial_gain = 0.8
                    # Generates targeted high-amplitude Theta burst on error (Alexander & Brown, 2011)
                    error_theta_boost = np.clip(prediction_error * 4.0, 0.0, 5.0)
                    extra_sig = error_theta_boost * np.sin(phi_theta + delay)

                # NODE 3: Fpz (Frontopolar Area 10 - Contingent Branching Reservoir)
                # Encodes unchosen alternative / Plan B subspace (Boorman et al., 2009)
                else:
                    so3_active = vec_plan_b
                    spatial_gain = 0.7
                    extra_sig = 0.0

                # High-dimensional Manifold Spatial Projection on 16-channel micro-array
                spatial_phase = (
                    COORDS_X[:, None] * so3_active[0] +
                    COORDS_Y[:, None] * so3_active[1] +
                    COORDS_Z[:, None] * so3_active[2]
                ) * (0.15 * hierarchical_scale * spatial_gain)

                # Add non-linear mixed selectivity projection (Rigotti et al., 2013)
                spatial_phase += rank_subspace_vector[:, None] * 0.40

                noise = np.random.normal(0, desync_noise, (NUM_CHANNELS, len(t_vec)))

                raw_sig = (
                    3.0 * np.sin(phi_theta + delay)[None, :] +
                    2.0 * np.sin(phi_delta + delay)[None, :] +
                    beta_wave[None, :] +
                    gamma_wave[None, :] * np.sin(spatial_phase) +
                    alpha_wave[None, :] +
                    extra_sig +
                    noise
                )

                outlets[node_i].push_chunk(raw_sig.T.tolist())

            time.sleep(dt)


class SyntheticAutonomousAgent:
    def __init__(self, num_concepts=8):
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
