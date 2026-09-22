# 🧠 NeuroCanvas × TBP.Monty

### Multi-Brain Over-Heterarchy, Continuous 30–100 Hz Prefrontal Manifolds, 89.5 Hz Dickey Ripple Binding, 5-Sigma Contrastive Orthogonalization, and Stigmergic Generative Active Inference

[![DOI:10.1038/s41562-024-02047-8](https://img.shields.io/badge/DOI-10.1038%2Fs41562--024--02047--8-blue.svg)](https://doi.org/10.1038/s41562-024-02047-8)
[![DOI:10.1073/pnas.2107797119](https://img.shields.io/badge/DOI-10.1073%2Fpnas.2107797119-green.svg)](https://doi.org/10.1073/pnas.2107797119)
[![DOI:10.1016/j.neuron.2024.07.024](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2024.07.024-red.svg)](https://doi.org/10.1016/j.neuron.2024.07.024)
[![DOI:10.1016/j.neuron.2018.09.023](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2018.09.023-purple.svg)](https://doi.org/10.1016/j.neuron.2018.09.023)
[![DOI:10.1088/1741-2552/aacfe4](https://img.shields.io/badge/DOI-10.1088%2F1741--2552%2Faacfe4-orange.svg)](https://doi.org/10.1088/1741-2552/aacfe4)
[![DOI:10.1126/science.aaf0941](https://img.shields.io/badge/DOI-10.1126%2Fscience.aaf0941-blue.svg)](https://doi.org/10.1126/science.aaf0941)
[![DOI:10.1371/journal.pone.0012165](https://img.shields.io/badge/DOI-10.1371%2Fjournal.pone.0012165-yellow.svg)](https://doi.org/10.1371/journal.pone.0012165)

---

## 📑 Table of Contents
1. [Paradigm Architecture: Collective Stigmergy & Cortical Generative Active Inference](#1-paradigm-architecture-collective-stigmergy--cortical-generative-active-inference)
2. [Functional Neuroanatomy of Prefrontal Cortical Modules](#2-functional-neuroanatomy-of-prefrontal-cortical-modules)
   - 2.1 [FCz (Supplementary Motor Area / Pre-SMA): Dedicated 4-Axis Kinematic Flight](#21-fcz-supplementary-motor-area--pre-sma-dedicated-4-axis-kinematic-flight)
   - 2.2 [AFz (Rostromedial Prefrontal Cortex / rmPFC / dACC): Topological Torus Geometry ($T^2$)](#22-afz-rostromedial-prefrontal-cortex--rmpfc--dacc-topological-torus-geometry-t2)
   - 2.3 [F3 (Left dlPFC): Categorical Attractors, Symbolic Tokens & $\beta$-Order Gating](#23-f3-left-dlpfc-categorical-attractors-symbolic-tokens--beta-order-gating)
   - 2.4 [F4 (Right dlPFC): Holistic Schemas, Visuospatial Coordinate Frames & Feigenbaum Chaos](#24-f4-right-dlpfc-holistic-schemas-visuospatial-coordinate-frames--feigenbaum-chaos)
   - 2.5 [Fpz (Frontopolar Cortex / BA 10): Cognitive Branching & Counterfactual Tracking](#25-fpz-frontopolar-cortex--ba-10-cognitive-branching--counterfactual-tracking)
3. [Electrophysiological & Biophysical Foundations](#3-electrophysiological--biophysical-foundations)
   - 3.1 [Working Memory 2.0: Laminar Push-Pull Rhythms (Superficial $\gamma$ vs. Deep Infragranular $\beta$)](#31-working-memory-20-laminar-push-pull-rhythms-superficial-gamma-vs-deep-infragranular-beta)
   - 3.2 [32-Slot $\theta$-$\gamma$ Phase-Amplitude Coupling with Downbeat Anchor Referencing](#32-32-slot-theta-gamma-phase-amplitude-coupling-with-downbeat-anchor-referencing)
   - 3.3 [89.5 Hz Cortical Ripples (Dickey et al., 2022) & Causal Directionality ($ci\text{PLV}$)](#33-895-hz-cortical-ripples-dickey-et-al-2022--causal-directionality-ciplv)
   - 3.4 [High-Dimensional Orthogonal Plasticity: Hebbian LTP + Anti-Hebbian LTD ($5\sigma$)](#34-high-dimensional-orthogonal-plasticity-hebbian-ltp--anti-hebbian-ltd-5sigma)
   - 3.5 [Continuous Drift-Diffusion Dynamics (DDM) & Synaptic Persistence](#35-continuous-drift-diffusion-dynamics-ddm--synaptic-persistence)
4. [Hardware Execution, Zero-CPU Guarantee & Maze Parity](#4-hardware-execution-zero-cpu-guarantee--maze-parity)
   - 4.1 [End-to-End CUDA Tensor Pipeline & Microsecond IPC](#41-end-to-end-cuda-tensor-pipeline--microsecond-ipc)
   - 4.2 [LSL Buffer Anti-Starvation & Lock-Free Asynchronous Decoupling](#42-lsl-buffer-anti-starvation--lock-free-asynchronous-decoupling)
   - 4.3 [Dual-Mode Display: Kinematic Vector Compass vs. Torus Phase Space](#43-dual-mode-display-kinematic-vector-compass-vs-torus-phase-space)
5. [Thousand Brains Project (`tbp.monty`) Integration](#5-thousand-brains-project-tbpmonty-integration)
   - 5.1 [Cortical Messaging Protocol (CMP) Packet Structure](#51-cortical-messaging-protocol-cmp-packet-structure)
   - 5.2 [Relational Compositionality & Reference Frame Transformations](#52-relational-compositionality--reference-frame-transformations)
6. [CLI Configuration, Topology Syntax & Controls](#6-cli-configuration-topology-syntax--controls)
7. [Comprehensive Scientific Bibliography & DOIs](#7-comprehensive-scientific-bibliography--dois)

---

## 1. Paradigm Architecture: Collective Stigmergy & Cortical Generative Active Inference

Traditional Brain-Computer Interfaces (BCIs) reduce high-dimensional cortical dynamics into low-dimensional mechanical effectors [24]. NeuroCanvas treats neocortex as an **endogenous generative simulation engine** [2, 10, 23].

```
                     COLLECTIVE MULTI-BRAIN OVER-HETERARCHY (STIGMERGY)
                     
    ┌──────────────────────┐                     ┌──────────────────────┐
    │  BRAIN SUBJECT 1     │                     │  BRAIN SUBJECT 2     │
    │  (e.g., Pilot @ FCz) │                     │  (e.g., Guide @ AFz) │
    │  • 32-Slot PAC Motor │                     │  • 32-Slot PAC Torus │
    │  • 4,096 L4 HTM Sheet│                     │  • 4,096 L4 HTM Sheet│
    │  • Action: [FORWARD] │                     │  • Concept: [CASTLE] │
    └──────────┬───────────┘                     └──────────┬───────────┘
               │                                            │
               │ Kinematic Vector (lx, ly, rx)             │ Semantic Trajectory (u, v, wm)
               ▼                                            ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │                OVER-BRAIN COLLECTIVE (TBT 2.0 / CMP)              │
    │   • Multi-Agent Markov Blankets (Zero Cross-Brain Averaging)      │
    │   • Resolves Compositional Heterarchy ($A \supset B$, $A \parallel B$)            │
    │   • Evaluates Causal Lead ($89.5\text{ Hz } ci\text{PLV}$) & Attractor Gating     │
    └─────────────────────────────────┬─────────────────────────────────┘
                                      │ Sensorimotor Warp (dx, dy) & Latent Code c ∈ R^768
                                      ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │              DIFFUSION MANIFOLD & RECURSIVE TREEMAP               │
    │   • Real-Time SD/LCM Pipeline with Manifold Affine Warping        │
    │   • Feigenbaum Period-Doubling Bifurcations ($\delta \approx 4.6692$)         │
    │   • Emergent Proportional Partitioning & Boundary Crises          │
    └─────────────────────────────────┬─────────────────────────────────┘
                                      │ Closed-Loop Sensory Feedback
                                      ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │                 ENVIRONMENTAL PERCEPTION (CLIP)                   │
    │   • Asynchronous Zero-Shot Semantic Density Estimation            │
    └──────────┬────────────────────────────────────────────┬───────────┘
               │                                            │
               └────────► Perceived by Brain 1              └────────► Perceived by Brain 2
```

In nature, multi-agent coordination operates via **Stigmergy** [56] and **Distributed Active Inference** [21, 42]: biological brains never average their raw potentials. Averaging potentials across separate nervous systems violates the **Markov Blanket** [21] and introduces destructive phase interference [42]. NeuroCanvas maintains completely isolated mathematical pipelines for each user, allowing cooperative and competitive dynamics to emerge purely through modifications of the shared generative canvas.

---

## 2. Functional Neuroanatomy of Prefrontal Cortical Modules

The system rigorously maps electrode coordinates to distinct functional cytoarchitectonic regions of the human prefrontal cortex (PFC), preserving their native neurophysiological roles:

```
                            ANATOMICAL MONTAGE (10-20 EEG)
                                      
                                      [ Fpz ]  <-- Cognitive Branching (BA 10)
                                      /     \
                Left PFC (Order) [ F3 ]     [ F4 ]  Right PFC (Chaos)
                                      \     /
                       Cognitive Torus [ AFz ]
                                          |
                         Motor Flight  [ FCz ]  <-- Supplementary Motor Area (SMA)
```

### 2.1 FCz (Supplementary Motor Area / Pre-SMA): Dedicated 4-Axis Kinematic Flight
* **Brodmann Area:** BA 6 / BA 8 boundary.
* **Neurocomputational Function:** Planning, initiating, and executing voluntary sensorimotor action sequences [6, 7].
* **System Implementation:** **`FCz` is the sole electrode channel authorized to drive physical translation and camera warping.**
  When `FCz` is assigned in the routing topology (`--users "User1:FCz=0"`), the 32-slot $\theta$-$\gamma$ phase trajectory is projected onto the physical dipole axes to yield real-time steering vectors:
  $$\mathbf{intent} = (lx, ly), \quad \text{yaw} = rx, \quad \text{temporal lookahead} = ry$$
  **If `FCz` is absent from the configuration (e.g., `--users "User1:AFz=0"`), physical camera motion is strictly held at zero ($dx = 0, dy = 0$).** The canvas remains stationary; cognitive noise does not induce false camera drift.

### 2.2 AFz (Rostromedial Prefrontal Cortex / rmPFC / dACC): Topological Torus Geometry ($T^2$)
* **Brodmann Area:** BA 9 / BA 32.
* **Neurocomputational Function:** Tracking cognitive space, monitoring internal schemas, and navigating conceptual relational geometries [45, 60].
* **System Implementation:** **`AFz` operates in non-translational conceptual coordinates.** It samples the endogenous $\theta$ carrier ($6.0\text{ Hz}$) and $\delta$ macro-frame ($1.5\text{ Hz}$) to anchor the state vector onto the **Janata Torus** ($T^2 = S^1 \times S^1$):
  $$u = \angle(\Phi_\theta) \pmod{2\pi}, \quad v = \angle(\Phi_\delta) \pmod{2\pi}$$
  In `AFz` mode, the polar radar displays **Torus Phase Space** rather than a flight compass. Concept selection, contrastive LTM learning ($5\sigma$), and Treemap area allocation are mediated through `AFz`.

### 2.3 F3 (Left dlPFC): Categorical Attractors, Symbolic Tokens & $\beta$-Order Gating
* **Brodmann Area:** Left BA 9 / BA 46.
* **Neurocomputational Function:** Hierarchical sequence grammar, categorical boundary maintenance, symbolic linguistic chunking, and analytical execution [11, 32].
* **System Implementation:** Evaluates left-hemispheric infragranular $\beta$-power ($15\text{--}30\text{ Hz}$). Elevated $\beta$ on `F3` enforces **$\beta$-Order Gating** ($R_{\beta,\text{order}}$), stabilizing the active conceptual attractor, suppressing chaotic drift, and triggering boundary crises to prune redundant sub-branches when order exceeds $0.80$.

### 2.4 F4 (Right dlPFC): Holistic Schemas, Visuospatial Coordinate Frames & Feigenbaum Chaos
* **Brodmann Area:** Right BA 9 / BA 46.
* **Neurocomputational Function:** Holistic scene analysis, coordinate spatial relations, novel associative discovery, and divergent thinking [46, 47, 65].
* **System Implementation:** Evaluates right-hemispheric $\beta$-desynchronization ($1.0 - R_{\beta,\text{chaos}}$). A decline in right-frontal $\beta$ injects **Lyapunov Chaos Drive**, driving the accumulation of nonlinear energy in the active concept:
  $$\Delta E = (1.0 - R_{\beta,\text{chaos}}) \cdot \kappa_{\text{chaos}} - R_{\beta,\text{order}} \cdot \kappa_{\text{order}}$$
  When energy breaches the scaled Feigenbaum threshold ($E > \frac{0.70}{\delta^{\text{depth}}}$), it forces a **period-doubling bifurcation**, spawning specialized daughter concepts in the generative model.

### 2.5 Fpz (Frontopolar Cortex / BA 10): Cognitive Branching & Counterfactual Tracking
* **Brodmann Area:** Lateral and rostral BA 10.
* **Neurocomputational Function:** Evaluating alternative courses of action while maintaining the primary behavioral goal (cognitive branching) [28, 29, 33].
* **System Implementation:** Evaluates the sub-granular gating ratio $g(t)$. When alternative evidence accumulates past the critical threshold ($S > 1.35$), `Fpz` executes `trigger_fpz_cognitive_branch` or voluntary sub-child integration (`attempt_heterarchical_integration`), shifting the active Markov hierarchy without resetting baseline memory banks.

---

## 3. Electrophysiological & Biophysical Foundations

```
                            CORTICAL COLUMN MICROCIRCUIT
                                   
   L2/3 Superficial:  [ Gamma Bursts (30-100 Hz) ] <===> Feedforward Sensory & Concept Tokens
                             ▲                 ▲
                             │ (Gating Factor) │ (Dendritic Calcium C(t))
                             ▼                 ▼
   L4 Granular:       [ 4,096-Column HTM Sheet ] <===> Sparse Distributed Representations (k=80)
                             ▲                 ▲
                             │                 │ (3-Factor Synaptic Plasticity)
                             ▼                 ▼
   L5/6 Infragranular:[ Beta Rhythms (15-30 Hz) ] <===> Top-Down Executive Inhibitory Brake
```

### 3.1 Working Memory 2.0: Laminar Push-Pull Rhythms (Superficial $\gamma$ vs. Deep Infragranular $\beta$)
NeuroCanvas implements the laminar oscillatory control architecture validated by Miller, Lundqvist, and Bastos [2, 4]:
* **Superficial Layers (L2/3):** Pyramidal-Interneuron Gamma (PING) oscillations ($30\text{--}100\text{ Hz}$) encode transient, sparse bursts of active memoranda [5]. Spiking is bursty, minimizing metabolic consumption and preventing catastrophic attractor collapse.
* **Deep Layers (L5/6):** Synchronized infragranular $\beta$ oscillations ($15\text{--}30\text{ Hz}$) act as an **inhibitory executive brake**. When deep $\beta$ couples to superficial layers, it terminates gamma bursts and clears active memory contents.
* **Biophysical Gating Equation:**
  $$g(t) = \frac{R_\gamma(t)}{R_\gamma(t) + \kappa R_\beta(t) + \epsilon}$$
  A drop in deep-layer beta disinhibits L2/3 recurrent microcircuits, opening the gate for novel concept entry.

### 3.2 32-Slot $\theta$-$\gamma$ Phase-Amplitude Coupling with Downbeat Anchor Referencing
Sensory representations, spatial navigation trajectories, and abstract conceptual sequences are time-division multiplexed into high-gamma subcycles ($70\text{--}100\text{ Hz}$) nested within an endogenous $\theta$ carrier ($4\text{--}8\text{ Hz}$) [6, 20]:
* Each single theta cycle is discretized into **32 temporal phase bins** $\theta_k \in [-\pi, \pi)$.
* **Downbeat Slot 0 Anchor Referencing:** To eliminate zero-frequency baseline drift, cross-spectral density matrices across all 120 electrode pairs are referenced to the initial phase anchor (Slot 0):
  $$\mathbf{\Psi}_{k} = \Im \left( \mathbf{\Phi}_k \odot \mathbf{\Phi}_0^* \right) \in \mathbb{R}^{120}, \quad k \in \{0, \dots, 31\}$$
  Because Slot 0 referenced to itself identically satisfies $\Im(\mathbf{\Phi}_0 \odot \mathbf{\Phi}_0^*) \equiv 0$, the initial downbeat functions as an invariant mathematical origin.

### 3.3 89.5 Hz Cortical Ripples (Dickey et al., 2022) & Causal Directionality ($ci\text{PLV}$)
During memory recall and cognitive integration, human neocortex exhibits widespread $\sim 70\text{ ms}$, $89.5\text{ Hz}$ ripples that phase-synchronize across long distances [15].
NeuroCanvas isolates the **Volume-Conduction-Free Corrected Imaginary Phase-Locking Value ($ci\text{PLV}$)** across all 120 electrode pairs [12]:

$$ci\text{PLV}_{i, j} = \frac{\frac{1}{T} \sum_{t=1}^T \Im \left( z_i(t) z_j^*(t) \right)}{\sqrt{1 - \left( \frac{1}{T} \sum_{t=1}^T \Re \left( z_i(t) z_j^*(t) \right) \right)^2}}$$

* **Singular Value Decomposition (SVD) Rank Depth:** The centered ripple matrix is analyzed via SVD. The normalized singular value spectrum ($S_1, \dots, S_4$) extracts the effective intrinsic dimensionality of the prefrontal manifold.
* **Causal Order Metrics:**
  * $\text{Lead} > +0.03 \implies$ Node $A$ leads Node $B$ (Super-ordinate Parent Container: $A \supset B$).
  * $\text{Lead} < -0.03 \implies$ Node $B$ leads Node $A$ (Sub-ordinate Child Component: $B \supset A$).
  * $|\text{Lead}| \le 0.03 \implies$ Synchronous Co-occurrence (Peer Coexistence: $A \parallel B$).

### 3.4 High-Dimensional Orthogonal Plasticity: Hebbian LTP + Anti-Hebbian LTD ($5\sigma$)
To avoid catastrophic collinear weight collapse ($W_0 \approx W_1$), each `BrainSubject` maintains a **4,096-column CUDA Cortical Macrocolumn Sheet** with $1.95\%$ SDR sparsity ($k = 80$ active columns):
1. **Hebbian Long-Term Potentiation (LTP):**
   $$\mathbf{W}_{\text{target}} \leftarrow \text{clamp}\left( \mathbf{W}_{\text{target}} + \eta \cdot \mathbf{C}(t), 0.01, 1.0 \right)$$
2. **Anti-Hebbian Long-Term Depression (LTD):**
   $$\mathbf{W}_{\text{competitor}} \leftarrow \text{clamp}\left( \mathbf{W}_{\text{competitor}} - \eta \cdot \lambda_{\text{LTD}} \cdot \mathbf{C}(t), 0.01, 1.0 \right)$$
3. **$5\sigma$ Statistical Criterion ($d' \ge 4.75$, $p < 3 \cdot 10^{-7}$):**
   The system continuously calculates Fisher’s discriminant ratio across all active memory classes:
   $$d' = \frac{|\mu_0 - \mu_1|}{\sqrt{\frac{1}{2}(\sigma_0^2 + \sigma_1^2)}}$$
   **The engine refuses to unlock free inference until $d' \ge \text{threshold}$ (default: $3.0\sigma\text{--}3.5\sigma$, up to $5.0\sigma$).**

### 3.5 Continuous Drift-Diffusion Dynamics (DDM) & Synaptic Persistence
Following Gold & Shadlen [51] and Peixoto et al. [36], decision variables accumulate continuously without step-function artifacts:
* **Analog Projection:** $\rho(t) = \frac{\vec{f} \cdot \vec{f}_{\text{target}}}{\|\vec{f}\|_2 \|\vec{f}_{\text{target}}\|_2}$.
* **Synaptic Persistence Accumulation:**
  $$\text{Persistence}_t = \text{Persistence}_{t-1} \cdot 0.94 + 0.06 \cdot \text{Alignment} \cdot \tanh(\|\vec{f}\|_2 \cdot 2.0)$$
* **Active Boost Acceleration:** $\text{Boost} = 1.0 + 4.0 \cdot \text{Persistence}$, scaling velocity up to $5\times$ during sustained mental focus.

---

## 4. Hardware Execution, Zero-CPU Guarantee & Maze Parity

### 4.1 End-to-End CUDA Tensor Pipeline & Microsecond IPC
The runtime guarantees strict synchronization with `neuro_monty_bci_maze.py`:
* **Zero CPU Recomputation:** The 32-slot PAC field $\mathbf{\Psi}_{32 \times 120}$, Cartesian trajectory $\mathbf{P}_{32 \times 2}$, and kinematic state variables (`lx`, `ly`, `rx`, `ry`) are evaluated strictly within PyTorch CUDA tensors in `GPU_Daemon_Process`.
* **Deprecation of Legacy CPU Loops:** The `@property def gamepad_axes` bypasses all legacy host-side Python loops, returning the pre-packaged GPU tensor directly.

### 4.2 LSL Buffer Anti-Starvation & Lock-Free Asynchronous Decoupling
To eliminate frame stutter and maintain exact theta-rate sampling ($60\text{ FPS}$):
* **Buffer Purging (`max_samples=8192`):** `inlets[i].pull_chunk` pulls all accumulated samples, slicing the trailing `[-BUF_SIZE:]` to guarantee the processing window remains locked to real-time regardless of background load.
* **CLIP Decoupling:** The Vision-Language CLIP teacher evaluates frames in an isolated background thread at $5\text{ Hz}$, freeing GPU capacity for real-time EEG processing.
* **SVD & Treemap Throttling:** SVD decomposition and recursive Treemap box scaling evaluate every 6 frames ($10\text{ Hz}$), preventing CUDA stream synchronization stalls.

### 4.3 Dual-Mode Display: Kinematic Vector Compass vs. Torus Phase Space

| Screen Indicator | Motor Navigation Mode (`FCz`) | Semantic Torus Mode (`AFz`) |
| :--- | :--- | :--- |
| **Yellow Vector** | Target Direction ($\vec{f}_{\text{target}}$, Screen $N/S/E/W$) | Target Concept Position ($\vec{u}_{\text{target}}$ on Torus) |
| **Cyan Vector** | Real Human Motor Intent ($\vec{f}_{\text{human}}$, Screen $N/S/E/W$) | Current Torus Coordinate ($\cos u, \sin v$) |
| **Pink Vector** | Monty Decoded Prediction ($\vec{f}_{\text{monty}}$, Screen $N/S/E/W$) | Decoded Concept Attractor ($\vec{u}_{\text{decoded}}$) |
| **Camera Response** | Active Optical Flow (Zoom In/Out, Lateral Pan) | Fixed Camera / Static Manifold Rendering |
| **Treemap Display** | Kinematic Axes (`ВПЕРЕД`, `НАЗАД`, `ВПРАВО`, `ВЛЕВО`) | Conceptual Classes (`ГОРА`, `ДЖУНГЛИ`, `ЗАМОК`, `ОКЕАН`...) |

---

## 5. Thousand Brains Project (`tbp.monty`) Integration

### 5.1 Cortical Messaging Protocol (CMP) Packet Structure
At each cognitive cycle, the executive state is packaged into a standard `tbp.monty.cmp.Message` instance:

```python
Message(
    location=pilot.x, pilot.y, 0.0,                    # Allocentric 3D spatial location
    morphological_features={
        "pose_vectors": rot_matrix,                   # SO(3) Frenet frame orientation
        "pose_fully_defined": bool(smooth_depth >= 1.8),# True if SVD rank >= 1.8
        "on_object": 1.0                              # Manifold engagement marker
    },
    non_morphological_features={
        "object_id": int(leader_idx),                 # Active categorical attractor ID
        "torus_u": float(node_afz.torus_u),           # Theta coordinate u on Janata Torus
        "torus_v": float(node_afz.torus_v),           # Delta coordinate v on Janata Torus
        "beta_order": float(node_f3.beta_power)       # Infragranular inhibitory state
    },
    confidence=float(np.clip(confidence / 100.0, 0.0, 1.0)),
    pass_message=True,
    sender_id=subject_id,
    sender_type="SM",
    process_features_in_lm=True
)
```

### 5.2 Relational Compositionality & Reference Frame Transformations
Following TBT 2.0 [1], thalamocortical loops compute the relative rotation matrix $\Delta \mathbf{R} \in SO(3)$ between parent and child reference frames from Torus phase offsets:

$$\Delta u = u_{\text{child}} - u_{\text{parent}}, \quad \Delta v = v_{\text{child}} - v_{\text{parent}}$$

$$\Delta \mathbf{p}_{SO(3)} = \begin{bmatrix} \sin(\Delta u) \\ \cos(\Delta v) \\ \sin(\Delta u + \Delta v) \end{bmatrix}$$

---

## 6. CLI Configuration, Topology Syntax & Controls

### Command-Line Arguments
```bash
# Mode 1: Pure 4-Axis Kinematic Flight (Identical to Maze Engine)
python neuro_prefrontal_heterarchy_live.py --users "User1:FCz=0" --calib-mode motion

# Mode 2: Full Prefrontal Cognitive Heterarchy (AFz Torus + F3 Order + F4 Chaos)
python neuro_prefrontal_heterarchy_live.py --users "User1:AFz=0,F3=1,F4=2" --concepts 4

# Mode 3: Combined Pilot & Navigator Multi-Region Configuration
python neuro_prefrontal_heterarchy_live.py --users "User1:FCz=0,AFz=1,F3=2,F4=3" --concepts 4

# Mode 4: Multi-Subject Cooperative/Competitive Brain Space
python neuro_prefrontal_heterarchy_live.py --users "Pilot:FCz=0; Guide:AFz=1; Critic:F3=2,F4=3"
```

| Argument | Default | Type | Description |
| :--- | :--- | :--- | :--- |
| `--users` | `Dmitry:FCz=0`| `str` | Prefrontal routing string (`Subject:Region=DevIdx,...`). |
| `--calib-mode` | `auto` | `str` | Calibration mode: `auto` (auto-detect by region), `motion` (FCz), `semantic` (AFz). |
| `--concepts` | `1` | `int` | Number of semantic categories ($2\text{--}16$). |
| `--calib-sigma` | `3.5` | `float`| Statistical separation threshold $d'$ for unlocking inference ($3.0\sigma\text{--}5.0\sigma$). |
| `--calib-seconds` | `6.0` | `float`| Duration in seconds per guided calibration block. |
| `--calib-cycles` | `3` | `int` | Minimum alternation cycles before validation. |
| `--sensitivity` | `0.05` | `float`| Pilot response sensitivity (`moveSensitivity` from web). |
| `--max-speed` | `9.0` | `float`| Maximum allowable kinematic velocity. |
| `--intent-gain` | `1.5` | `float`| Raw motor intention multiplier. |
| `--chaos` | `False` | `flag` | Enables open-world Feigenbaum bifurcation cascades on launch. |
| `--sensory-sub` | `False` | `flag` | Enables Full-Duplex inter-brain concept broadcasting (Key `M`). |
| `--gamma-100` | `False` | `flag` | Extends spectral integration ceiling from $65\text{ Hz}$ to $100\text{ Hz}$. |

### Keybindings Reference
| Key | Action | Functional Description |
| :--- | :--- | :--- |
| `SPACE` | **Toggle Autopilot / Chaos** | Toggles between manual BCI intent and autonomous predictive steering. |
| `C` | **Toggle Chaos Engine** | Enables/disables live Feigenbaum period-doubling bifurcations ($\delta \approx 4.6692$). |
| `M` | **Sensory Substitution** | Alternates between isolated self-feedback and full-duplex swarm synchronization. |
| `R` | **Recalibrate** | Erases cached session parameters and restarts guided calibration. |
| `↑` / `↓` | **Longitudinal Drive** | Manual override for forward/backward translation. |
| `←` / `→` | **Lateral Strafe** | Manual override for left/right translation. |
| `.` / `,` | **Yaw Rotation** | Manual override for angular torque. |
| `ESC` | **Safe Shutdown** | Safely parks GPU daemons and releases POSIX shared memory buffers. |

---

## 7. Comprehensive Scientific Bibliography & DOIs

1. **Hawkins, J., Leadholm, N., & Clay, V. (2025/2026).** The Thousand Brains Theory 2.0: An Extension for the Long-Range Connections of the Neocortical Heterarchy. *arXiv preprint*, [arXiv:2507.05888](https://arxiv.org/abs/2507.05888).
2. **Miller, E. K., Lundqvist, M., & Bastos, A. M. (2018).** Working Memory 2.0. *Neuron*, 100(2), 463–475. [DOI: 10.1016/j.neuron.2018.09.023](https://doi.org/10.1016/j.neuron.2018.09.023)
3. **Hawkins, J., Ahmad, S., & Cui, Y. (2017).** A theory of how columns in the neocortex enable learning the structure of the world. *Frontiers in Neural Circuits*, 11, 81. [DOI: 10.3389/fncir.2017.00081](https://doi.org/10.3389/fncir.2017.00081)
4. **Bastos, A. M., Loonis, R., Kornblith, S., Lundqvist, M., & Miller, E. K. (2018).** Laminar recordings in frontal cortex suggest distinct layers for maintenance and control of working memory. *PNAS*, 115(5), 1117–1122. [DOI: 10.1073/pnas.1714522115](https://doi.org/10.1073/pnas.1714522115)
5. **Lundqvist, M., Herman, P., Warden, M. R., Brincat, S. L., & Miller, E. K. (2018).** Gamma and beta bursts during working memory readout suggest roles in its volitional control. *Nature Communications*, 9(1), 394. [DOI: 10.1038/s41467-017-02791-8](https://doi.org/10.1038/s41467-017-02791-8)
6. **Lisman, J. E., & Jensen, O. (2013).** The theta-gamma neural code. *Neuron*, 77(6), 1002–1016. [DOI: 10.1016/j.neuron.2013.03.007](https://doi.org/10.1016/j.neuron.2013.03.007)
7. **Churchland, M. M., et al. (2012).** Neural population dynamics during reaching. *Nature*, 487(7405), 51–56. [DOI: 10.1038/nature11129](https://doi.org/10.1038/nature11129)
8. **Tort, A. B., Komorowski, R., Eichenbaum, H., & Kopell, N. (2010).** Measuring phase-amplitude coupling between neuronal oscillations. *Journal of Neurophysiology*, 104(2), 1195–1210. [DOI: 10.1152/jn.00106.2010](https://doi.org/10.1152/jn.00106.2010)
9. **Fusi, S., Miller, E. K., & Rigotti, M. (2016).** Why neurons mix: high dimensionality for higher cognition. *Current Opinion in Neurobiology*, 37, 66–74. [DOI: 10.1016/j.conb.2016.01.010](https://doi.org/10.1016/j.conb.2016.01.010)
10. **Badre, D., & Nee, D. E. (2018).** Frontal cortex and the hierarchical control of behavior. *Trends in Cognitive Sciences*, 22(2), 170–188. [DOI: 10.1016/j.tics.2017.11.005](https://doi.org/10.1016/j.tics.2017.11.005)
11. **Ding, N., Melloni, L., Zhang, H., Tian, X., & Poeppel, D. (2016).** Cortical tracking of hierarchical linguistic structures in connected speech. *Nature Neuroscience*, 19(1), 158–164. [DOI: 10.1038/nn.4186](https://doi.org/10.1038/nn.4186)
12. **Bruña, R., Maestú, F., & Pereda, E. (2018).** Phase Locking Value revisited: teaching new tricks to an old dog. *Journal of Neural Engineering*, 15(5), 056011. [DOI: 10.1088/1741-2552/aacfe4](https://doi.org/10.1088/1741-2552/aacfe4)
13. **Nolte, G., et al. (2004).** Identifying true brain interaction from EEG data using the imaginary part of coherency. *Clinical Neurophysiology*, 115(10), 2292–2307. [DOI: 10.1016/j.clinph.2004.04.029](https://doi.org/10.1016/j.clinph.2004.04.029)
14. **Kikumoto, A., & Mayr, U. (2020).** Decoding hierarchical control of sequential behavior in oscillatory EEG activity. *eLife*, 9, e53589. [DOI: 10.7554/eLife.53589](https://doi.org/10.7554/eLife.53589)
15. **Dickey, C. W., et al. (2022).** Widespread ripples synchronize human cortical activity during sleep, waking, and memory recall. *PNAS*, 119(28), e2107797119. [DOI: 10.1073/pnas.2107797119](https://doi.org/10.1073/pnas.2107797119)
16. **Arnulfo, G., et al. (2020).** Long-range phase synchronization of high-frequency oscillations in human cortex. *Nature Communications*, 11(1), 5363. [DOI: 10.1038/s41467-020-18975-8](https://doi.org/10.1038/s41467-020-18975-8)
17. **Hawkins, J., Lewis, M., Klukas, M., Purdy, S., & Ahmad, S. (2019).** A framework for intelligence and cortical function based on grid cells. *Frontiers in Neural Circuits*, 13, 86. [DOI: 10.3389/fncir.2019.00086](https://doi.org/10.3389/fncir.2019.00086)
18. **Mountcastle, V. B. (1997).** The columnar organization of the neocortex. *Brain*, 120(4), 701–722. [DOI: 10.1093/brain/120.4.701](https://doi.org/10.1093/brain/120.4.701)
19. **Besio, W. G., Koka, K., & Aakula, R. (2006).** Tri-polar concentric ring electrode development for Laplacian electroencephalography. *IEEE TBME*, 53(5), 926–933. [DOI: 10.1109/TBME.2006.873398](https://doi.org/10.1109/TBME.2005.863887)
20. **Lakatos, P., et al. (2005).** An oscillatory hierarchy controlling neuronal excitability and stimulus processing. *Journal of Neurophysiology*, 94(3), 1904–1911. [DOI: 10.1152/jn.00263.2005](https://doi.org/10.1152/jn.00263.2005)
21. **Friston, K. (2010).** The free-energy principle: a unified brain theory?. *Nature Reviews Neuroscience*, 11(2), 127–138. [DOI: 10.1038/nrn2787](https://doi.org/10.1038/nrn2787)
22. **Mante, V., Sussillo, D., Shenoy, K. V., & Newsome, W. T. (2013).** Context-dependent computation by recurrent dynamics in prefrontal cortex. *Nature*, 503(7474), 78–84. [DOI: 10.1038/nature12742](https://doi.org/10.1038/nature12742)
23. **Miller, E. K., & Cohen, J. D. (2001).** An integrative theory of prefrontal cortex function. *Annual Review of Neuroscience*, 24(1), 167–202. [DOI: 10.1146/annurev.neuro.24.1.167](https://doi.org/10.1146/annurev.neuro.24.1.167)
24. **Voloh, B., Valiante, T. A., & Womelsdorf, T. (2015).** Theta–gamma coordination between anterior cingulate and prefrontal cortex. *PNAS*, 112(27), 8457–8462. [DOI: 10.1073/pnas.1502092112](https://doi.org/10.1073/pnas.1502092112)
25. **Colgin, L. L., et al. (2009).** Frequency of gamma oscillations routes flow of information in the hippocampus. *Nature*, 462(7271), 353–357. [DOI: 10.1038/nature08573](https://doi.org/10.1038/nature08573)
26. **Bieri, K. W., Bobbitt, K. N., & Colgin, L. L. (2014).** Slow and fast gamma rhythms coordinate different spatial coding modes. *Neuron*, 82(3), 670–681. [DOI: 10.1016/j.neuron.2014.03.013](https://doi.org/10.1016/j.neuron.2014.03.013)
27. **Norman, Y., et al. (2021).** Post-activation pause: An anatomical signature of memory recall in human cortex. *Science*, 373(6560), eabg7595. [DOI: 10.1126/science.abg7595](https://doi.org/10.1126/science.abg7595)
28. **Boorman, E. D., et al. (2009).** How green is the grass on the other side? Frontopolar cortex and evidence for alternatives. *Neuron*, 62(5), 733–743. [DOI: 10.1016/j.neuron.2009.05.014](https://doi.org/10.1016/j.neuron.2009.05.014)
29. **Koechlin, E., & Hyafil, A. (2007).** Anterior prefrontal function and the limits of human decision-making. *Science*, 318(5850), 594–598. [DOI: 10.1126/science.1142995](https://doi.org/10.1126/science.1142995)
30. **Alexander, W. H., & Brown, J. W. (2011).** Mediodorsal prefrontal cortex as a prediction error hub. *Nature Neuroscience*, 14(10), 1338–1344. [DOI: 10.1038/nn.2921](https://doi.org/10.1038/nn.2921)
31. **Mongillo, G., Barak, O., & Tsodyks, M. (2008).** Synaptic theory of working memory. *Science*, 319(5869), 1543–1546. [DOI: 10.1126/science.1150769](https://doi.org/10.1126/science.1150769)
32. **Badre, D., & D'Esposito, M. (2007).** FMRI evidence for a hierarchical organizing principle in PFC. *Nature Neuroscience*, 10(9), 1138–1144. [DOI: 10.1038/nn1953](https://doi.org/10.1038/nn1953)
33. **Tsujimoto, S., Genovesio, A., & Wise, S. P. (2010).** Frontopolar cortex: neuronal networks for decision-making. *Journal of Neuroscience*, 30(50), 16756–16759. [DOI: 10.1523/JNEUROSCI.6667-09.2010](https://doi.org/10.1523/JNEUROSCI.6667-09.2010)
34. **Shenhav, A., Botvinick, M. M., & Cohen, J. D. (2013).** The expected value of control. *Nature Neuroscience*, 16(7), 885–892. [DOI: 10.1038/nn.3423](https://doi.org/10.1038/nn.3423)
35. **Womelsdorf, T., et al. (2010).** Theta-activity in anterior cingulate cortex predicts task rules. *Journal of Neuroscience*, 30(38), 12694–12702. [DOI: 10.1523/JNEUROSCI.2861-10.2010](https://doi.org/10.1523/JNEUROSCI.2861-10.2010)
36. **Chen, J., Zhang, C., Hu, P., Min, B., & Wang, L. (2024).** Flexible control of sequence working memory in the macaque frontal cortex. *Neuron*, 112(20), 3502–3514. [DOI: 10.1016/j.neuron.2024.07.024](https://doi.org/10.1016/j.neuron.2024.07.024)
37. **Fan, Y., Wang, M., Ding, N., & Luo, H. (2024).** Two-dimensional neural geometry underpins hierarchical organization of sequence in human working memory. *Nature Human Behaviour*, 8, 2150–2163. [DOI: 10.1038/s41562-024-02047-8](https://doi.org/10.1038/s41562-024-02047-8)
38. **Siegel, M., Warden, M. R., & Miller, E. K. (2009).** Phase-dependent neuronal coding of objects. *PNAS*, 106(50), 21341–21346. [DOI: 10.1073/pnas.0908193106](https://doi.org/10.1073/pnas.0908193106)
39. **Buzsáki, G., & Tingley, D. (2018).** Space and time: the hippocampus as a sequence generator. *Trends in Cognitive Sciences*, 22(10), 853–869. [DOI: 10.1016/j.tics.2018.07.006](https://doi.org/10.1016/j.tics.2018.07.006)
40. **Takagi, Y., & Nishimoto, S. (2023).** High-resolution image reconstruction with latent diffusion models from brain activity. *Nature Communications*, 14(1), 1568. [DOI: 10.1038/s41467-023-36701-1](https://doi.org/10.1038/s41467-023-36701-1)
41. **Clark, A. (2008).** *Supersizing the Mind: Embodiment, Action, and Cognitive Extension.* Oxford University Press. [DOI: 10.1093/acprof:oso/9780195333213.001.0001](https://doi.org/10.1093/acprof:oso/9780195333213.001.0001)
42. **Dumas, G., et al. (2010).** Inter-brain synchronization during social interaction. *PLoS ONE*, 5(8), e12165. [DOI: 10.1371/journal.pone.0012165](https://doi.org/10.1371/journal.pone.0012165)
43. **Zhang, Y. (2026).** Recurrent Looped Transformer: Latent Reasoning with Unbounded Temporal Depth. *alphaXiv preprint*, [alphaxiv:2609.recurrent-looped-transformer](https://www.alphaxiv.org/abs/2609.recurrent-looped-transformer).
44. **Tonegawa, S., et al. (2015).** Memory engram cells have come of age. *Neuron*, 87(5), 918–931. [DOI: 10.1016/j.neuron.2015.08.002](https://doi.org/10.1016/j.neuron.2015.08.002)
45. **Janata, P., et al. (2002).** The cortical topography of tonal structures underlying Western music. *Science*, 298(5601), 2167–2170. [DOI: 10.1126/science.1076262](https://doi.org/10.1126/science.1076262)
46. **Fink, G. R., et al. (1996).** Where in the brain does visual attention select the forest and the trees? *Nature*, 382(6592), 626–628. [DOI: 10.1038/382626a0](https://doi.org/10.1038/382626a0)
47. **Iaccino, J. F. (2014).** *Left Brain-Right Brain Differences: Inquiries, Evidence, and New Approaches.* Psychology Press. [DOI: 10.4324/9781315806655](https://doi.org/10.4324/9781315806655)
48. **Ewald, A., et al. (2012).** Estimating true brain connectivity from EEG/MEG data invariant to linear transformations. *NeuroImage*, 60(1), 476–488. [DOI: 10.1016/j.neuroimage.2011.11.084](https://doi.org/10.1016/j.neuroimage.2011.11.084)
49. **Sun, J., et al. (2026).** VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model. *arXiv preprint*, [arXiv:2602.10098](https://arxiv.org/abs/2602.10098).
50. **Assran, M., et al. (2025).** V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. *arXiv preprint*, [arXiv:2506.09985](https://arxiv.org/abs/2506.09985).
51. **Gold, J. I., & Shadlen, M. N. (2007).** The neural basis of decision making. *Annual Review of Neuroscience*, 30(1), 535–574. [DOI: 10.1146/annurev.neuro.29.051605.113038](https://doi.org/10.1146/annurev.neuro.29.051605.113038)
52. **Muller, L., et al. (2018).** Cortical travelling waves: mechanisms and computational principles. *Nature Reviews Neuroscience*, 19(5), 255–268. [DOI: 10.1038/nrn.2018.20](https://doi.org/10.1038/nrn.2018.20)
53. **Grossberg, S. (1980).** How does a brain build a cognitive code? *Psychological Review*, 87(1), 1–51. [DOI: 10.1037/0033-295X.87.1.1](https://doi.org/10.1037/0033-295X.87.1.1)
54. **Torgerson, W. S. (1952).** Multidimensional scaling: I. Theory and method. *Psychometrika*, 17(4), 401–419. [DOI: 10.1007/BF02288916](https://doi.org/10.1007/BF02288916)
55. **Kriegeskorte, N., et al. (2008).** Representational similarity analysis. *Frontiers in Systems Neuroscience*, 2, 4. [DOI: 10.3389/neuro.06.004.2008](https://doi.org/10.3389/neuro.06.004.2008)
56. **Grassé, P. P. (1959).** La reconstruction du nid et les coordinations interindividuelles... la théorie de la stigmergie. *Insectes Sociaux*, 6(1), 41–80. [DOI: 10.1007/BF02223791](https://doi.org/10.1007/BF02223791)
57. **Buzsáki, G. (2015).** Hippocampal sharp wave-ripple: A cognitive biomarker for episodic memory. *Hippocampus*, 25(10), 1073–1188. [DOI: 10.1002/hipo.22488](https://doi.org/10.1002/hipo.22488)
58. **Nunez, P. L., & Srinivasan, R. (2006).** *Electric fields of the brain: the neurophysics of EEG*. Oxford University Press. [DOI: 10.1093/acprof:oso/9780195050387.001.0001](https://doi.org/10.1093/acprof:oso/9780195050387.001.0001)
59. **Breakspear, M., Heitmann, S., & Daffertshofer, A. (2010).** Generative models of cortical oscillations: neurobiological implications of the Kuramoto model. *Frontiers in Human Neuroscience*, 4, 190. [DOI: 10.3389/fnhum.2010.00190](https://doi.org/10.3389/fnhum.2010.00190)
60. **Constantinescu, A. O., O'Reilly, J. X., & Behrens, T. E. (2016).** Organizing conceptual knowledge in humans with a gridlike code. *Science*, 352(6292), 1464–1468. [DOI: 10.1126/science.aaf0941](https://doi.org/10.1126/science.aaf0941)
61. **Freeman, W. J. (1987).** Simulation of chaotic EEG patterns with a dynamic model of the olfactory system. *Biological Cybernetics*, 56(2-3), 139–150. [DOI: 10.1007/BF00317988](https://doi.org/10.1007/BF00317988)
62. **Chialvo, D. R. (2010).** Emergent complex neural dynamics. *Nature Physics*, 6(10), 744–750. [DOI: 10.1038/nphys1803](https://doi.org/10.1038/nphys1803)
63. **Grebogi, C., Ott, E., & Yorke, J. A. (1983).** Crises, sudden changes in chaotic attractors, and transient chaos. *Physica D: Nonlinear Phenomena*, 7(1-3), 181–200. [DOI: 10.1016/0167-2789(83)90126-4](https://doi.org/10.1016/0167-2789(83)90126-4)
64. **Chklovskii, D. B., Mel, B. W., & Svoboda, K. (2004).** Cortical rewiring and information storage. *Nature*, 431(7010), 782–788. [DOI: 10.1038/nature03012](https://doi.org/10.1038/nature03012)
65. **Kosslyn, S. M., et al. (1992).** Categorical versus coordinate spatial relations: computational analyses and computer-simulated hemispheric asymmetries. *J Exp Psychol Hum Percept Perform*, 18(2), 562–577. [DOI: 10.1037//0096-1523.18.2.562](https://doi.org/10.1037//0096-1523.18.2.562)
66. **Corbetta, M., & Shulman, G. L. (2002).** Control of goal-directed and stimulus-driven attention in the brain. *Nature Reviews Neuroscience*, 3(3), 201–215. [DOI: 10.1038/nrn755](https://doi.org/10.1038/nrn755)
67. **Xie, Y., Hu, P., Li, J., Chen, J., Song, W., Wang, X. J., Yang, T., Dehaene, S., Tang, S., Min, B., & Wang, L. (2022).** Geometry of sequence working memory in macaque prefrontal cortex. *Science*, 375(6581), 632–639. [DOI: 10.1126/science.abm0204](https://doi.org/10.1126/science.abm0204)
68. **McCulloch, W. S. (1945).** A heterarchy of values determined by the topology of nervous nets. *The Bulletin of Mathematical Biophysics*, 7(2), 89–93. [DOI: 10.1007/BF02478357](https://doi.org/10.1007/BF02478357)

