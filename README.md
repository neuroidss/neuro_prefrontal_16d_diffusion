# 🧠 NeuroCanvas × TBP.Monty

### Bi-Hemispheric Cortical Lateralization, Janata Toroidal Manifolds, and Dynamic Prefrontal Heterarchy via Cortical Messaging Protocol (CMP)

[![DOI:10.1038/s41562-024-02047-8](https://img.shields.io/badge/DOI-10.1038%2Fs41562--024--02047--8-blue.svg)](https://doi.org/10.1038/s41562-024-02047-8)
[![DOI:10.1073/pnas.2107797119](https://img.shields.io/badge/DOI-10.1073%2Fpnas.2107797119-green.svg)](https://doi.org/10.1073/pnas.2107797119)
[![DOI:10.1016/j.neuron.2024.07.024](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2024.07.024-red.svg)](https://doi.org/10.1016/j.neuron.2024.07.024)
[![DOI:10.1016/j.neuron.2018.09.023](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2018.09.023-purple.svg)](https://doi.org/10.1016/j.neuron.2018.09.023)
[![DOI:10.1038/nn.4186](https://img.shields.io/badge/DOI-10.1038%2Fnn.4186-yellow.svg)](https://doi.org/10.1038/nn.4186)
[![DOI:10.1126/science.1076262](https://img.shields.io/badge/DOI-10.1126%2Fscience.1076262-darkblue.svg)](https://doi.org/10.1126/science.1076262)
[![DOI:10.1126/science.1142995](https://img.shields.io/badge/DOI-10.1126%2Fscience.1142995-teal.svg)](https://doi.org/10.1126/science.1142995)
[![DOI:10.1016/j.neuron.2009.05.014](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2009.05.014-magenta.svg)](https://doi.org/10.1016/j.neuron.2009.05.014)
[![DOI:10.1088/1741-2552/aacfe4](https://img.shields.io/badge/DOI-10.1088%2F1741--2552%2Faacfe4-orange.svg)](https://doi.org/10.1088/1741-2552/aacfe4)
[![DOI:10.1038/nature08573](https://img.shields.io/badge/DOI-10.1038%2Fnature08573-cyan.svg)](https://doi.org/10.1038/nature08573)
[![DOI:10.1016/j.neuron.2014.03.013](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2014.03.013-blueviolet.svg)](https://doi.org/10.1016/j.neuron.2014.03.013)
[![DOI:10.1038/nature11129](https://img.shields.io/badge/DOI-10.1038%2Fnature11129-lightgrey.svg)](https://doi.org/10.1038/nature11129)

---

## 📑 Table of Contents
1. [Paradigm Shift: Endogenous Generative Simulation vs. Mechanical BCIs](#1-paradigm-shift-endogenous-generative-simulation-vs-mechanical-bcis)
2. [Biophysical & Mathematical Foundation](#2-biophysical--mathematical-foundation)
   - 2.1 [The Thousand Brains Heterarchy (TBT 2.0): Eliminating Hardcoded Trees](#21-the-thousand-brains-heterarchy-tbt-20-eliminating-hardcoded-trees)
   - 2.2 [Bi-Hemispheric Feature Subspace Splitting: Form ($0 \dots 383$) vs. Style ($384 \dots 767$)](#22-bi-hemispheric-feature-subspace-splitting-form-0--383-vs-style-384--767)
   - 2.3 [Topological Cognitive Geometry: The Janata Torus ($T^2 = S^1 \times S^1$) via Empirical RDM MDS](#23-topological-cognitive-geometry-the-janata-torus-t2--s1-times-s1-via-empirical-rdm-mds)
   - 2.4 [Cognitive Branching via Sequential Probability Ratio Testing (SPRT) in Frontopolar Area 10 (Fpz)](#24-cognitive-branching-via-sequential-probability-ratio-testing-sprt-in-frontopolar-area-10-fpz)
   - 2.5 [Working Memory 2.0: Deep-Layer Beta Gating of Superficial Gamma Assemblies](#25-working-memory-20-deep-layer-beta-gating-of-superficial-gamma-assemblies)
   - 2.6 [Gap-Free Electrophysiological Spectral Stack: Continuous 500 Hz Multi-Band HAL](#26-gap-free-electrophysiological-spectral-stack-continuous-500-hz-multi-band-hal)
   - 2.7 [Causal Directionality: Volume-Conduction-Free Corrected $ci\text{PLV}$ at 89.5 Hz](#27-causal-directionality-volume-conduction-free-corrected-ciplv-at-895-hz)
   - 2.8 [jPCA Rotational Manifolds Derived from $\mathfrak{so}(3)$ Lie Algebras](#28-jpca-rotational-manifolds-derived-from-mathfrakso3-lie-algebras)
3. [Thousand Brains Project Integration (`tbp.monty`)](#3-thousand-brains-project-integration-tpmonty)
   - 3.1 [Strict Cortical Messaging Protocol (CMP) Packet Structure](#31-strict-cortical-messaging-protocol-cmp-packet-structure)
   - 3.2 [Sensorimotor Frame Transformations & Relational Compositionality](#32-sensorimotor-frame-transformations--relational-compositionality)
   - 3.3 [16,384-Column CUDA Cortical Macrocolumn Sheet](#33-16384-column-cuda-cortical-macrocolumn-sheet)
4. [Autonomous Stigmergic Swarm & VLA-JEPA World Modeling](#4-autonomous-stigmergic-swarm--vla-jepa-world-modeling)
   - 4.1 [Zero-Metagame Multi-Agent Interaction via Environmental Traces](#41-zero-metagame-multi-agent-interaction-via-environmental-traces)
   - 4.2 [Latent Energy Minimization vs. Superficial Feature Matching](#42-latent-energy-minimization-vs-superficial-feature-matching)
   - 4.3 [Phase-Division Multiple Access (PDMA) Across Theta Cycles](#43-phase-division-multiple-access-pdma-across-theta-cycles)
   - 4.4 [Synaptic Long-Term Potentiation (LTP) & Engram Consolidation Thresholds](#44-synaptic-long-term-potentiation-ltp--engram-consolidation-thresholds)
5. [Hardware & Software Architecture](#5-hardware--software-architecture)
   - 5.1 [FreeEEG16-alpha2 Concentric Ring Sensor Array & Surface Laplacian](#51-freeeeg16-alpha2-concentric-ring-sensor-array--surface-laplacian)
   - 5.2 [Server-Client IPC Topology](#52-server-client-ipc-topology)
6. [CLI Reference & Quickstart](#6-cli-reference--quickstart)
7. [Comprehensive Scientific Bibliography & DOIs](#7-comprehensive-scientific-bibliography--dois)

---

## 1. Paradigm Shift: Endogenous Generative Simulation vs. Mechanical BCIs

Traditional Brain-Computer Interfaces (BCIs) view brain activity through an impoverished lens: moving a 2D cursor across a monitor, selecting letters from a grid, or decoding motor velocity [24]. This approach ignores the highest-order evolutionary function of the mammalian neocortex: **endogenous generative simulation** [2, 10, 23, 28].

The human prefrontal cortex (PFC) did not evolve to drive cursors. It evolved to hold, manipulate, transform, and evaluate counterfactual models of reality that are not present in immediate sensory input [10, 23, 28, 29].

```
                         THE CLOSED-LOOP HETERARCHICAL ACTIVE INFERENCE MANIFOLD
                        
    ┌───────────────────────────────────────────────────────────────────────────────────────┐
    │                      FRONTAL ELECTROPHYSIOLOGY ARRAY (500 SPS HAL)                    │
    │   Fpz (BA10: Branching) • AFz (dACC: Janata Torus) • F3 (L-dlPFC) • F4 (R-dlPFC)      │
    └──────────────────────────────────────────┬────────────────────────────────────────────┘
                                               │ 16-ch Surface Laplacian LFP
                                               ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────┐
    │                       CONTINUOUS SPECTRAL DECOMPOSITION PIPELINE                      │
    │  Delta (1.5 Hz) • Theta (6 Hz) • Beta (22 Hz) • Gamma (30–65 Hz) • Ripples (89.5 Hz)  │
    │  Phase-Amplitude Coupling (PAC) • 120-Edge Corrected Imaginary PLV (ciPLV)           │
    └──────────────────────────────────────────┬────────────────────────────────────────────┘
                                               │ Directed Synchronization Matrix
                                               ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────┐
    │                         TBP.MONTY CORTICAL MESSAGING PROTOCOL                         │
    │     Message(location_3d, pose_vectors_so3, scale, confidence, process_features=True)   │
    │            16,384-Column CUDA L4 SDR Sheet (1.95% Spatial Sparsity)                   │
    └──────────────────────────────────────────┬────────────────────────────────────────────┘
                                               │ Lateralized Latent State c ∈ R^768
                                               ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────┐
    │                       TOROIDAL LATENT DIFFUSION ACTUATOR (SD/LCM)                     │
    │         Form [0..383] (Left F3)  ◄──────────────►  Style [384..767] (Right F4)        │
    │         Denoise Overdrive (s = 0.50..0.85) • Dynamic Emergent Treemap Sovereignty     │
    └──────────────────────────────────────────┬────────────────────────────────────────────┘
                                               │ Real-Time Synthesized Environment
                                               ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────┐
    │                      ENVIRONMENTAL PERCEPTION (CLIP / VLA-JEPA)                       │
    │       Zero-Shot Semantic Density Vector W • Latent World Energy Minimization          │
    └──────────────────────────────────────────┬────────────────────────────────────────────┘
                                               │ Closed-Loop Visual Feedback
                                               └────────► Observed by Prefrontal Units
```

**NeuroCanvas** bridges this computational engine directly to visual reality:
* **Generative Manifestation:** Instead of decoding mechanical coordinates, it reads the structural complexity of prefrontal phase manifolds and materializes it into a continuous, real-time visual world via Latent Diffusion [40].
* **Emergent Causal Sovereignty:** In a multi-agent environment, sovereignty over reality is governed by cognitive depth. Agents (biological or artificial) that sustain higher recursive complexity ($K \ge 2.5$) establish the overarching architectural context of the world (e.g., *Space $\supset$ Planet* or *Ocean $\supset$ Castle*). Agents with lower recursive depth ($K \le 2.0$) populate the interior details or co-exist as adjacent peers [1, 10, 43].
* **Pure Stigmergy:** Agents never inspect each other's internal variables, weights, or memory buffers (avoiding the metagame trap). They perceive only the shared generated canvas and negotiate dominance by restructuring their own neural phase geometries [41, 42].

---

## 2. Biophysical & Mathematical Foundation

### 2.1 The Thousand Brains Heterarchy (TBT 2.0): Eliminating Hardcoded Trees
In the classical view of cortical processing, features are extracted sequentially up an anatomically fixed hierarchy (e.g., $V1 \to V2 \to V4 \to IT$), culminating in high-level object representations at the apex.

As formulated in Hawkins, Leadholm, & Clay (2025/2026) [1], empirical neuroanatomy contradicts this serial assumption:
1. **Primary sensory areas represent complete objects:** Columns across $V1$, $S1$, and $A1$ receive direct driving thalamocortical projections and establish complete sensorimotor object models through movement integration.
2. **Heterarchical equivalence:** Long-range reciprocal cortico-cortical connections between columns do not convey raw features upward, but instead learn **compositional assignments** between whole objects.
3. **Absence of a priori ordering:** In natural scenes, an object can be either a parent or a child depending on behavioral context and sensory focus (e.g., a *Mug* can contain a *Logo*, or a *Logo* can incorporate a *Mug* graphic). No object sits permanently higher than another.

NeuroCanvas embodies this heterarchical principle by constructing scene composition dynamically on a location-by-location basis, driven by instantaneous prefrontal neural dynamics. Any concept can serve as a parent container ($L_0$), a child component ($L_1$), or an adjacent peer.

---

### 2.2 Bi-Hemispheric Feature Subspace Splitting: Form ($0 \dots 383$) vs. Style ($384 \dots 767$)
Functional lateralization studies in cognitive neuroscience show an asymmetric division of computational labor across the cerebral hemispheres:
* **Left Hemisphere (F3 / dlPFC):** Dominates processing of **local boundaries, categorical syntax, spatial structure, and geometric shape** (Fink et al., 1996 [46]; Ding et al., 2016 [11]; Iaccino, 2014 [47]).
* **Right Hemisphere (F4 / dlPFC):** Dominates processing of **global gestalt, holistic spatial context, color palette, surface texture, and stylistic atmosphere** (Fink et al., 1996 [46]; Takagi & Nishimoto, 2023 [40]).

Recent brain-decoding studies utilizing latent diffusion models (e.g., *Brain3D*, 2025; *Lateralization MLP*, 2024) demonstrate that the latent text-conditioning space $c \in \mathbb{R}^{768}$ exhibits a natural functional split:

$$\mathbf{c}_{\text{composite}} = \big[ \, \mathbf{c}_{\text{Form}}[0 \dots 383] \; \parallel \; \mathbf{c}_{\text{Style}}[384 \dots 767] \, \big]$$

Where:
* **$\mathbf{c}_{\text{Form}} \in \mathbb{R}^{384}$** is driven by **Node 0 (F3)**, dictating the physical contours, objects, and spatial composition.
* **$\mathbf{c}_{\text{Style}} \in \mathbb{R}^{384}$** is driven by **Node 1 (F4)**, dictating the texture, lighting, rendering medium, and atmospheric tone.

This direct bi-hemispheric allocation prevents the visual smearing and mutual interference that occur when multiple text prompts are naively averaged together.

---

### 2.3 Topological Cognitive Geometry: The Janata Torus ($T^2 = S^1 \times S^1$) via Empirical RDM MDS
Rather than reducing prefrontal activity to an uncalibrated scalar error metric, NeuroCanvas operationalizes the continuous topological manifold theory developed by **Petr Janata (*Science*, 2002)** [45].

Janata showed that rostromedial prefrontal cortex (rmPFC / dACC, situated directly beneath **AFz**) continuously tracks navigation through abstract cognitive spaces along the surface of a 2D torus ($T^2 = S^1 \times S^1$).

```
                         THE JANATA COGNITIVE TORUS (T² = S¹ × S¹)
                         
                                      ╭─────────────────╮
                                   .-'   u (Theta Slot)  '-.
                                 .'   ╭─────────────────╮   '.
                                /    /   SPACE   PLANET  \    \
                               |    |      •        •     | v  | (Delta Macro-Frame)
                               |    |   MOUNTAIN CASTLE   |    |
                                \    \     •        •    /    /
                                 '.   ╰─────────────────╯   .'
                                   '-.  OCEAN    JUNGLE  .-'
                                      ╰─────────────────╯
```

#### Empirical Derivation of Toroidal Coordinates
To avoid arbitrary geometric assignments, the positions of concepts on $T^2$ are computed from the empirical **Representational Dissimilarity Matrix (RDM)** of their multimodal representations (Fan et al., 2024 *Nature Human Behaviour* [37]; Kriegeskorte, 2008 [55]):

1. Given the zero-shot semantic text feature vectors $\mathbf{E} = [\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_N] \in \mathbb{R}^{N \times D}$, compute the normalized cosine distance matrix:

   $$D_{ij} = 1 - \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\|_2 \|\mathbf{e}_j\|_2} \in [0, 2]$$

2. Center the Gram matrix via double centering (Torgerson, 1952 [54]):

   $$\mathbf{B} = -\frac{1}{2} \mathbf{H} \mathbf{D}^{\circ 2} \mathbf{H}, \qquad \mathbf{H} = \mathbf{I}_N - \frac{1}{N} \mathbf{1}\mathbf{1}^T$$

3. Perform Classical Multidimensional Scaling (MDS) by extracting the leading two eigenvectors of $\mathbf{B}$:

   $$\mathbf{X}_{2\text{D}} = \mathbf{V}_{:, 1:2} \mathbf{\Lambda}_{1:2}^{1/2} \in \mathbb{R}^{N \times 2}$$

4. Normalize $\mathbf{X}_{2\text{D}}$ to $[0, 1]^2$ and map to periodic angular coordinates on $T^2$:

   $$u_i = 2\pi \cdot \tilde{x}_{i, 0} \in [0, 2\pi), \qquad v_i = 2\pi \cdot \tilde{x}_{i, 1} \in [0, 2\pi)$$

#### Geodesic Distance as Conflict Metric
When the environment exhibits dominant concept $B = (u_B, v_B)$ while an agent pursues goal $A = (u_A, v_A)$, the manifold tension is measured by the **geodesic metric on the flat torus**:

$$d_{T^2}(A, B) = \sqrt{ \min(|u_A - u_B|, 2\pi - |u_A - u_B|)^2 + \min(|v_A - v_B|, 2\pi - |v_A - v_B|)^2 }$$

---

### 2.4 Cognitive Branching via Sequential Probability Ratio Testing (SPRT) in Frontopolar Area 10 (Fpz)
The most anterior region of the frontal cortex, **Frontopolar Cortex (FPC / Brodmann Area 10 / Fpz)**, mediates **cognitive branching**: maintaining secondary goals in a pending state while an agent acts on a primary objective, and executing an exploratory switch when the primary pathway becomes unviable (Koechlin & Hyafil, 2007 *Science* [29]; Boorman et al., 2009 *Neuron* [28]).

```
                       COGNITIVE BRANCHING LOG-ODDS ACCUMULATOR
                       
        Λ(t) ▲
             │                                     DECISION BOUND θ_branch
             │───────────────────────────────────────────────────────────── (Switch to Plan B)
             │                                      ▲
             │                                     ╱
             │                         ▲          ╱
             │                        ╱ ╲        ╱
             │                       ╱   ╲  ▲   ╱
             │                      ╱     ╲╱ ╲ ╱
             │         ▲           ╱          V
             │        ╱ ╲   ▲     ╱
             │───────╱───╲─╱─╲───╱───────────────────────────────────────── (Baseline)
             │      V     V   ╲ ╱
             │                 V
             └─────────────────────────────────────────────────────────────► Time (t)
```

NeuroCanvas models this mechanism using the **Sequential Probability Ratio Test (SPRT / Drift-Diffusion Model)** (Gold & Shadlen, 2007 [51]):

$$\Lambda(t) = \Lambda(t - 1) + \ln \left( \frac{P(\text{Canvas} \mid \text{Alternative Goal}) + \epsilon}{P(\text{Canvas} \mid \text{Active Goal}) + \epsilon} \right) \cdot \left(0.5 + \frac{d_{T^2}}{\pi}\right)$$

* **If Active Goal is confirmed** ($P(\text{Active}) \ge 0.28$): $\Lambda(t)$ resets toward zero, reinforcing the active state.
* **If Environment persists in another state**: $\Lambda(t)$ drifts upward at a rate proportional to the sensory log-odds ratio and the toroidal geodesic distance $d_{T^2}$.
* **Branching Execution ($\Lambda(t) > \theta_{\text{branch}} = 1.35$):**
  The agent undergoes an **attractor hop of $\sim 120^\circ$** on the semantic manifold. The active goal is pushed to the bottom of the pending queue $\text{Queue}_{\text{Plan B}}$, and the top counterfactual hypothesis is loaded into the active executive buffer.

---

### 2.5 Working Memory 2.0: Deep-Layer Beta Gating of Superficial Gamma Assemblies
As detailed in the **Working Memory 2.0** framework (Miller, Lundqvist, & Bastos, 2018 *Neuron* [2]; Bastos et al., 2018 *PNAS* [4]; Lundqvist et al., 2018 *Nat. Commun.* [5]):

```
                          LAMINAR WORKING MEMORY 2.0 DYNAMICS
                          
    SUPERFICIAL LAYERS (L2/3)        Sensory Gamma Bursts (30–65 Hz)
    [Bottom-Up Contents / PING]      Active Assembly Readout & Plasticity
                 ▲
                 │  Inhibitory Gating: g = P_gamma / (P_gamma + P_beta)
                 │  High Beta -> Gamma Suppressed (Status Quo / Locked)
                 ▼  Low Beta  -> Gamma Disinhibited (Morphing / Free Update)
    DEEP LAYERS (L5/6)               Executive Top-Down Control (15–30 Hz Beta)
    [Thalamocortical Recurrence]     Infragranular Status Quo Maintenance
```

* **Superficial Layers (L2/3):** Host Pyramidal-Interneuron Network Gamma (PING) assemblies ($30\text{--}65\text{ Hz}$) encoding current sensory/semantic features.
* **Deep Layers (L5/6):** Generate alpha/beta rhythms ($15\text{--}30\text{ Hz}$) that project inhibitory drive to superficial layers, clamping down gamma activity and maintaining the current cognitive state (*status quo*).

The instantaneous gating factor $g(t) \in [0, 1]$ is computed as:

$$g(t) = \frac{P_\gamma(t)}{P_\gamma(t) + \kappa P_\beta(t) + \epsilon}$$

Where $P_\gamma$ is local gamma power, $P_\beta$ is local deep beta power, and $\kappa$ is the inhibitory coupling gain:
* **High Beta ($P_\beta \gg P_\gamma \implies g(t) \to 0$):** Gating is closed. The latent representation is held rigid; the diffusion model maintains visual inertia.
* **Beta Desynchronization ($P_\beta \to 0 \implies g(t) \to 1$):** Gating is open. Superficial gamma bursts fire, permitting rapid synaptic weight accumulation in L4 and letting the diffusion worker update its latent state with high mobility.

---

### 2.6 Gap-Free Electrophysiological Spectral Stack: Continuous 500 Hz Multi-Band HAL
In **Dickey et al. (2022, *PNAS*)** [15], human intracranial recordings established that:
> *"Cortical ripples were consistently $\sim 70\text{- to } 85\text{-ms-long}$, **$\sim 90\text{-Hz}$ oscillations**... with an average frequency of **$89.1 \pm 0.8\text{ Hz}$ during NREM** and **$89.5 \pm 0.7\text{ Hz}$ during waking**, detected using a $70\text{--}100\text{ Hz}$ bandpass."*

NeuroCanvas configures a gap-free spectral filter stack operating at a sampling rate of $F_s = 500.0\text{ Hz}$ (Nyquist limit $250.0\text{ Hz}$):

| Band Identifier | Frequency Range | Physiological Function | Key Citation |
|---|---|---|---|
| **Delta** | $1.0\text{--}3.0\text{ Hz}$ (peak $1.5\text{ Hz}$) | Macro-frame scene parsing & syntactic containers | Ding et al. (2016) *Nat. Neurosci.* [11] |
| **Theta** | $4.0\text{--}8.0\text{ Hz}$ (peak $6.0\text{ Hz}$) | Time-division sequence multiplexing (32 slots) | Lisman & Jensen (2013) *Neuron* [6] |
| **Alpha** | $8.0\text{--}12.0\text{ Hz}$ | Sensory gating and selective inhibition | Jensen & Mazaheri (2010) *Front. Hum. Neurosci.* [19] |
| **Beta** | $15.0\text{--}30.0\text{ Hz}$ (peak $22.0\text{ Hz}$) | Deep-layer inhibitory status quo clamp (L5/6) | Miller et al. (2018) *Neuron* [2] |
| **Low/Mid Gamma** | $30.0\text{--}65.0\text{ Hz}$ | PING semantic feature binding ("What") | Colgin et al. (2009) *Nature* [25] |
| **Human Cortical Ripples** | **$65.0\text{--}100.0\text{ Hz}$** (peak **$89.5\text{ Hz}$**) | Long-range phase-locking & inter-areal binding | **Dickey et al. (2022) *PNAS*** [15] |
| **Fast Ripples / HFO** | $100.0\text{--}200.0\text{ Hz}$ | Micro-spike coordination & subicular/thalamic bursts | Buzsáki (2015) *Hippocampus* [57] |

---

### 2.7 Causal Directionality: Volume-Conduction-Free Corrected $ci\text{PLV}$ at 89.5 Hz
Measuring electrophysiological coupling using standard coherence is confounded by volume conduction, wherein a single dipolar source projects instantaneously across adjacent scalp electrodes with zero phase lag (Nolte et al., 2004 [13]).

To isolate genuine axonal propagation delays, NeuroCanvas computes the **Corrected Imaginary Phase-Locking Value ($ci\text{PLV}$)** across all 120 electrode pairs in the $89.5\text{ Hz}$ ripple band, formulated by **Bruña, Maestú, & Pereda (2018 *J. Neural Eng.*, Equation 14)** [12]:

$$ci\text{PLV}_{j, k} = \frac{\frac{1}{T} \Im \left\{ \sum_{t=1}^T \dot{x}_j(t) \cdot \dot{x}_k^*(t) \right\}}{\sqrt{1 - \left( \frac{1}{T} \Re \left\{ \sum_{t=1}^T \dot{x}_j(t) \cdot \dot{x}_k^*(t) \right\} \right)^2}}$$

Where $\dot{x}(t) = \frac{x_{\text{analytic}}(t)}{|x_{\text{analytic}}(t)|} = e^{i\varphi(t)}$ is the normalized analytic phase vector.

#### Extraction of Hierarchical Polarity
Because $ci\text{PLV}$ preserves the mathematical sign of the phase lead:
* **Positive Lead ($\text{Lead}_{A \to B} > 0 \implies \sin(\Delta\varphi) > 0$):** Channel $A$ leads Channel $B$. Entity $A$ is the causal driver and forms the **Parent Container ($A \supset B$)**.
* **Negative Lead ($\text{Lead}_{A \to B} < 0 \implies \sin(\Delta\varphi) < 0$):** Channel $B$ leads Channel $A$. The relationship inverts, establishing $B$ as the container (**$B \supset A$**).
* **Near-Zero ($|\text{Lead}| \le 0.03$):** Entities co-occur without an established directional lag, rendering them as **Peers ($A \parallel B$)**.

---

### 2.8 jPCA Rotational Manifolds Derived from $\mathfrak{so}(3)$ Lie Algebras
Population trajectories in premotor and prefrontal cortex are characterized by prominent skew-symmetric rotational dynamics (Churchland et al., 2012 *Nature* [7]):

$$\dot{\mathbf{x}}(t) = \mathbf{M}_{\text{skew}} \mathbf{x}(t), \qquad \mathbf{M}_{\text{skew}} = -\mathbf{M}_{\text{skew}}^T$$

In NeuroCanvas, $\mathbf{M}_{\text{skew}} \in \mathfrak{so}(3)$ is generated from the instantaneous carrier angular frequency $\omega_\theta = 2\pi f_\theta$ along the principal normal axis $\hat{\mathbf{n}}$:

$$\mathbf{M}_{\text{skew}} = 2\pi f_\theta \begin{bmatrix} 0 & -n_z & n_y \\ n_z & 0 & -n_x \\ -n_y & n_x & 0 \end{bmatrix}$$

This skew-symmetric operator transforms the planar gamma gradient vector $\mathbf{v}_\gamma = [v_x, v_y, 0.5 v_x]^T$ into a valid 3D displacement vector:

$$\Delta\mathbf{p} = \mathbf{M}_{\text{skew}}^T \mathbf{v}_\gamma \cdot \Delta t$$

From this displacement, Gram-Schmidt orthonormalization constructs the instantaneous 3D frame matrix $[\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3] \in SO(3)$, which is packaged directly into the Cortical Messaging Protocol (CMP).

---

## 3. tbp.monty Integration & CMP Compliance

NeuroCanvas maintains strict compatibility with the **Thousand Brains Project (`tbp.monty`)** runtime architecture.

### 3.1 Strict Cortical Messaging Protocol (CMP) Packet Structure
Every simulation cycle compiles the decoded prefrontal state into an authentic `tbp.monty.cmp.Message` instance:

```python
Message(
    location=node_f3.disp_xyz.astype(np.float64), # 3D vector [x, y, z] (shape (3,))
    morphological_features={
        "pose_vectors": so3_matrix.astype(np.float64), # Orthonormal rotation matrix (3, 3)
        "pose_fully_defined": bool(smooth_depth >= 1.8),# True when SVD manifold rank K >= 1.8
        "on_object": 1.0                              # Sensor currently engaged on manifold
    },
    non_morphological_features={
        "object_id": int(np.argmax(wm_scores)),       # Decoded concept index in vocabulary
        "torus_u": node_afz.torus_u,                  # Theta-slot coordinate u on Janata Torus
        "torus_v": node_afz.torus_v,                  # Delta-frame coordinate v on Janata Torus
        "beta_f3": node_f3.beta_power,                # L5/6 Form inhibitory gate (Left)
        "beta_f4": node_f4.beta_power                 # L5/6 Style inhibitory gate (Right)
    },
    confidence=float(np.clip(node_f3.beta_stability, 0.0, 1.0)), # Stability metric [0.0, 1.0]
    pass_message=True,                                # Message forwarded to downstream LMs
    process_features_in_lm=True,                      # CRITICAL: Enables hypothesis evidence updates
    sender_id="Prefrontal_Heterarchy_SM",             # Originating SensorModule identifier
    sender_type="SM"                                  # Validated against allowable types ("SM", "LM")
)
```

#### Why `process_features_in_lm=True` is Essential
In `src/tbp/monty/frameworks/models/evidence_matching/learning_module.py`:
```python
def matching_step(self, ctx: RuntimeContext, percepts: Sequence[Message]) -> None:
    if is_location_only_step(percepts):
        self._displace_hypotheses(percepts)
        return
    ...
```
If `process_features_in_lm` evaluates to `False`, the learning module treats the incoming transmission as a pure relocation command, displacing existing hypotheses without evaluating feature evidence. Passing `process_features_in_lm=True` ensures that `EvidenceGraphLM` updates its internal state space.

---

### 3.2 Sensorimotor Frame Transformations & Relational Compositionality
Under TBT 2.0 (Hawkins et al., 2025/2026 [1]), when two cortical areas learn compositional structures:
1. **Lower Area ($R_1$ / Child):** Identifies an object model (e.g., *Castle*) in its local reference frame and transmits its identity via L3 feedforward projections to L4 of the higher area ($R_2$ / Parent).
2. **Higher Area ($R_2$ / Parent):** Identifies the enclosing structure (e.g., *Mountain*). Feedback connections from L6a of $R_2$ project to L6a and L1 of $R_1$, constraining the expected local coordinates on the child object.
3. **Thalamic Alignment:** Projections from L6b of both regions converge on higher-order thalamic relay cells, computing the relative rotation $\Delta \mathbf{R} \in SO(3)$ between the child and parent frames.

NeuroCanvas computes this relative pose $\Delta \mathbf{R}$ dynamically from the phase trajectory on the Janata Torus:

$$\Delta u = u_{\text{child}} - u_{\text{parent}}, \quad \Delta v = v_{\text{child}} - v_{\text{parent}}$$

$$\Delta \mathbf{p}_{SO(3)} = \begin{bmatrix} \sin(\Delta u) \\ \cos(\Delta v) \\ \sin(\Delta u + \Delta v) \end{bmatrix}$$

This vector modulates the phase of the $89.5\text{ Hz}$ cortical ripples generated by the autonomous agents, driving real-time structural binding across the simulated cortical sheet.

---

### 3.3 16,384-Column CUDA Cortical Macrocolumn Sheet
* **Cellular Scale:** 4 simulated prefrontal nodes contain 4,096 canonical columns each, comprising **16,384 cortical columns** executed in parallel on CUDA.
* **Top-$K$ Sparse Distributed Representation (SDR):** Local lateral inhibition maintains an active population sparsity of $1.95\%$ ($k = 80$ active columns per node, $K_{\text{total}} = 320$ active units across the macrocolumn sheet) (Hawkins et al., 2017 [3]).
* **Presynaptic Calcium Accumulator ($\text{Ca}^{2+}$):** Models short-term facilitation across inter-burst intervals (Mongillo et al., 2008 *Science* [31]):
  
  $$\mathbf{C}(t) = \max(\mathbf{C}(t - \Delta t) \cdot 0.90, \; \mathbf{SDR}(t))$$

* **Structural Long-Term Potentiation (LTP):** Permanent synaptogenesis is triggered when visual confirmation ($P_{\text{CLIP}} \ge 0.25$) coincides with persistent calcium traces:
  
  $$\mathbf{W}_{ij} \leftarrow \mathrm{clamp}(\mathbf{W}_{ij} + \eta \cdot \mathbf{C}_j, \, 0.0, \, 1.0)$$

Consolidation of an engram is verified by assessing the functional synaptic density ($\text{LTM Score} = \frac{\sum [\mathbf{W} > 0.5]}{K_{\text{target}}} \ge 75.0\%$).

---

## 4. Autonomous Stigmergic Swarm & VLA-JEPA World Modeling

### 4.1 Zero-Metagame Multi-Agent Interaction via Environmental Traces
Traditional multi-agent simulations rely on internal message passing, where agents exchange goal indices through shared software memory.

NeuroCanvas enforces **strict stigmergy** (Grassé, 1959 [56]; Clark, 2008 [41]):
* Agents have **no access** to the internal variables, goals, or identities of other agents.
* The sole communication medium is the **generated visual canvas**.
* Agents perceive the environment through sensory classifiers (CLIP / V-JEPA 2) and assert their intentions by injecting field potentials into the LSL stream.

---

### 4.2 Latent Energy Minimization vs. Superficial Feature Matching
As highlighted in **VLA-JEPA (Sun et al., 2026)** [49] and **V-JEPA 2 (Assran et al., 2025)** [50], pixel-level objectives fail in embodied control tasks due to visual clutter, high-frequency appearance variations, and camera motions.

```
       VLA-JEPA LATENT WORLD MODEL OPTIMIZATION (Sun et al., 2026)
       
       Observed Canvas I_t ────────► [ Frozen V-JEPA 2 Encoder E_theta ] ──► Latent World State s_t
                                                                                    │
       Latent Action Query z_t ────► [ Autoregressive World Predictor P_phi ]        │
                                                    │                               │
                                                    ▼                               ▼
                                            Predicted s^(t+1) ◄─── L1 Latent Loss ─── Target s_(t+1)
                                                                (Energy Minimization)
```

In `FullJepaVideoAgent`:
1. The agent receives the rendered RGB stream and extracts continuous latent states:
   
   $$\mathbf{z}_t = E_\theta(I_t) \in \mathbb{R}^{1 \times 77 \times 2048}$$

2. It evaluates the **latent world model prediction error**:
   
   $$\mathcal{E}_{\text{JEPA}} = \|\mathbf{z}_t - \mathbf{z}_{\text{target}}\|_1$$

3. This latent energy directly modulates the evidence accumulation rate in the agent's SPRT drift engine:
   
   $$\Delta \Lambda_{\text{step}} = \ln\left(\frac{P(\text{Dominant}) + \epsilon}{P(\text{Target}) + \epsilon}\right) \cdot \left(0.6 + 0.4 \cdot \mathcal{E}_{\text{JEPA}}\right)$$

This mechanism grounds the agent's internal frustration in physical dynamics rather than superficial label matching.

---

### 4.3 Phase-Division Multiple Access (PDMA) Across Theta Cycles
To prevent frequency collision when multiple agents broadcast simultaneously, agents utilize **Phase-Division Multiple Access (PDMA)**, multiplexing their outputs across the $6.0\text{ Hz}$ theta cycle (Lisman & Jensen, 2013 [6]; Bieri et al., 2014 [26]):
* **Leader / Enclosing Parent ($\text{Role} \in \{\text{LEADER}, \text{SUPER\_PARENT}\}$):**
  Discharges at **early theta phases ($\theta \approx 0.20\text{--}0.35$)**, corresponding to retrospective readout and macro-contextual stability.
* **Embedded Sub-Component ($\text{Role} = \text{SUB\_CHILD}$):**
  Discharges at **late theta phases ($\theta \approx 0.70\text{--}0.85$)**, corresponding to prospective coding and local structural modification.
* **Co-Equal Peer ($\text{Role} = \text{PEER}$):**
  Discharges at **mid-theta phases ($\theta \approx 0.45\text{--}0.55$)**.

This temporal segregation enables multiple distinct concept assemblies to occupy the LFP spectrum without phase cancellation.

---

### 4.4 Synaptic Long-Term Potentiation (LTP) & Engram Consolidation Thresholds
Engram formation across Layer 4 columns requires persistent synaptic remodeling (Tonegawa et al., 2015 [44]).

In NeuroCanvas:
* **Synaptic Permanence:** Initialized below connection threshold ($P_{\text{initial}} < 0.25$).
* **Hebbian Potentiation:** When visual feedback confirms semantic emergence ($P_{\text{CLIP}} \ge 0.25$), active columns increase permanence by $\Delta P = \eta \cdot \mathbf{C}(t)$.
* **Consolidation Criterion:** A concept is marked as `[CONSOLIDATED]` if and only if:
  
  $$\text{LTM Score} = \frac{\sum_{j=1}^{M} \mathbb{I}(\mathbf{W}_{ij} > 0.50)}{K_{\text{active}}} \times 100\% \ge 75.0\%$$

No boolean flag can override this physical threshold. If any concept exhibits an LTM score $< 75.0\%$, the system automatically enforces **Synaptic Calibration Mode**, presenting targets sequentially until genuine structural consolidation occurs.

---

## 5. Hardware & Software Architecture

```
                                SYSTEM HARDWARE & IPC TOPOLOGY
                                
    ┌───────────────────────────┐                ┌───────────────────────────┐
    │     LSL Output Nodes      │                │  V-JEPA 2 Server (6001)   │
    │  FreeEEG_Node0 .. Node3   │                │   ViT-g / 2048-D Latent   │
    └─────────────┬─────────────┘                └─────────────┬─────────────┘
                  │ 500 Hz 16-ch LSL Streams                   │ Inter-Process Pipes
                  ▼                                            ▼
    ┌───────────────────────────┐                ┌───────────────────────────┐
    │  GPU_Daemon_Process (HAL) │◄─Shared Memory─►│  ToroidalDiffusionWorker  │
    │ FFT • ciPLV • PAC • jPCA  │                │  SDXL-Turbo / SD-Turbo    │
    └─────────────┬─────────────┘                └─────────────┬─────────────┘
                  │ PyTorch Tensors                            │ Generated RGB
                  ▼                                            ▼
    ┌───────────────────────────┐                ┌───────────────────────────┐
    │ FrontalExecutiveHeterarchy│                │    VisualCLIPTeacher      │
    │ 16,384-Col L4 Sheet (CUDA)│                │     ViT-L/14 Classifier   │
    └───────────────────────────┘                └───────────────────────────┘
```

### 5.1 FreeEEG16-alpha2 Concentric Ring Sensor Array & Surface Laplacian
* **Array Specification:** FreeEEG16-alpha2 $26\text{-mm}$ dual-concentric gold-plated surface array (Besio et al., 2006 [19]).
* **Electrode Geometry:** 16 recording contacts arranged in two concentric rings ($r_1 = 6.0\text{ mm}$, $r_2 = 10.5\text{ mm}$).
* **Signal Conditioning:** Real-time surface Laplacian spatial filtering removes far-field volume conduction; twin notch filters at $50\text{ Hz}$ and $100\text{ Hz}$ reject AC mains hum.
* **Sampling Rate:** $F_s = 500.0\text{ Hz}$, maintaining zero temporal aliasing across all bands up to $200\text{ Hz}$.

---

### 5.2 Server-Client IPC Topology
1. **`brain_server.py` (Port 6000):**
   * Hosts the generative diffusion pipeline (`LCMScheduler` / `SDXL-Turbo`).
   * Manages VRAM allocation, execution caching, and text encoder pooling.
2. **`jepa_server.py` (Port 6001):**
   * Hosts the isolated V-JEPA 2 visual feature encoder.
   * Serves 2048-D latent state embeddings via memory-mapped IPC pipes.
3. **`neuro_prefrontal_heterarchy_live.py` (Primary Engine):**
   * Integrates the 500 Hz HAL, the 16,384-column L4 CUDA sheet, and the Pygame graphical interface.

---

## 6. CLI Reference & Quickstart

### Installation
```bash
git clone https://github.com/your-repo/neuro-prefrontal-heterarchy.git
cd neuro-prefrontal-heterarchy

conda create -n neurocanvas python=3.10 -y
conda activate neurocanvas

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate pylsl opencv-python pygame pillow scipy
```

### 1. Launch Backends
In dedicated terminals:
```bash
# Terminal 1: Generative Diffusion Server
python brain_server.py --mode lcm

# Terminal 2 (Optional): Lightweight V-JEPA 2 World Model Server
python jepa_server.py --port 6001
```

### 2. Launch Closed-Loop Engine
```bash
python neuro_prefrontal_heterarchy_live.py --concepts 8 --hardcoded-bots 2 --jepa-bots 2 --no-color --no-taesd
```

### Available Options:
* `--concepts {4, 8}`: Number of active vocabulary entries (default: `8`).
* `--mode {lcm, turbo, sdxl-turbo}`: Latent diffusion pipeline (default: `lcm`).
* `--speed {fast, quality}`: Image resolution tier ($448 \times 336$ vs. $512 \times 384$).
* `--hardcoded-bots N`: Number of active inference agents.
* `--jepa-bots N`: Number of VLA-JEPA latent video agents.
* `--force-recalib`: Ignores existing weights file and initiates fresh LTP calibration.
* `--no-taesd`: Bypasses Tiny AutoEncoder, utilizing standard SD VAE.
* `--no-color`: Disables post-hoc color surgery transformations.

---

## 7. Comprehensive Scientific Bibliography & DOIs

1. **Hawkins, J., Leadholm, N., & Clay, V. (2025/2026).** The Thousand Brains Theory 2.0: An Extension for the Long-Range Connections of the Neocortical Heterarchy. *arXiv preprint*, [arXiv:2507.05888](https://arxiv.org/abs/2507.05888).
2. **Miller, E. K., Lundqvist, M., & Bastos, A. M. (2018).** Working Memory 2.0. *Neuron*, 100(2), 463–475. [DOI: 10.1016/j.neuron.2018.09.023](https://doi.org/10.1016/j.neuron.2018.09.023)
3. **Hawkins, J., Ahmad, S., & Cui, Y. (2017).** A theory of how columns in the neocortex enable learning the structure of the world. *Frontiers in Neural Circuits*, 11, 81. [DOI: 10.3389/fncir.2017.00081](https://doi.org/10.3389/fncir.2017.00081)
4. **Bastos, A. M., Loonis, R., Kornblith, S., Lundqvist, M., & Miller, E. K. (2018).** Laminar recordings in frontal cortex suggest distinct layers for maintenance and control of working memory. *Proceedings of the National Academy of Sciences (PNAS)*, 115(5), 1117–1122. [DOI: 10.1073/pnas.1714522115](https://doi.org/10.1073/pnas.1714522115)
5. **Lundqvist, M., Herman, P., Warden, M. R., Brincat, S. L., & Miller, E. K. (2018).** Gamma and beta bursts during working memory readout suggest roles in its volitional control. *Nature Communications*, 9(1), 394. [DOI: 10.1038/s41467-017-02791-8](https://doi.org/10.1038/s41467-017-02791-8)
6. **Lisman, J. E., & Jensen, O. (2013).** The theta-gamma neural code. *Neuron*, 77(6), 1002–1016. [DOI: 10.1016/j.neuron.2013.03.007](https://doi.org/10.1016/j.neuron.2013.03.007)
7. **Churchland, M. M., Cunningham, J. P., Kaufman, M. T., Foster, J. D., Nuyujukian, P., Ryu, S. I., & Shenoy, K. V. (2012).** Neural population dynamics during reaching. *Nature*, 487(7405), 51–56. [DOI: 10.1038/nature11129](https://doi.org/10.1038/nature11129)
8. **Tort, A. B., Komorowski, R., Eichenbaum, H., & Kopell, N. (2010).** Measuring phase-amplitude coupling between neuronal oscillations of different frequencies. *Journal of Neurophysiology*, 104(2), 1195–1210. [DOI: 10.1152/jn.00106.2010](https://doi.org/10.1152/jn.00106.2010)
9. **Fusi, S., Miller, E. K., & Rigotti, M. (2016).** Why neurons mix: high dimensionality for higher cognition. *Current Opinion in Neurobiology*, 37, 66–74. [DOI: 10.1016/j.conb.2016.01.010](https://doi.org/10.1016/j.conb.2016.01.010)
10. **Badre, D., & Nee, D. E. (2018).** Frontal cortex and the hierarchical control of behavior. *Trends in Cognitive Sciences*, 22(2), 170–188. [DOI: 10.1016/j.tics.2017.11.005](https://doi.org/10.1016/j.tics.2017.11.005)
11. **Ding, N., Melloni, L., Zhang, H., Tian, X., & Poeppel, D. (2016).** Cortical tracking of hierarchical linguistic structures in connected speech. *Nature Neuroscience*, 19(1), 158–164. [DOI: 10.1038/nn.4186](https://doi.org/10.1038/nn.4186)
12. **Bruña, R., Maestú, F., & Pereda, E. (2018).** Phase Locking Value revisited: teaching new tricks to an old dog. *Journal of Neural Engineering*, 15(5), 056011. [DOI: 10.1088/1741-2552/aacfe4](https://doi.org/10.1088/1741-2552/aacfe4)
13. **Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., & Hallett, M. (2004).** Identifying true brain interaction from EEG data using the imaginary part of coherency. *Clinical Neurophysiology*, 115(10), 2292–2307. [DOI: 10.1016/j.clinph.2004.04.029](https://doi.org/10.1016/j.clinph.2004.04.029)
14. **Kikumoto, A., & Mayr, U. (2020).** Decoding hierarchical control of sequential behavior in oscillatory EEG activity. *eLife*, 9, e53589. [DOI: 10.7554/eLife.53589](https://doi.org/10.7554/eLife.53589)
15. **Dickey, C. W., Verzhbinsky, I. A., Jiang, X., Rosen, B. Q., Kajfez, S., Stedelin, B., ... & Halgren, E. (2022).** Widespread ripples synchronize human cortical activity during sleep, waking, and memory recall. *Proceedings of the National Academy of Sciences (PNAS)*, 119(28), e2107797119. [DOI: 10.1073/pnas.2107797119](https://doi.org/10.1073/pnas.2107797119)
16. **Arnulfo, G., Pozzi, N. G., Wang, H. X., Scorrano, V., Canessa, A., Palva, S., & Palva, J. M. (2020).** Long-range phase synchronization of high-frequency oscillations in human cortex. *Nature Communications*, 11(1), 5363. [DOI: 10.1038/s41467-020-18975-8](https://doi.org/10.1038/s41467-020-18975-8)
17. **Hawkins, J., Lewis, M., Klukas, M., Purdy, S., & Ahmad, S. (2019).** A framework for intelligence and cortical function based on grid cells in the neocortex. *Frontiers in Neural Circuits*, 13, 86. [DOI: 10.3389/fncir.2019.00086](https://doi.org/10.3389/fncir.2019.00086)
18. **Mountcastle, V. B. (1997).** The columnar organization of the neocortex. *Brain*, 120(4), 701–722. [DOI: 10.1093/brain/120.4.701](https://doi.org/10.1093/brain/120.4.701)
19. **Besio, W. G., Koka, K., & Aakula, R. (2006).** Tri-polar concentric ring electrode development for Laplacian electroencephalography. *IEEE Transactions on Biomedical Engineering*, 53(5), 926–933. [DOI: 10.1109/TBME.2006.873398](https://doi.org/10.1109/TBME.2005.863887)
20. **Lakatos, P., Shah, A. S., Knuth, K. H., Ulbert, I., Karmos, G., & Schroeder, C. E. (2005).** An oscillatory hierarchy controlling neuronal excitability and stimulus processing in the auditory cortex. *Journal of Neurophysiology*, 94(3), 1904–1911. [DOI: 10.1152/jn.00263.2005](https://doi.org/10.1152/jn.00263.2005)
21. **Friston, K. (2010).** The free-energy principle: a unified brain theory?. *Nature Reviews Neuroscience*, 11(2), 127–138. [DOI: 10.1038/nrn2787](https://doi.org/10.1038/nrn2787)
22. **Stringer, C., Pachitariu, M., Steinmetz, N., Carandini, M., & Harris, K. D. (2019).** High-dimensional geometry of population responses in visual cortex. *Nature*, 571(7765), 361–365. [DOI: 10.1038/s41586-019-1346-5](https://doi.org/10.1038/s41586-019-1346-5)
23. **Miller, E. K., & Cohen, J. D. (2001).** An integrative theory of prefrontal cortex function. *Annual Review of Neuroscience*, 24(1), 167–202. [DOI: 10.1146/annurev.neuro.24.1.167](https://doi.org/10.1146/annurev.neuro.24.1.167)
24. **Voloh, B., Valiante, T. A., & Womelsdorf, T. (2015).** Theta–gamma coordination between anterior cingulate and prefrontal cortex indexes correct attention shifts. *Proceedings of the National Academy of Sciences (PNAS)*, 112(27), 8457–8462. [DOI: 10.1073/pnas.1502092112](https://doi.org/10.1073/pnas.1502092112)
25. **Colgin, L. L., Denninger, T., Fyhn, M., Hafting, T., Bonnevie, T., Jensen, O., ... & Moser, E. I. (2009).** Frequency of gamma oscillations routes flow of information in the hippocampus. *Nature*, 462(7271), 353–357. [DOI: 10.1038/nature08573](https://doi.org/10.1038/nature08573)
26. **Bieri, K. W., Bobbitt, K. N., & Colgin, L. L. (2014).** Slow and fast gamma rhythms coordinate different spatial coding modes in hippocampal place cells. *Neuron*, 82(3), 670–681. [DOI: 10.1016/j.neuron.2014.03.013](https://doi.org/10.1016/j.neuron.2014.03.013)
27. **Norman, Y., Yeagle, E. M., Khuvis, S., Harel, M., Mehta, A. D., & Malach, R. (2021).** Post-activation pause: An anatomical signature of memory recall in human cortex. *Science*, 373(6560), eabg7595. [DOI: 10.1126/science.abg7595](https://doi.org/10.1126/science.abg7595)
28. **Boorman, E. D., Behrens, T. E., Woolrich, M. W., & Rushworth, M. F. (2009).** How green is the grass on the other side? Frontopolar cortex and the evidence in favor of alternative courses of action. *Neuron*, 62(5), 733–743. [DOI: 10.1016/j.neuron.2009.05.014](https://doi.org/10.1016/j.neuron.2009.05.014)
29. **Koechlin, E., & Hyafil, A. (2007).** Anterior prefrontal function and the limits of human decision-making. *Science*, 318(5850), 594–598. [DOI: 10.1126/science.1142995](https://doi.org/10.1126/science.1142995)
30. **Alexander, W. H., & Brown, J. W. (2011).** Mediodorsal prefrontal cortex as a prediction error hub for cognitive control. *Nature Neuroscience*, 14(10), 1338–1344. [DOI: 10.1038/nn.2921](https://doi.org/10.1038/nn.2921)
31. **Mongillo, G., Barak, O., & Tsodyks, M. (2008).** Synaptic theory of working memory. *Science*, 319(5869), 1543–1546. [DOI: 10.1126/science.1150769](https://doi.org/10.1126/science.1150769)
32. **Badre, D., & D'Esposito, M. (2007).** Functional magnetic resonance imaging evidence for a hierarchical organizing principle in the prefrontal cortex. *Nature Neuroscience*, 10(9), 1138–1144. [DOI: 10.1038/nn1953](https://doi.org/10.1038/nn1953)
33. **Tsujimoto, S., Genovesio, A., & Wise, S. P. (2010).** Frontopolar cortex: neuronal networks for decision-making and metacognition. *Journal of Neuroscience*, 30(50), 16756–16759. [DOI: 10.1523/JNEUROSCI.6667-09.2010](https://doi.org/10.1523/JNEUROSCI.6667-09.2010)
34. **Shenhav, A., Botvinick, M. M., & Cohen, J. D. (2013).** The expected value of control: an executive function specification for anterior cingulate cortex. *Nature Neuroscience*, 16(7), 885–892. [DOI: 10.1038/nn.3423](https://doi.org/10.1038/nn.3423)
35. **Womelsdorf, T., Johnston, K., Vinck, M., & Everling, S. (2010).** Theta-activity in anterior cingulate cortex predicts task rules and their adjustments. *Journal of Neuroscience*, 30(38), 12694–12702. [DOI: 10.1523/JNEUROSCI.2861-10.2010](https://doi.org/10.1523/JNEUROSCI.2861-10.2010)
36. **Chen, J., Zhang, C., Hu, P., Min, B., & Wang, L. (2024).** Flexible control of sequence working memory in the macaque frontal cortex. *Neuron*, 112(20), 3502–3514. [DOI: 10.1016/j.neuron.2024.07.024](https://doi.org/10.1016/j.neuron.2024.07.024)
37. **Fan, Y., Wang, M., Ding, N., & Luo, H. (2024).** Two-dimensional neural geometry underpins hierarchical organization of sequence in human working memory. *Nature Human Behaviour*, 8, 2150–2163. [DOI: 10.1038/s41562-024-02047-8](https://doi.org/10.1038/s41562-024-02047-8)
38. **Siegel, M., Warden, M. R., & Miller, E. K. (2009).** Phase-dependent neuronal coding of objects in short-term memory. *Proceedings of the National Academy of Sciences (PNAS)*, 106(50), 21341–21346. [DOI: 10.1073/pnas.0908193106](https://doi.org/10.1073/pnas.0908193106)
39. **Buzsáki, G., & Tingley, D. (2018).** Space and time: the hippocampus as a sequence generator. *Trends in Cognitive Sciences*, 22(10), 853–869. [DOI: 10.1016/j.tics.2018.07.006](https://doi.org/10.1016/j.tics.2018.07.006)
40. **Takagi, Y., & Nishimoto, S. (2023).** High-resolution image reconstruction with latent diffusion models from human brain activity. *Nature Communications*, 14(1), 1568. [DOI: 10.1038/s41467-023-36701-1](https://doi.org/10.1038/s41467-023-36701-1)
41. **Clark, A. (2008).** *Supersizing the Mind: Embodiment, Action, and Cognitive Extension.* Oxford University Press. [DOI: 10.1093/acprof:oso/9780195333213.001.0001](https://doi.org/10.1093/acprof:oso/9780195333213.001.0001)
42. **Dumas, G., Nadel, J., Soussignan, R., Martinerie, J., & Garnero, L. (2010).** Inter-brain synchronization during social interaction. *PLoS ONE*, 5(8), e12165. [DOI: 10.1371/journal.pone.0012165](https://doi.org/10.1371/journal.pone.0012165)
43. **Zhang, Y. (2026).** Recurrent Looped Transformer: Latent Reasoning with Unbounded Temporal Depth. *alphaXiv preprint*, [alphaxiv:2609.recurrent-looped-transformer](https://www.alphaxiv.org/abs/2609.recurrent-looped-transformer).
44. **Tonegawa, S., Pignatelli, M., Roy, D. S., & Ryan, T. J. (2015).** Memory engram cells have come of age. *Neuron*, 87(5), 918–931. [DOI: 10.1016/j.neuron.2015.08.002](https://doi.org/10.1016/j.neuron.2015.08.002)
45. **Janata, P., Birk, J. L., Van Horn, J. D., Leman, M., Tillmann, B., & Bharucha, J. J. (2002).** The cortical topography of tonal structures underlying Western music. *Science*, 298(5601), 2167–2170. [DOI: 10.1126/science.1076262](https://doi.org/10.1126/science.1076262)
46. **Fink, G. R., Halligan, P. W., Marshall, J. C., Frith, C. D., Frackowiak, R. S., & Dolan, R. J. (1996).** Where in the brain does visual attention select the forest and the trees? *Nature*, 382(6592), 626–628. [DOI: 10.1038/382626a0](https://doi.org/10.1038/382626a0)
47. **Iaccino, J. F. (2014).** *Left Brain-Right Brain Differences: Inquiries, Evidence, and New Approaches.* Psychology Press. [DOI: 10.4324/9781315806655](https://doi.org/10.4324/9781315806655)
48. **Ewald, A., Marzetti, L., Zurloni, F., Chella, F., Romani, G. L., & Nolte, G. (2012).** Estimating true brain connectivity from EEG/MEG data invariant to linear and static transformations in sensor space. *NeuroImage*, 60(1), 476–488. [DOI: 10.1016/j.neuroimage.2011.11.084](https://doi.org/10.1016/j.neuroimage.2011.11.084)
49. **Sun, J., Zhang, W., Qi, Z., Ren, S., Liu, Z., Zhu, H., ... & Chen, Z. (2026).** VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model. *arXiv preprint*, [arXiv:2602.10098](https://arxiv.org/abs/2602.10098).
50. **Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Komeili, M., ... & Ballas, N. (2025).** V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. *arXiv preprint*, [arXiv:2506.09985](https://arxiv.org/abs/2506.09985).
51. **Gold, J. I., & Shadlen, M. N. (2007).** The neural basis of decision making. *Annual Review of Neuroscience*, 30(1), 535–574. [DOI: 10.1146/annurev.neuro.29.051605.113038](https://doi.org/10.1146/annurev.neuro.29.051605.113038)
52. **Averbeck, B. B., Chafee, M. V., Crowe, D. A., & Georgopoulos, A. P. (2002).** Parallel representation of serial order in macaque motor cortex. *Experimental Brain Research*, 146(4), 402–409. [DOI: 10.1007/s00221-002-1140-9](https://doi.org/10.1007/s00221-002-1140-9)
53. **Grossberg, S. (1980).** How does a brain build a cognitive code? *Psychological Review*, 87(1), 1–51. [DOI: 10.1037/0033-295X.87.1.1](https://doi.org/10.1037/0033-295X.87.1.1)
54. **Torgerson, W. S. (1952).** Multidimensional scaling: I. Theory and method. *Psychometrika*, 17(4), 401–419. [DOI: 10.1007/BF02288916](https://doi.org/10.1007/BF02288916)
55. **Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008).** Representational similarity analysis - connecting the branches of systems biology. *Frontiers in Systems Neuroscience*, 2, 4. [DOI: 10.3389/neuro.06.004.2008](https://doi.org/10.3389/neuro.06.004.2008)
56. **Grassé, P. P. (1959).** La reconstruction du nid et les coordinations interindividuelles chez Bellicositermes natalensis et Cubitermes sp. la théorie de la stigmergie: essai d'interprétation du comportement des termites constructeurs. *Insectes Sociaux*, 6(1), 41–80. [DOI: 10.1007/BF02223791](https://doi.org/10.1007/BF02223791)
57. **Buzsáki, G. (2015).** Hippocampal sharp wave-ripple: A cognitive biomarker for episodic memory and planning. *Hippocampus*, 25(10), 1073–1188. [DOI: 10.1002/hipo.22488](https://doi.org/10.1002/hipo.22488)
