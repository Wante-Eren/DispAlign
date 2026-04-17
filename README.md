# DispAlign

Official implementation of **DispAlign: Dispersion-driven Frequency-aware Alignment for Multimodal Object Re-Identification**.

---

## 🔍 Overview

Multimodal object re-identification (Re-ID) aims to match targets across heterogeneous sensing conditions such as RGB, NIR, TIR, and text. Existing approaches typically rely on direct alignment in a shared feature space, which often assumes that different modalities can be treated uniformly.

In this work, we revisit multimodal interaction from a **frequency-aware perspective**. Instead of performing direct matching, feature interaction is modeled as a **propagation process**, where different frequency components evolve differently.

To achieve this, we introduce a dispersion-driven framework (**DispAlign**) that enables frequency-dependent feature propagation, leading to more stable and consistent cross-modal representations.

---

## 🧠 Key Components

- **HAOP (Holographic Acousto-Optic Projector)**  
  Transforms textual features into a frequency-compatible representation for better interaction with visual modalities.

- **ADWO (Advanced Dispersive Wave Operator)**  
  Models feature propagation with frequency-dependent dispersion and damping mechanisms.

- **AD-WSR (Adaptive Dual-stage Wave-Semantic Router)**  
  Refines multimodal fusion through wave-guided routing and controlled alignment.

> The main implementation of the proposed modules is located in:
> - `modeling/HAOP.py`
> - `modeling/fusion_part/CDA_new.py`

---

## 📁 Project Structure

```text
DispAlign-main
├── assets/                     # figures and supplementary resources
├── config/                     # default configuration definitions
├── configs/
│   └── MSVR310.yml             # example config for reproduction
├── data/
│   └── datasets/               # dataset loaders and sampling utilities
├── engine/
│   └── processor.py            # training and evaluation pipeline
├── layers/                     # loss functions
├── modeling/
│   ├── HAOP.py                 # text-to-frequency module
│   ├── meta_arch.py            # main architecture
│   ├── make_newmodel.py
│   ├── backbones/              # visual backbones
│   ├── clip/                   # CLIP-related modules
│   └── fusion_part/
│       └── CDA_new.py          # fusion / propagation-related module
├── scripts/
│   └── MSVR310.sh              # example training script
├── solver/                     # optimizer and scheduler
├── tool/
│   └── test.py                 # evaluation utilities
├── utils/                      # logging, metrics, FLOPs, reranking
├── requirements.txt            # environment dependencies
├── train.py                    # training entry
└── test.py                     # testing entry