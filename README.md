# Q-Delta: Beyond Key–Value Associative State Evolution

Official implementation of **Q-Delta: Beyond Key–Value Associative State Evolution** (ICML 2026).

*Sumin Park, Seojin Kim, Noseong Park* — Korea Advanced Institute of Science and Technology (KAIST).
Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026.

This repository contains the Q-Delta layer, configuration, and training scripts as a custom model for the [flame framework](https://github.com/fla-org/flame).

## Repository layout

```
Q-Delta/
├── qdelta/                    # model code → flame/custom_models/qdelta/
│   ├── qdelta.py
│   ├── configuration_qdelta.py
│   ├── modeling_qdelta.py
│   └── qdelta_rule/           # Triton kernels (chunk + fused-recurrent)
├── configs/                   # → flame/configs/
│   ├── qdelta_340M.json
│   └── qdelta_1B.json
└── scripts/                   # → flame/
    ├── launch_qdelta_340m.sh
    └── launch_qdelta_1b.sh
```

## Installation

From the root of a flame checkout:

```bash
cp -r path/to/Q-Delta/qdelta   flame/custom_models/
cp    path/to/Q-Delta/configs/*.json flame/configs/
cp    path/to/Q-Delta/scripts/*.sh   flame/
```

## Usage

From within the `flame` directory:

```bash
bash launch_qdelta_340m.sh   # 340M (default GPUs 0–3)
bash launch_qdelta_1b.sh     # 1B   (default GPUs 4–7)
```

Override the GPU set / count with `CUDA_VISIBLE_DEVICES=... NGPU=... bash <script>`.

## Configs

`configs/qdelta_{340M,1B}.json` hold the model hyperparameters. The query-gating bias is exposed as `lamb_bias` (default `0.9`) — edit the JSON to sweep.
