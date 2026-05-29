# Q-Delta: Beyond Key–Value Associative State Evolution

Official implementation of **Q-Delta: Beyond Key–Value Associative State Evolution** (ICML 2026).

*Sumin Park, Seojin Kim, Noseong Park* — Korea Advanced Institute of Science and Technology (KAIST).
Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026.

This repository contains the Q-Delta layer, configuration, and training scripts as a custom model plugged into the [flame framework](https://github.com/fla-org/flame).

## Installation

1. Place the `qdelta` folder in the `custom_models` directory of flame:
```
flame/
├── custom_models/
│   └── qdelta/
└── ...
```

2. Copy the config files from `qdelta_script` to the `configs` directory:
```bash
cp custom_models/qdelta/qdelta_script/*.json configs/
```

## Usage

From within the `flame` directory, run the following commands:

**Train 1B model:**
```bash
bash launch_qdelta_1b.sh
```

**Train 340M model:**
```bash
bash launch_qdelta_340m.sh
```

## Model Configurations

- `qdelta_1B.json`: Configuration for 1B parameter model
- `qdelta_340M.json`: Configuration for 340M parameter model

## Implementation Details

The model implementation includes:
- `modeling_qdelta.py`: Main model architecture
- `configuration_qdelta.py`: Model configuration
- `qdelta_rule/`: Custom attention mechanisms and kernels
  - `chunk.py`: Chunkwise computation
  - `fused_recurrent.py`: Fused recurrent operations
  - `common/`: Common utilities for chunk operations
