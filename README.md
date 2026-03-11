# CloudMamba

## File Structure

```
cloudmamba/
├── train.py                  # Training entry point
├── eval.py                   # Inference and evaluation entry point
├── config.py                 # Training arguments
├── models/
│   ├── model_zoo.py          # Model factory
│   └── sseg/                 # Model implementations
│       ├── cloudmamba.py     # CloudMamba (proposed)
│       ├── cloudnet.py
│       ├── cdnetv2.py
│       ├── hrcloudnet.py
│       ├── swinunet.py
│       └── rdunet.py
├── utils/
│   ├── cloud_dection.py      # Dataset loader
│   ├── trainers.py           # Training loop
│   ├── metric.py             # Evaluation metrics
│   └── loss.py               # Loss functions
├── data/
│   ├── train/  image/ + gt/
│   ├── val/    image/ + gt/
│   └── test/   image/ + gt/
├── checkpoints/              # Saved weights (auto-created)
└── results/                  # Prediction outputs (auto-created)
```

## Training

```bash
# Edit model_name in train.py, then:
python train.py
```

## Evaluation

```bash
python eval.py 
