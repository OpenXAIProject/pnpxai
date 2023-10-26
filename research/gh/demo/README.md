# Pnpxai Demo

by GH

## Installation

Please make sure
- your virtual environment is activated
- install demo and its requirements with no dependencies (there are some version issues in "torchcam")

```
python -m venv .venv
source venv/bin/activate
pip install -r requirements.txt --no-deps
pip install -e . --no-deps
```

## Notebooks

Please check notebooks by following order:

1. ./notebooks/setup_for_notebooks.ipynb
2. ./notebooks/explainers.ipynb
3. ./notebooks/client.ipynb
