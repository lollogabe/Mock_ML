# Running on Kaggle

This guide explains how to run this project on Kaggle using their free GPUs.

## Step 1: Create a New Kaggle Notebook

1.  Go to [Kaggle Notebooks](https://www.kaggle.com/code) and click **New Notebook**.
2.  Enable Internet access (Settings -> Internet -> On).
3.  Ensure you have a GPU accelerator selected (Settings -> Accelerator -> GPU T4 x2 or P100).

## Step 2: Setup Environment and Clone Repo

In the first cell of your notebook, run the following to clone the project directly into the Kaggle environment:

```bash
# Clone the repository (replace with your actual GitHub URL if you uploaded it there)
!git clone https://github.com/your-username/Mock_ML.git

# Move into the project directory
%cd Mock_ML

# Install the required packages
!pip install -r requirements.txt
```

*(Note: If you uploaded the folder as a Kaggle Dataset instead, it will be in `../input/dataset-name`. You can copy it to `./` or run scripts directly).*

## Step 3: Run the Pipeline

You can run the existing scripts directly from cells. Kaggle's `/kaggle/working` directory functions perfectly with the relative paths in the scripts.

```python
# 1. Preprocess / Download data from CERN
!python scripts/preprocess.py --group 37

# 2. Train the model (will automatically use the Kaggle GPU)
!python scripts/train.py --config configs/config.yaml

# 3. Evaluate the model
!python scripts/evaluate.py --config configs/config.yaml --checkpoint checkpoints/ae_best.pt
```

## Step 4: Accessing Logs and Checkpoints

After training, all output files will safely be in the `Mock_ML/checkpoints` and `Mock_ML/logs` directories. You can browse them in the Kaggle UI (right-hand sidebar under 'Output') or download them.

If you want to use the codebase interactively inside the Kaggle Notebook instead of via scripts:
```python
import sys
import os
sys.path.append(os.path.abspath('.')) 

from src.model import build_model
from src.data_loader import build_dataloaders
```
