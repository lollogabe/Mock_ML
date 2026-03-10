# Running on Google Colab

This guide explains how to run this project on Google Colab using a GPU.

## Step 1: Upload Project to Google Drive

1.  Zip your project folder (`Mock_ML`).
2.  Upload it to your Google Drive.
3.  Alternatively, if you have pushed this to GitHub, you can just clone it in Colab (see Step 3).

## Step 2: Open a New Colab Notebook

1.  Go to [colab.research.google.com](https://colab.research.google.com/).
2.  Ensure you are using a GPU runtime:
    -   `Runtime` -> `Change runtime type` -> `Hardware accelerator` -> `GPU` (T4, L4, or A100).

## Step 3: Setup Environment

In a Colab cell, run:

```python
# 1. Mount Google Drive (if using Drive)
from google.colab import drive
drive.mount('/content/drive')

# 2. Navigate to your project folder
# Change the path to where you uploaded the project
%cd /content/drive/MyDrive/Mock_ML

# OR Clone from GitHub
# !git clone https://github.com/your-username/Mock_ML.git
# %cd Mock_ML

# 3. Install dependencies
!pip install -r requirements.txt
```

## Step 4: Run the Pipeline

You can run the existing scripts directly from cells:

```python
# 1. Preprocess / Download data
!python scripts/preprocess.py --group 37

# 2. Train the model
!python scripts/train.py --config configs/config.yaml

# 3. Evaluate the model
# Interactive plots will show up in Colab if you don't use --no-plot
!python scripts/evaluate.py --config configs/config.yaml --checkpoint checkpoints/ae_best.pt
```

## Step 5: Using the Notebook

If you want to use the original notebook in `notebooks/`:

1.  Open `notebooks/DiProfio_Franco_Gabellini_37.ipynb` in Colab.
2.  Add this cell to the top to allow importing from the `src/` folder:

```python
import sys
import os
# Assuming you are in the project root or adjust accordingly
sys.path.append(os.path.abspath('..')) 
```

3.  You can then import your modular code:
```python
from src.model import build_model
from src.data_loader import build_dataloaders
```
