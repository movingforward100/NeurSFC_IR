# Neural Space-filling Curves for Image  Restoration Tasks

Please refer to the official pytorch implementation (NeurSFC, ECCV 2022) for conda env development.

## Requirements

First, create a new virtual environment (conda). Then,

```bash
pip install -r requirements.txt
```
Please follow https://pytorch.org/get-started/locally/ to install **pytorch** and https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html to install **torch_geometric** properly.

Our code is tested under **Python 3.9.13**, **pytorch 1.11.0**, **cuda 11.3**, and **torch_geometric(pyg) 2.0.4**.

Our specific installation follows:

```bash
cd path/to/project
conda create -n neuralsfc python=3.9.13
conda activate neuralsfc
pip install -r requirements.txt
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg
```

## Training  

### Step 1: Prepare Dataset

Tasks: 
  LLIE:
    LOLv1:
  Deblurring:
    GoPro: 
  SR:
    XXX:
  Dehazing:
    XXX


### Step 2: Pre-processing

- Calculate centroids: 
      - crop training images: run python preprocessing/crop.py;
      - centroids calculation: run python preprocessing/cal_centroid.py
- Set normalize_lo and normalize_hi for configuration file (necessary when e_class = 'lzwl')
      - run python preprocessing/cal_p5_p95.py --cfg configs/IR/xxx/xxx.yml
      - set normalize_lo and normalize_hi in configs/IR/xxx/xxx.yml; 

- Please note normalize_lo should be slightly lower than p5 and normalize_hi should be slightly higher than p95. I have done this part for GoPro and LOLv1 datasets.




To Terry: I need you to check: 

- Is there incorrect settings in .yml files?
- check data.py， model.py, and train.py under neuralsfc folder? Please note my plan is to do validations on test subset. In model.py, this version aims to work equally to the original repo.
- check utils/lzw.py. I did some changes here.
- run each .yml for several epoches (e.g., 3 epoches) to check if the output is as expected. 



