# Fake image detectors are worse than you think

Expected file structure:

- data
  - test
  - train
  - val
- detective
  - checkpoints
- eval_results
- polimi
  - weights
- stylegan3

## Installation

### Running attacks/generate_adversarials

1. ```pip install advertorch```
3. Add the zero_gradients(x) function to advertorch/attacks/utils.py (Carful: Do this in your corresponding conda environment!). See this [thread](https://discuss.pytorch.org/t/from-torch-autograd-gradcheck-import-zero-gradients/127462).
```
def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
```
3. Replace line 14 in advertorch/attacks/fast_adaptive_boundary.py with: 
```from advertorch.attacks.utils import zero_gradients```

### Running Polimi for CPU (& "old" GPU)

1. Follow approach presented by authors in polimi/README.md
2. If it didn't work to resolve environment (like for me: Alex), try
```
conda create -n polimi python=3.8.12
conda activate polimi
conda install pytorch=1.6.0 torchvision=0.10.1 -c conda-forge
conda install albumentations=0.5.2 efficientnet-pytorch=0.6.3 -c conda-forge
pip install pytorchcv=0.0.58 
#conda install tqdm, scikit-learn and anything else needed to run eval.py
```
Probably also works with default install versions like below
but this was my workable setup that resembled closest the author's environment.

3. Download weights.zip (also in Polybox), unzip and place in polimi folder
4. Set model_name in config.yaml accordingly
5. Run eval.py and obtain results in eval_results folder

### Running Polimi for 3090

1. Create and activate the conda environment
```bash
conda create --name polimi
conda activate polimi
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install albumentations
pip install efficientnet_pytorch
pip install pytorchcv
```

2. Download the model's weights from [this link](https://www.dropbox.com/s/g1z2u8wl6srjh6v/weigths.zip) and unzip the file under the main folder
```bash
wget https://www.dropbox.com/s/g1z2u8wl6srjh6v/weigths.zip
unzip weigths.zip
```

## Run in Visual Studio Code

1. Open Main.py from the root folder
2. Run it
