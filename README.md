# Fake image detectors are worse than you think

Expected file structure:

- data
  - test
  - train
  - val
- detective
  - checkpoints
- polimi
- stylegan3

## Installation

### Attacks

1. pip install advertorch
2. Add the zero_gradients(x) function to advertorch/attacks/utils.py (Carful: Do this in your corresponding conda envrionment!!s)
3. Replace line 14 in advertorch/attacks/fast_adaptive_boundary.py with: "from advertorch.attacks.utils import zero_gradients"

### Installation Polimi for CPU & "old" GPU

1. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate gan-image-detection
```

2. Download the model's weights from [this link](https://www.dropbox.com/s/g1z2u8wl6srjh6v/weigths.zip) and unzip the file under the main folder

```bash
wget https://www.dropbox.com/s/g1z2u8wl6srjh6v/weigths.zip
unzip weigths.zip
```

### Installation Polimi for 3090

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
