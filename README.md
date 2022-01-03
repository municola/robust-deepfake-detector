# Fake image detectors are worse than you think [[PDF](https://github.com/municola/robust-deepfake-detector/blob/master/reports/Report.pdf)]
## Abstract
The recently published synthetic image generator StyleGAN3~\cite{karras2021stylegan3} creates strikingly realistic synthetic images. To prevent such images from being misused e.g. as deepfakes, a synthetic image detection contest was held, in which the PoliMi Team~\cite{mandelli2020training} outperformed others with impressive performance. Hence it was concluded that state-of-the-art synthetic image detectors are effective at detecting fake images in an open-world setting. We argue that in a truly open-world setting, an adversary will not challenge these discriminators with normal synthetic samples. Instead, he will modify these samples with adversarial perturbations and make use of adversarial sample transferability to launch successful attacks even against unknown black-box model architectures. In this work we demonstrate that using simple model architectures and the transferability principle we can perturb images enough to notably deteriorate the quality of the best-performing discriminator model in the image detection contest. Thus, we highlight that current state-of-the-art synthetic image detectors lose their discriminative power in a truly open-world setting and are unable to detect fake images crafted by an adversary. We then design our own synthetic image detector and show that we can improve adversarial robustness to black-box attacks using adversarial training. We advocate for the consideration of such adversarial robustness when designing synthetic image generators in order to reliably apply them to real-life situations

## Setup
In order to run any file, please follow the Installation intructions in the *Installation* section.<br>
Here is a short description of the most important files:

-  ```train.py```: Run this file to train a model on the normal data set.
-  ```adversarial_training.py```: Run this file to use adversarial training on a model.
-  ```generate_adversarials.py```: Run this file to generate the adversarial test sets.
-  ```eval.py```: With this file you can evaluate a model on a given test set. We report ROC AUC and Accuracy metrics.

For each of these main four files a config-file is used, that describes all necessary parameters. In order to give you an intuition what parameters would be expected we have created sample config files in the ```/config``` folder. The used config file for any run is however the ```config.yaml``` file in the main folder.<br>

For any further information we ask you to read through the comments in the code and the [report](https://github.com/municola/robust-deepfake-detector/blob/master/reports/Report.pdf). If it is still unclear you can write us an E-Mail or open a new issue.

## Results
See also the ```/eval_results``` folder for data that can be used with ```eval.py``` to re-create this table.

| Model    | test set | AUC      | Acc    |
|----------|----------|----------|--------|
| Polimi   |  normal  | 0.991053 | 0.7011 |
|          |   adv1   | 0.792699 | 0.7003 |
|          |   adv2   | 0.716880 | 0.6211 |
|          |   adv3   | 0.931072 | 0.7169 |
|          |          |          |        |
| Watson   |  normal  | 0.992820 | 0.9464 |
|          |   adv1   | 0.981916 | 0.9182 |
|          |   adv2   | 0.898157 | 0.7183 |
|          |   adv3   | 0.988455 | 0.9394 |
|          |          |          |        |
| Sherlock |  normal  | 0.997113 | 0.9688 |
|          |   adv1   | 0.996207 | 0.9696 |
|          |   adv2   | 0.980525 | 0.8475 |
|          |   adv3   | 0.996076 | 0.9608 |
|          |          |          |        |
| Moriarty |  normal  | 0.904720 | 0.5334 |
|          |   adv1   | 0.158173 | 0.1975 |
|          |   adv2   | 0.078968 | 0.1890 |
|          |   adv3   | 0.085589 | 0.1925 |

*normal*: normal testset using StyleGAN3 and FFHQ unseen images. <br>
*adv1*: adversarial test set with attack FGSM with eps=0.01. <br>
*adv2*: adversarial test set with attack PGD with eps=0.05. <br>
*adv3*: adversarial test set with attack PGD with eps=0.01. <br>

## Installation

1.  ```git clone https://github.com/municola/robust-deepfake-detector.git```
2. We expect the file structure in the local repo as denoted below, it is vital to follow this to work properly.
3. Install the necessary dependencies in the section below (*Installing dependencies*)
4. Finish. 

If you run into issues you may find a fix in the sections below. Otherwise please open a new issue and let us know.

### Expected file structure
Follow this file structure for you weights, data, folders and checkpoints. <br>

- data
  - test
  - test_adv_v1
  - test_adv_v2
  - test_adv_v3
  - train
  - val
- models
  - checkpoints
- eval_results
- polimi
  - weights
- stylegan3

### Installing dependencies

1. ```pip install requirements.txt```

2. Download the model weights for Polimi from [this link](https://www.dropbox.com/s/g1z2u8wl6srjh6v/weigths.zip) unzip them into the ```polimi/weights``` folder.
```bash
wget https://www.dropbox.com/s/g1z2u8wl6srjh6v/weigths.zip
unzip weigths.zip -d path/to/this/repo/polimi/weights
```

3. Download the weights for our models (Moriarty, Watson, Sherlock) from [this link](https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN?path=%2Fcheckpoints) and place them into the ```models/checkpoints``` folder.
```bash
wget https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN/download?path=%2Fcheckpoints&files=moriaty.pt
wget https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN/download?path=%2Fcheckpoints&files=watson.pt
wget https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN/download?path=%2Fcheckpoints&files=sherlock.pt
```

4. Download the data and unzip the different data folders into the ```data/``` folder structure.

- [test](https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN?path=%2Ftest)
- [test_adv_v1](https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN?path=%2Ftest_adv_v1)
- [test_adv_v2](https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN?path=%2Ftest_adv_v2.zip)
- [test_adv_v3](https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN?path=%2Ftest_adv_v3.zip)
- [train](https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN?path=%2Ftrain)
- [val](https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN?path=%2Fval)

5. Finish.

#### Installing and running advertorch
The package ```advertorch``` is used to launch the adversarial attacks. Its most recent pip release contains legacy code from the ```pytorch``` package. Either directly install and compile it from the [source](https://github.com/BorealisAI/) or perform the following steps:

1. Add the ```zero_gradients(x)``` function to ```advertorch/attacks/utils.py``` (Careful: do this to the version installed in your relevant conda environment!). For more information see this [thread](https://discuss.pytorch.org/t/from-torch-autograd-gradcheck-import-zero-gradients/127462).
```
def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
```

2. Replace line 14 in ```advertorch/attacks/fast_adaptive_boundary.py``` with: <br>
```from advertorch.attacks.utils import zero_gradients```.

#### Running PoliMi model for CPU (& "old" GPU)

1. Follow approach presented by authors in ```polimi/README.md```
2. If it doesn't work to resolve the environment, one can try
```
conda create -n polimi python=3.8.12
conda activate polimi
conda install pytorch=1.6.0 torchvision=0.10.1 -c conda-forge
conda install albumentations=0.5.2 efficientnet-pytorch=0.6.3 -c conda-forge
pip install pytorchcv=0.0.58 
#conda install tqdm, scikit-learn and anything else needed to run eval.py
```
This was a workable setup used that resembled closest the PoliMi author's environment. It might be unnecessarily specific and probably most environments will also work properly with default install versions as below.

3. Download weights.zip, unzip and place in ```polimi/weights``` folder.
4. Set ```model_name``` and other params in ```config.yaml``` accordingly.
5. Run ```eval.py``` and obtain predictions in ```/eval_results``` folder.

#### Running PoliMi model for 3090 GPU

The following conda environment should be enough to properly run PoliMi for most cases. Modify as needed to make it work for the CPU case.
```bash
conda create --name polimi
conda activate polimi
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install albumentations
pip install efficientnet_pytorch
pip install pytorchcv
```
Then follow the remaining steps as above.
