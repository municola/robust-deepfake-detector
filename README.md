# Fake image detectors are worse than you think [[PDF](https://github.com/municola/robust-deepfake-detector/blob/master/reports/Report.pdf)]
The recently published synthetic image generator StyleGAN3  generates  state  of  the  art  human-like  images.  In  orderto  not  have  these  images  used  maleficently  as  deepfakes,they  launched  a  fake  image  detection  contest  in  which  thePoliMi  team  won  with  staggering  performance.  Hence,they  conclude  that  state  of  the  art  detection  algorithms  areeffective  at  detecting  fake  images  in  a  open-world  setting.In  this  work  we  argue  that  in  a  truly  open-world  settingthe  adversary  will  not  stop  after  having  generated  syntheticimages. Instead, the attacker will add adversarial perturbationsto the images, making use of adversarial transferability whichimplies  that  a  targeted  perturbation  on  a  self-made  detectorwill have high chances to also work on the black-box syntheticimage detector.We  show  that  with  a  simple  detector  architecture  we  canperturb  images  enough  to  deteriorate  the  performance  of  thestate  of  the  art  synthetic  image  detector.  Hence,  we  arguethat  in  a  truly  open-world  setting  the  current  state  of  the  artdetectors are unable to detect fake images by an adversary.We  then  design  our  own  synthetic  image  detector  and  showthat  using  adversarial  training,  we  can  make  it  much  morerobust to adversarial black-box perturbations. We conclude byadvocating the importance of adversarial training as well as ro-bust architectures when designing synthetic image generatorsin order to work in a real-life setting.

## Setup
In order to run any file, please follow the Installation intructions in the *Installation* section.<br>
Here is a small description of the most important files:

- train.py (Run this file to train a model on the normal data set)
- adversarial_training.py (Run this file to use adversarial training on a model)
- generate_adversarials.py (Run this file to generate the adversarial test sets)
- eval.py (With this file you can evaluate a model on a given test set. We report AUCROC and Accuracy)

For each of these main four files a config-file is used, that describes all necerssary parameters. In order to give you an intuition what parameters would be expected we have created sample config files in the /config folder. The used config file for any run is however the config.yaml file in the main folder.<br>

For any further information we ask you to read through the comments in the code and the [report](https://github.com/municola/robust-deepfake-detector/blob/master/reports/Report.pdf). If it is still unclear you can write us an E-Mail or open a new issue.

## Results

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

normal: normal testset with stylegan3 and ffhq <br>
adv1: adversarial attack FGSM with eps=0.01 <br>
adv2: adversarial attack PGD with eps=0.05 <br>
adv3: adversarial attack PGD with eps=0.01 <br>


## Installation

1.  ```git clone https://github.com/municola/robust-deepfake-detector.git```
2. We expect the file structure denoted below in section (*Expeted file structure*). Please follow it.
3. Install the necessary dependencies in the section below (*Intalling dependencies*)
4. Finish. 

If you run into issues you may find a fix in the sections below. Otherwise please open a new issue and let us know.

### Expected file structure:
Follow this filestructure for you weights, data, folders and checkpoints<br>
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

2. Download the model weights for Polimi [this link](https://www.dropbox.com/s/g1z2u8wl6srjh6v/weigths.zip) unzip them into the polimi/weights folder
```bash
wget https://www.dropbox.com/s/g1z2u8wl6srjh6v/weigths.zip
unzip weigths.zip -d path/to/this/repo/polimi/weights
```

3. Download the weights for our models (moriarty, sherlock, watson) from [this link](https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN?path=%2Fcheckpoints) and place them into models/checkpoints
```bash
wget https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN/download?path=%2Fcheckpoints&files=moriaty.pt
wget https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN/download?path=%2Fcheckpoints&files=watson.pt
wget https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN/download?path=%2Fcheckpoints&files=sherlock.pt
```

4. Download the data and unzip the differnt data folders into data/.

- [train](https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN?path=%2Ftrain)
- [val](https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN?path=%2Fval)
- [test](https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN?path=%2Ftest)
- [test_adv_v1](https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN?path=%2Ftest_adv_v1)
- [test_adv_v2](https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN?path=%2Ftest_adv_v2.zip)
- [test_adv_v3](https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN?path=%2Ftest_adv_v3.zip)

5. Finish.

#### Installing Advertorch from source
The current pip release does not contain the newest code changes. Either directely install from [source](https://github.com/BorealisAI/) or perfrom the following steps:

1. Add the zero_gradients(x) function to advertorch/attacks/utils.py (Carful: Do this in your corresponding conda environment!). See this [thread](https://discuss.pytorch.org/t/from-torch-autograd-gradcheck-import-zero-gradients/127462).
```
def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
```

2. Replace line 14 in advertorch/attacks/fast_adaptive_boundary.py with: 
```from advertorch.attacks.utils import zero_gradients```

#### Running Polimi for CPU (& "old" GPU)

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
4. Set model_name and other params in config.yaml accordingly
5. Run eval.py and obtain predictions in eval_results folder, manually save ROC plot

#### Running Polimi for 3090

The following conda envrionment should be enough to just run polimi.
```bash
conda create --name polimi
conda activate polimi
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install albumentations
pip install efficientnet_pytorch
pip install pytorchcv
```
