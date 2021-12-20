import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
from advertorch.attacks import LinfPGDAttack, PGDAttack, GradientSignAttack


def PGD_attack(X, y, model, loss_fn, eps=0.15, eps_iter=0.1, nb_iter=40):
    adversary = PGDAttack(
        model, loss_fn=loss_fn, eps=eps,
        nb_iter=nb_iter, eps_iter=eps_iter, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)

    X = adversary.perturb(X, y)

    return X, y


def LinfPGD_Attack(X, y, model, loss_fn, eps=0.15, eps_iter=0.1, nb_iter=40):
    adversary = LinfPGDAttack(
        model, loss_fn=loss_fn, eps=eps,
        nb_iter=nb_iter, eps_iter=eps_iter, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)

    X = adversary.perturb(X, y)

    return X, y


def FGSM_attack(X, y, model, loss_fn, eps=0.15, eps_iter=0.1):
    adversary = GradientSignAttack(model, loss_fn=loss_fn, eps=eps, targeted=False)

    X = adversary.perturb(X, y)

    return X, y


def generate_adversarials(model, dataloader, output_dir, attack_type='PGD', device='cpu', epsilon = 0.01):
    """
    Generates adversarials for a given model and attack, then saves them to the output folder.

    Args:
        model (nn.Module): The Pytorch model on which we do the adversarial attack
        dataloader (torch.utils.data.dataloader.DataLoader): Dataloader for the input data
        output_dir (str): Paths to your output folder. Place to save the generated adversarials.
        attack_type (str): Which attack you want to run on the model
                        Default: 'PGD'
        device (torch.device): To run your model on the gpu ('cuda') or cpu ('cpu')
                        Default: 'cpu'
    """

    model.eval()
    count_ffhq, count_stylegan3 = 0, 20000 # guarantees ordered naming despite data shuffling

    with tqdm(dataloader) as tepoch:
        for batch, (X, y) in enumerate(tepoch):
            X, y = X.to(device), y.to(device)
            y = torch.unsqueeze(y.to(torch.float32), dim=1)
            loss_fn = F.binary_cross_entropy

            # Make the attack
            if attack_type == 'LinfPGD':
                adv_samp, _ = LinfPGD_Attack(X, y, model, loss_fn, eps_iter=epsilon/10)
            elif attack_type == 'PGD':
                adv_samp, _ = PGD_attack(X, y, model, loss_fn, eps=epsilon, eps_iter=epsilon/10)
            elif attack_type == 'FGSM':
                adv_samp, _ = FGSM_attack(X, y, model, loss_fn, eps=epsilon)
            else:
                raise NotImplementedError("This type of attack is not implemented")

            # Save images
            for img in range(dataloader.batch_size):
                if count_ffhq > 40000 or  count_stylegan3 > 40000:
                    raise ResourceWarning("You exceed the total nr of adversarials of 40000")
                
                if y[img] == 0: # Adv sample for ffhq image
                    save_image(adv_samp[img], f'{output_dir[0]}/{count_ffhq:05}.png')
                    count_ffhq += 1
                elif y[img] == 1: # Adv sample for stylegan3 image
                    save_image(adv_samp[img], f'{output_dir[1]}/{count_stylegan3:05}.png')
                    count_stylegan3 += 1
                else:
                    raise ValueError(f"Expected label value of 0 or 1 but got {y[img]}")

