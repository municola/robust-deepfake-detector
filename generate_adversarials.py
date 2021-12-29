import os
import torch
import yaml
import argparse
from utils import load_model, load_data
from attacks import generate_adversarials


def main():
    """Loads and calls on generate_adversarials for a given test set"""

    # Config arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", default="config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))

    batch_size = config['batch_size']
    model_name = config['model_name']
    adversarial_attack_type = config['adversarial_attack_type']
    test_path = config['test_path']
    test_adv_path = config['test_adv_path']
    epsilon = config['adversarial_eps']
    seed = config['seed']
    num_workers = config['num_workers']

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Model + data
    model, _, _, _ = load_model(model_name, config, device)
    test_dataloader = load_data(test_path, batch_size, model_name, seed, num_workers)

    # Generate directory to store adv. samples that satisfies assert statement.
    # Make sure that the directory doesn't exist in-place already, i.e.
    # move or rename any existing test_adv directory for some trained model
    adv_path_ffhq = test_adv_path + '/ffhq'
    adv_path_stylegan3 = test_adv_path + '/stylegan3'
    os.makedirs(adv_path_ffhq, exist_ok=False)
    os.makedirs(adv_path_stylegan3, exist_ok=False)
    print(f"\nCreated directories for storage:\n{adv_path_ffhq} \n{adv_path_stylegan3}")

    print(f"\nGenerating adversarial samples for {model_name} on test set...")
    generate_adversarials(
        model = model, 
        dataloader = test_dataloader,
        output_dir = [adv_path_ffhq, adv_path_stylegan3],
        attack_type = adversarial_attack_type,
        device = device,
        epsilon = epsilon
    )


if __name__ == "__main__":
    main()
