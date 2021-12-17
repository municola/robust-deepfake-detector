from attacks import *
import torchvision
import yaml
import argparse

def main():
    # TODO: Have a separate File for Data Loaders

    # Config arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", default="config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))
    batch_size = config['batch_size']
    test_path = config['test_path']
    model_name = config['model_name']
    adversarial_attack_type = config['adversarial_attack_type']

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model, _, _, _ = load_model(model_name, config, device)
    model_summary(model) # nr of params

    transform = transforms.Compose([
        transforms.ToTensor()
        # Maybe Normalize !!!!
    ])

    # 1: Fake, 0: Real
    test_data = torchvision.datasets.ImageFolder(root=test_path,transform = transform)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Generate the Adverserials
    path = os.path.dirname('/data/adversarials/test')
    os.makedirs(path, exist_ok=True)
    generateAdversarials(
        model = model, 
        data_loader = test_data_loader, 
        output_dir = path,
        attack_type= adversarial_attack_type,
        device = device
    )

if __name__ == "__main__":
    main()
