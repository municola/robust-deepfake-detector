from attacks import *

def main():
    # TODO: Have a separate File for Data Loaders
    batch_size = 10
    user = "Nici"
    path_model='./our-detector/checkpoints/checkpoint.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = get_path(user, 'test')
    model, _, _ = load_model('DetectorNet', path_model)

    transform = transforms.Compose([
        transforms.ToTensor()
        # Maybe Normalize !!!!
    ])

    # 1: Fake, 0: Real
    test_data = torchvision.datasets.ImageFolder(root=path,transform = transform)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Generate the Adverserials
    path = os.path.dirname(path) + '/adversarials'
    generateAdverserials(
        model = model, 
        data_loader = test_data_loader, 
        output_dir = path,
        attack_type='LinfPGD',
        device = device
    )

if __name__ == "__main__":
    main()
