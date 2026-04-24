from pathlib import Path

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split


def _resolve_train_root(data_dir: str) -> str:
    base = Path(data_dir)
    candidates = [
        base / "train",
        base / "chest_xray" / "train",
        base / "chest_xray" / "chest_xray" / "train",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)

    raise FileNotFoundError(
        "Could not find a train folder under the provided data_dir. "
        "Expected one of: train/, chest_xray/train/, chest_xray/chest_xray/train/."
    )


def get_dataloaders(data_dir, batch_size=32):
    transform = transforms.Compose([transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224, 0.225])])

    train_root = _resolve_train_root(data_dir)
    dataset = datasets.ImageFolder(root=train_root, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset,[train_size,val_size])

    print("Training Images: ", len(train_dataset))
    print("Validation Images: ", len(val_dataset))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,shuffle=False)
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders("/content/xray/data")