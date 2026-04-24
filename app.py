import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from PIL import UnidentifiedImageError
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["Normal", "Pneumonia"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model(model_path="model/pneumonia_model.pth"):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except (FileNotFoundError, RuntimeError) as exc:
        raise ValueError(f"Unable to load model from '{model_path}': {exc}") from exc
    model = model.to(device)
    model.eval()
    return model


def predict_image(image_path, model):
    try:
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Unable to load image from '{image_path}': {exc}") from exc
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    return classes[pred.item()]


def main():
    parser = argparse.ArgumentParser(description="Predict pneumonia from a chest X-ray image.")
    parser.add_argument("--image", required=True, help="Path to input image (jpg/png/jpeg).")
    parser.add_argument("--model", default="model/pneumonia_model.pth", help="Path to trained model weights.")
    args = parser.parse_args()

    try:
        model = load_model(args.model)
        result = predict_image(args.image, model)
        print(f"Prediction: {result}")
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
