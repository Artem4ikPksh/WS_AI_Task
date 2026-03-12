import argparse
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with trained animal image classifier")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to trained model directory (with animal_classifier.pth)")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image")
    return parser.parse_args()

def main():
    args = parse_args()

    #Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transformations (same as in train_classifier.py)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load class names (stored in ImageFolder)
    # Here you need to have a classes.txt file in model_dir, which contains a list of classes
    classes_file = os.path.join(args.model_dir, "classes.txt")
    if not os.path.exists(classes_file):
        raise FileNotFoundError("classes.txt not found in model_dir. "
                                "Save class names during training.")
    with open(classes_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    num_classes = len(class_names)

    # Model ResNet50
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "animal_classifier.pth"),
                                     map_location=device))
    model = model.to(device)
    model.eval()

    # Loading the image
    image = Image.open(args.image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)

    predicted_class = class_names[pred.item()]
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()
