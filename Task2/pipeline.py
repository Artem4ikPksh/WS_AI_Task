import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Final pipeline: text + image -> boolean")
    parser.add_argument("--ner_model_dir", type=str, required=True,
                        help="Path to trained NER model directory")
    parser.add_argument("--clf_model_dir", type=str, required=True,
                        help="Path to trained classifier model directory")
    parser.add_argument("--text", type=str, required=True,
                        help="Input text")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image")
    return parser.parse_args()

def extract_animal(text, ner_model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(ner_model_dir)
    model = AutoModelForTokenClassification.from_pretrained(ner_model_dir).to(device)
    inputs = tokenizer(text.split(), is_split_into_words=True,
                       return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    id2label = model.config.id2label

    animals = []
    for token, pred_id in zip(tokens, predictions):
        if token not in ["[CLS]", "[SEP]", "[PAD]"]:
            label = id2label[pred_id]
            if label != "O":
                animals.append(token)
    return " ".join(animals) if animals else None

def classify_image(image_path, clf_model_dir, device):
    # Load class names
    with open(os.path.join(clf_model_dir, "classes.txt"), "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    num_classes = len(class_names)

    # Model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(os.path.join(clf_model_dir, "animal_classifier.pth"),
                                     map_location=device))
    model = model.to(device)
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)

    return class_names[pred.item()]

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    animal_text = extract_animal(args.text, args.ner_model_dir, device)
    animal_image = classify_image(args.image_path, args.clf_model_dir, device)

    print(f"Text animal: {animal_text}")
    print(f"Image animal: {animal_image}")

    result = (animal_text is not None) and (animal_text.lower() == animal_image.lower())
    print(f"Pipeline result: {result}")

if __name__ == "__main__":
    main()
