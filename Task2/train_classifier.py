import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def parse_args():
    parser = argparse.ArgumentParser(description="Train animal image classifier")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset folder (animal_data/)")
    parser.add_argument("--output_dir", type=str, default="./classifier_model",
                        help="Directory to save trained model")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for DataLoader (set 0 for Windows)")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of dataset for validation")
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transformations
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # Load the dataset
    full_dataset = datasets.ImageFolder(args.data_dir, transform=data_transforms["train"])
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    # Separation train/val
    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=args.val_split,
        stratify=full_dataset.targets,
        random_state=42
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    val_dataset.dataset.transform = data_transforms["val"]

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers),
        "val": DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers)
    }

    # ResNet50 model
    model = models.resnet50(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    # Training
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # Saving the model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "animal_classifier.pth"))
    print("Model saved!")

    # Saving classes
    with open(os.path.join(args.output_dir, "classes.txt"), "w") as f:
        for cls in class_names:
            f.write(cls + "\n")
    print("Classes saved!")

    # Validation score
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloaders["val"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    main()
