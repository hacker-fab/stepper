import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import pattern_net
import numpy as np
from datasets import AugmentedDataset, visualize_transformation
import os

if __name__ == '__main__':
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    num_copies = 5  # Number of augmented samples per image
    image_size = 224  # Resize images to 224x224

    # Transforms for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),                # Resize images to 224x224
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),  # Random color jitter
        transforms.RandomHorizontalFlip(),            # Random horizontal flip
        transforms.RandomRotation(10),               # Random rotation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    # Load custom dataset
    train_dir = "/home/louis/project/pattern_net_data/train"
    test_dir = "/home/louis/project/pattern_net_data/test"

    base_train_dataset = datasets.ImageFolder(root=train_dir)
    base_test_dataset = datasets.ImageFolder(root=test_dir)

    # Visualize a sample transformation
    # visualize_transformation(image_path="/home/louis/project/pattern_net_data/train/pattern/Copy of Chip507_lab3_afterHF1.png", transform=transform)

    train_dataset = AugmentedDataset(base_dataset=base_train_dataset, transform=transform, num_copies=num_copies)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    model = pattern_net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("Starting training loop...")
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, targets) in enumerate(train_loader):

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Evaluation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    
    # Save the trained model
    model_folder = "/home/louis/project/stepper/src/pattern_net/model"
    torch.save(model.state_dict(), os.path.join(model_folder, "pattern_net_model.pth"))
    print("Model saved successfully.")
