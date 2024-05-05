import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import models, transforms
from KittiROI_Dataset import KittiROIDataset
def train(n_epochs, optimizer, model, loss_fn, train_loader, test_loader, scheduler, device, p_name):
    # Training and Validation Loop

    train_losses, val_losses = [], []
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_trainloss = running_loss / len(train_loader)
        scheduler.step(avg_trainloss)
        train_losses.append(avg_trainloss)

        # Validation
        model.eval()
        correct = 0.0
        samples = 0.0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                samples += labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        accuracy = correct / samples
        val_losses.append(val_loss / len(test_loader))
        print(
            f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}, Accuracy: {accuracy:.4f}")


    # Plotting training and validation loss
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(p_name)
    plt.show()

def plot_confusion_matrix(cm, class_names):
    """
    Plots a confusion matrix using Seaborn's heatmap.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def test(model, dataloader, loss_fn, device):
    model.eval()
    correct = 0
    samples = 0
    y_true = []
    y_pred = []
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            samples += labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = correct / samples
    avg_loss = val_loss / len(test_loader)
    print(f'Test Accuracy: {accuracy:.4f}, Test Loss: {avg_loss:.4f}')
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, ['NoCar','Car'])

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Training the Kitti8 ROIs Dataset')
    parser.add_argument('-e', type=int, default=25, help='Number of epochs')
    parser.add_argument('-p', type=str, default='loss_step3.png', help='Plot save path')
    parser.add_argument('-s', type=str, default='step3_train.pth', help='Model save path')
    parser.add_argument('-m', type=str, help='train or test')

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model setup
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to match ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dataset = KittiROIDataset(label_file='./data/Kitti8_ROIs/train/labels.txt', img_dir='./data/Kitti8_ROIs/train', transform=transform)
    test_dataset = KittiROIDataset(label_file='./data/Kitti8_ROIs/test/labels.txt', img_dir='./data/Kitti8_ROIs/test',transform=transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # Training
    if(args.m == 'train'):
        train(args.e, optimizer, model, loss_fn, train_loader, test_loader, scheduler, device, args.p)
        # Save the trained model
        torch.save(model.state_dict(), args.s)
    #Testing
    elif(args.m == 'test'):
        model.load_state_dict(torch.load(args.s))
        test(model,test_loader,loss_fn,device)