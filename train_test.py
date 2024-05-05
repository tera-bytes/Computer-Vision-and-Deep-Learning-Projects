import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torchvision import models, transforms
from get_dataset import PetDataset
import torch
import numpy as np
import os
import cv2
import time

def test(model, dataloader, show_img, device):
    model.eval()
    distances = []
    images_shown = 0
    total_time = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            start = time.time()
            outputs = model(images)
            end = time.time()
            total_time = (end-start)*1000
            if(show_img == 'Y'):
                for idx, (pred, true) in enumerate(zip(outputs, labels)):

                    # Get the original image (without transformation)
                    original_image_path = dataloader.dataset.img_labels[idx].split(',')[0]
                    original_image_path = os.path.join(dataloader.dataset.img_dir, original_image_path)
                    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
                    if original_image is None:
                        raise RuntimeError(f"Failed to load image: {original_image_path}")

                    # Scale predicted coordinates back to original image size
                    pred_x, pred_y = pred.cpu().numpy()
                    pred_x *= original_image.shape[0]
                    pred_y *= original_image.shape[1]

                    # Draw a circle at the predicted location
                    cv2.circle(original_image, (int(pred_x), int(pred_y)), radius=5, color=(0, 0, 255), thickness=-1)

                    # Display the image
                    cv2.imshow(f"Predicted Nose Location {images_shown + 1}", original_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            images_shown += 1
            # Calculate Euclidean distances
            for pred, true in zip(outputs, labels):
                distance = torch.sqrt(torch.sum((pred - true) ** 2)).item()
                distances.append(distance)

    # Calculate statistics
    min_distance = np.min(distances)
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    std_distance = np.std(distances)
    avg_time = total_time/images_shown
    print(f"Average time: {avg_time}")
    print(f"Minimum Distance: {min_distance}")
    print(f"Mean Distance: {mean_distance}")
    print(f"Maximum Distance: {max_distance}")
    print(f"Standard Deviation: {std_distance}")

    return min_distance, mean_distance, max_distance, std_distance

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

        print(
            f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_losses[-1]:.4f}")

    # Plotting training and validation loss
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Plot')
    plt.savefig(p_name)
    plt.show()


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Training the Pet Noses Dataset')
    parser.add_argument('-e', type=int, default=25, help='Number of epochs')
    parser.add_argument('-p', type=str, default='loss_plot.png', help='Plot save path')
    parser.add_argument('-s', type=str, default='trained_model.pth', help='Model save path')
    parser.add_argument('-b', type=int, default=32, help='Number of batches')
    parser.add_argument('-m', type=str, help='train or test')
    parser.add_argument('-show', type=str,  help='Y/N')

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model setup
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to match ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dataset = PetDataset(annotations_file='train_noses.3.txt', img_dir='images', transform=data_transform)
    test_dataset = PetDataset(annotations_file='test_noses.txt', img_dir='images',transform=data_transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=False)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer,mode='min')
    # Training
    if(args.m == 'train'):
        train(args.e, optimizer, model, loss_fn, train_loader, test_loader, scheduler, device, args.p)
        # Save the trained model
        torch.save(model.state_dict(), args.s)
    #Testing
    elif(args.m == 'test'):
        model.load_state_dict(torch.load(args.s))
        test(model,test_loader,args.show,device)