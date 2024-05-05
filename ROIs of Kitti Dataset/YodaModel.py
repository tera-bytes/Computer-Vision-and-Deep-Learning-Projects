import torch
import cv2
import torch.nn as nn
import argparse
from torchvision import models, transforms
from KittiDataset import KittiDataset
from KittiAnchors import Anchors

save_ROIs = True
def test_yoda(model, roi_batch):
    model.eval()
    with torch.no_grad():
        outputs = model(roi_batch)
        softmax = nn.Softmax(dim=1)
        probs = softmax(outputs)
        _, preds = torch.max(probs, 1)
    return preds

#Subdivide kitti images into set of rois, save coordinates of bounding boxes for each roi
#Build a batch from ROIs ^ and pass them through classifier
#for every roi classified as 'car', calculate iou score with original kitti img
def main():
    print('running KittiToYoda ...')

    label_file = 'labels.txt'

    argParser = argparse.ArgumentParser()

    argParser.add_argument('-d', metavar='display', type=str, help='[y/N]')
    argParser.add_argument('-m', metavar='mode', type=str, help='[train/test]')
    argParser.add_argument('-cuda', metavar='cuda', type=str, help='[y/N]')

    args = argParser.parse_args()

    show_images = False
    if args.d != None:
        if args.d == 'y' or args.d == 'Y':
            show_images = True

    training = True
    if args.m == 'test':
        training = False

    use_cuda = False
    if args.cuda != None:
        if args.cuda == 'y' or args.cuda == 'Y':
            use_cuda = True

    labels = []

    device = 'cpu'
    if use_cuda == True and torch.cuda.is_available():
        device = 'cuda'
    print('using device ', device)

    # Model setup
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to match ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = models.resnet18()
    model.to(device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load('step3_train.pth', map_location=torch.device('cpu')))


    dataset = KittiDataset('./data/Kitti8', training=training)
    anchors = Anchors()

    i = 0
    all_ious = []
    iou_threshold = 0.02 # Set a threshold for IoU

    for item in enumerate(dataset):
        image = item[1][0]
        label = item[1][1]
        # print(i, idx, label)

        idx = dataset.class_label['Car']
        car_ROIs = dataset.strip_ROIs(class_ID=idx, label_list=label)

        anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)

        ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)
        roi_tensors = [transform(roi) for roi in ROIs]
        roi_batch = torch.stack(roi_tensors).to(device)
        preds = test_yoda(model, roi_batch)
        # IoU calculations
        for j, pred in enumerate(preds):
            if pred == 1:
                predicted_box = boxes[j]
                max_iou = anchors.calc_max_IoU(predicted_box, car_ROIs)
                if max_iou > iou_threshold:
                    print("Going through data")
                    all_ious.append(max_iou)
                    if show_images:
                        # Draw bounding box on the image
                        pt1 = (predicted_box[0][1], predicted_box[0][0])
                        pt2 = (predicted_box[1][1], predicted_box[1][0])
                        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)
        if show_images:
            cv2.imshow('Detected Cars', image)
            cv2.waitKey(0)

    mean_iou = sum(all_ious)/len(all_ious)
    print(f"Mean IoU for detected 'Car' ROIs: {mean_iou}")

###################################################################

main()


