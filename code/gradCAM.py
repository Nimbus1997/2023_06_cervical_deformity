import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torchvision import models
import torchvision
import torch
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import pdb
import cv2
import numpy as np
from tqdm import tqdm


# Load the saved model from the .pth file
model = models.resnet50()
model.fc = nn.Linear(2048, 4)
model.load_state_dict(torch.load('/root/jieunoh/cervical_deformity/result/resnet50/resnet50_allclass_0515/model_fold0.pth'))
model.eval()
now_fold =0
data_dir = "/root/jieunoh/cervical_deformity/data"
resizee =512
class_list = ["Cancer", "HSIL", "LSIL", "normal"]
# Create a directory to save the CAM images
save_dir = '/root/jieunoh/cervical_deformity/result/resnet50/resnet50_allclass_0515'
os.makedirs(save_dir, exist_ok=True)


# Define the target layer for Grad-CAM
target_layer = model.layer4[-1].conv3  # Example: last convolutional layer in layer4
# Define the backward hook to capture gradients
grads = None
def backward_hook(module, grad_input, grad_output):
    # Save the gradients of the target feature map
    global grads
    grads = grad_output[0]

def forward_hook(module, input, output):
    # Save the activations of the target feature map
    global activations
    activations = output


# dataset
pretransform = transforms.Compose([transforms.Resize((resizee,resizee)), transforms.ToTensor()])
dataset = datasets.ImageFolder(root=data_dir, transform=pretransform)
class_labels = dataset.classes

# # 3-fold cross validation을 위한 인덱스 생성
kfold = KFold(n_splits=3, shuffle = True)


# Iterate over each class
for class_now in class_list:
    image_list = os.listdir(os.path.join(data_dir,class_now))
    print(class_now)
    for i, imgg  in tqdm(enumerate(image_list)):
        if i==5: # 5개 예시만
            break

        # Define the hooks to capture the gradients and activations
        grads = None
        activations = None


        hook_handles = []
        hook_handles.append(target_layer.register_backward_hook(backward_hook))
        hook_handles.append(target_layer.register_forward_hook(forward_hook))
        
        # Load and preprocess the input image
        input_image = cv2.imread(os.path.join(data_dir,class_now,imgg))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (512, 512))
        input_tensor = torch.from_numpy(np.transpose(input_image, (2, 0, 1))).float().unsqueeze(0) / 255.0

        # Forward pass the input image through the model
        with torch.no_grad():
            output = model(input_tensor)

        # Calculate the Grad-CAM heatmap using the gradients and activations
        grads = model.layer4.activations.grad
        weights = torch.mean(grads, dim=(2, 3))
        grad_cam = torch.matmul(weights.unsqueeze(1), activations.view(activations.size(0), activations.size(1), -1))
        grad_cam = grad_cam.squeeze(1)
        grad_cam = nn.functional.relu(grad_cam)
        grad_cam /= torch.max(grad_cam)

        # Resize the Grad-CAM heatmap to match the input image size
        grad_cam = torch.nn.functional.interpolate(grad_cam.unsqueeze(0), size=(input_image.shape[0], input_image.shape[1]), mode='bilinear', align_corners=False)
        grad_cam = grad_cam.squeeze().detach().numpy()

        # Normalize the Grad-CAM heatmap to the range [0, 1]
        grad_cam = (grad_cam - np.min(grad_cam)) / (np.max(grad_cam) - np.min(grad_cam))

        # Apply color map to the Grad-CAM heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)

        # Overlay the heatmap on the input image
        overlay = cv2.addWeighted(input_image, 0.7, heatmap, 0.3, 0)

        # Save the overlay image for the class
        save_path_overlay = os.path.join(save_dir, f'class{class_now}_{i}_overlay.jpg')
        cv2.imwrite(save_path_overlay, overlay)

        # Remove the hooks
        for handle in hook_handles:
            handle.remove()