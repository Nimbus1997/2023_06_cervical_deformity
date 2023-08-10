# 실행하려면,
#   1) CHANGE에서 load할 pretrained model (pth) 가져오기, 저장할 경로(save_dir), 이용할 data 경로 (data_dir) 수정
#   2) python 파일명.py 로 실행
#   3) use GPU  

import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Function
import cv2
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torchvision import models
import torchvision
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import pdb
from sklearn.metrics import confusion_matrix # edit 3
import seaborn as sns # edit 3

# Load the pretrained ResNet-50 model
model = models.efficientnet_b5(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4) 

# CHANGE -----------------------------------------------------------------
# image_path = '/root/jieunoh/cervical_deformity/data/Cancer/C_P0904_01_CS03_H_M_1_adenocarcinoma.jpg'
root_dir = '/root/jieunoh/cervical_deformity/result/efficientnet_b5/efficientnet_b5_allclass_0629'
data_dir = "/root/jieunoh/cervical_deformity/data"
device = torch.device("cuda:0")
# class_list = ["Cancer", "normal"]
class_list = ["Cancer", "HSIL", "LSIL", "normal"]
seed =42
# count = 3  # number of CMA images to produce for each class

# CHANGE (end) ----------------------------------------------------------------
# Load the saved model parameters from the .pth file
model.load_state_dict(torch.load(os.path.join(root_dir,"model_fold0_best.pth")))
model.eval()

# save path 만들기
save_dir = os.path.join(root_dir, 'CAM')
for class_ in class_list:
    pathh = os.path.join(save_dir, class_)
    if not os.path.isdir(pathh):
        os.makedirs(pathh)


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.feature_map = None
        self.gradient = None
        self.model.features[-1].register_forward_hook(self.save_feature_map)
        self.model.features[-1].register_full_backward_hook(self.save_gradient)

    def save_feature_map(self, module, input, output):
        self.feature_map = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def __call__(self, input_tensor):
        output = self.model(input_tensor)
        self.model.zero_grad()

        target_class = output.argmax()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1

        output.backward(gradient=one_hot, retain_graph=True)

        gradient_mean = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((gradient_mean * self.feature_map).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.size()[2:], mode='bilinear', align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize between 0-1

        return cam.cpu().squeeze().detach().numpy() , target_class

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 이걸안해야 색이 잘 나옴
    image = cv2.resize(image, (512, 512))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image)
    return image

model = model.to(device)
gradcam = GradCAM(model)

# # 데이터 전처리
pretransform = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
# 데이터셋 불러오기
dataset = datasets.ImageFolder(root=data_dir, transform=pretransform)
kfold = KFold(n_splits=3, shuffle = True, random_state=seed)
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    valloader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=val_idx)
    for i, (images, labels) in enumerate(valloader):
        images = images.to(device)
        labels = labels.to(device)
        cam, predict_label = gradcam(images)

        heatmap = cv2.resize(cam, (images.shape[2], images.shape[3]))
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)

        # Convert the input image and heatmap to RGBA
        images = np.uint8(255 * images.cpu().squeeze().permute(1, 2, 0))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        alpha = 0.2  # Adjust the transparency level as desired
        heatmap = np.uint8(heatmap * alpha)

        # Overlay the heatmap on the input image
        images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
        overlaid_image = cv2.addWeighted(images, 1, heatmap, 1, 0)
        # overlaid_image = cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(save_dir, class_list[labels],f'{i}_prediction_{class_list[predict_label]}_overlay.jpg')
        cv2.imwrite(save_path, overlaid_image)
    break


