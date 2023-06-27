# 실행하려면,
#   1) CHANGE에서 load할 pretrained model (pth) 가져오기, 저장할 경로(save_dir), 이용할 data 경로 (data_dir) 수정
#   2) python 파일명.py 로 실행

import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Function
import cv2
import numpy as np
import torch.nn as nn
import pdb
import os
from tqdm import tqdm

# Load the pretrained ResNet-50 model
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(2048, 2)

# CHANGE -----------------------------------------------------------------
# Load the saved model parameters from the .pth file
model.load_state_dict(torch.load('/root/jieunoh/cervical_deformity/result/resnet50/resnet50_onlytwo_0515/model_fold1.pth'))
model.eval()

# image_path = '/root/jieunoh/cervical_deformity/data/Cancer/C_P0904_01_CS03_H_M_1_adenocarcinoma.jpg'
save_dir = '/root/jieunoh/cervical_deformity/result/resnet50/resnet50_onlytwo_0515'
data_dir = "/root/jieunoh/cervical_deformity/data"
class_list = ["Cancer", "normal"]
# class_list = ["Cancer", "HSIL", "LSIL", "normal"]

count = 3  # number of CMA images to produce for each class
# CHANGE (end) -----------------------------------------------------------------


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.feature_map = None
        self.gradient = None
        self.model._modules.get('layer4').register_forward_hook(self.save_feature_map)
        self.model._modules.get('layer4').register_backward_hook(self.save_gradient)

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

        return cam.squeeze().detach().numpy()

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 이걸안해야 색이 잘 나옴
    image = cv2.resize(image, (512, 512))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image)
    return image

device = torch.device("cuda:3")
pdb.set_trace()
gradcam = GradCAM(model)

for class_now in class_list:
    image_list = os.listdir(os.path.join(data_dir,class_now))
    print(class_now)
    for i, imgg  in enumerate(tqdm(image_list)):
        if i==count: 
            break
        image_path = os.path.join(data_dir,class_now,imgg)
        input_image = preprocess_image(image_path)

        # Generate the GradCAM heatmap
        cam = gradcam(input_image)

        # Rescale the heatmap to the original image size
        # pdb.set_trace()
        heatmap = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)

        # Convert the input image and heatmap to RGBA
        input_image = np.uint8(255 * input_image.squeeze().permute(1, 2, 0))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # # Apply the heatmap as an overlay on the input image
        # alpha = 0.6  # Adjust the transparency level as desired
        # overlaid_image = cv2.addWeighted(input_image, 1-alpha, heatmap, alpha, 0)

        # Convert the overlaid image back to BGR for saving
        # overlaid_image = cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR)
        # Set transparency (alpha channel) based on heatmap intensity
        alpha = 0.2  # Adjust the transparency level as desired
        heatmap = np.uint8(heatmap * alpha)

        # Overlay the heatmap on the input image
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        overlaid_image = cv2.addWeighted(input_image, 1, heatmap, 1, 0)
        # overlaid_image = cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(save_dir,f'class{class_now}_{i}_overlay.jpg')
        cv2.imwrite(save_path, overlaid_image)



