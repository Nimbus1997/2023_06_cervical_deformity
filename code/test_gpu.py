import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Function
import cv2
import numpy as np
import torch.nn as nn
import pdb


class GradCAM:
    def __init__(self, model):
        self.model = model.to(device)
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

        # return cam.squeeze().detach().numpy()
        return cam.squeeze().detach().cpu().numpy()

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 이걸안해야 색이 잘 나옴
    image = cv2.resize(image, (512, 512))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).to(device)
    return image

device = torch.device("cuda:3")

# Load the pretrained ResNet-50 model
model = models.resnet50(pretrained=True).to(device)
model.fc = nn.Linear(2048, 4)

# Load the saved model parameters from the .pth file
model.load_state_dict(torch.load('/root/jieunoh/cervical_deformity/result/resnet50/resnet50_allclass_0515/model_fold0.pth'))
model.eval()

# image_path = '/root/jieunoh/cervical_deformity/data/Cancer/C_P0904_01_CS03_H_M_1_adenocarcinoma.jpg'
image_path = '/root/jieunoh/cervical_deformity/data/normal/C_P0975_02_CS03_N_L_1.jpg'


input_image = preprocess_image(image_path)


gradcam = GradCAM(model)

# Generate the GradCAM heatmap
cam = gradcam(input_image)

# Rescale the heatmap to the original image size
# pdb.set_trace()
# Move the heatmap and input image to CPU for further processing
heatmap = cam.cpu()
input_image = input_image.cpu().squeeze().permute(1, 2, 0).numpy()

heatmap = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
heatmap = heatmap - np.min(heatmap)
heatmap = heatmap / np.max(heatmap)

# Convert the input image and heatmap to RGB
# input_image = np.uint8(255 * input_image.squeeze().permute(1, 2, 0))
heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# Apply the heatmap as an overlay on the input image
alpha = 0.6  # Adjust the transparency level as desired
overlaid_image = cv2.addWeighted(input_image, 1-alpha, heatmap, alpha, 0)

# Convert the overlaid image back to BGR for saving
# overlaid_image = cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR)
# Set transparency (alpha channel) based on heatmap intensity
alpha = 0.6  # Adjust the transparency level as desired
heatmap = np.uint8(heatmap * alpha)

# Overlay the heatmap on the input image
overlaid_image = cv2.addWeighted(input_image, 1, heatmap, 1, 0)
cv2.imwrite('normal_image.png', overlaid_image)



# # Overlay the heatmap on the input image
# heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
# # overlaid_image = np.float32(heatmap)*0.5 + np.float32(input_image.squeeze())
# overlaid_image = np.float32(heatmap)*0.3 + np.float32(input_image.squeeze().permute(1, 2, 0))*0.7

# overlaid_image = overlaid_image / np.max(overlaid_image)

# Save the GradCAM visualization
# cv2.imwrite('gradcam.jpg', np.uint8(255 * overlaid_image))


