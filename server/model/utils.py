from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision import transforms

VGG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
VGG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_image(img_path, max_size=400, shape=None):
    img_path = "../images/" + img_path # parameter should be "content/test.jpg", etc.
    image = Image.open(img_path).convert('RGB')
    size = max(max_size, max(image.size))
    if shape: size = shape

    transform = transforms.Compose([
        transforms.Resize((size, int(1.5*size))),
        transforms.ToTensor(),
        transforms.Normalize(VGG_MEAN.tolist(), VGG_STD.tolist())]
    )

    return transform(image)[:3, :, :].unsqueeze(0) # return as tensor: batch dim, rgb channels, height, width

def deprocess(tensor):
    # convert tensor to numpy array and RGB dimensions
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * VGG_STD + VGG_MEAN
    image = image.clip(0, 1)
    return image

def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2', # content layer
        '28': 'conv5_1' # rest are style layers
    }
    features = {}
    x = image

    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

def gram_matrix(tensor):
    b, c, h, w = tensor.size() # recall: batch, channels, height, width
    tensor = tensor.view(b * c, h * w)
    return torch.mm(tensor, tensor.t())

def show_result(final_img):
    fig = plt.figure()
    plt.imshow(final_img)
    plt.axis('off')
    plt.savefig('../images/results/modern_starry.png')