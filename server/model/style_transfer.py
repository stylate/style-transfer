import matplotlib.pyplot as plt
import utils
import torch
import torch.optim as optim
from torchvision import models

def process(content_path, style_path, iterations=500):
    # init VGG model
    vgg = models.vgg19(pretrained=True)
    for param in vgg.parameters():
        param.requires_grad_(False)

    for i, layer in enumerate(vgg.features):
        if isinstance(layer, torch.nn.MaxPool2d):
            vgg.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg.to(device).eval()

    # init content and style image + features
    content = utils.load_image(content_path).to(device)
    style = utils.load_image(style_path).to(device)

    content_features = utils.get_features(content, vgg)
    style_features = utils.get_features(style, vgg)

    content_weight = 1e4
    style_weight = 1e2
    target = torch.randn_like(content).requires_grad_(True).to(device)

    # gradient descent
    optimizer = optim.Adam([target], lr=0.01)
    style_weights = {
        'conv1_1': 0.75,
        'conv2_1': 0.5,
        'conv3_1': 0.2,
        'conv4_1': 0.2,
        'conv5_1': 0.2
    }
    style_grams = {
        layer: utils.gram_matrix(style_features[layer]) for layer in style_features
    }

    for i in range(1, iterations + 1):
        optimizer.zero_grad()
        target_features = utils.get_features(target, vgg)

        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = utils.gram_matrix(target_feature)
            _, dim, h, w = target_feature.shape

            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss / (dim * h * w)

            total_loss = (content_weight * content_loss) + (style_weight * style_loss)
            total_loss.backward(retain_length=True)
            optimizer.step()

    final_img = utils.deprocess(target)
    return final_img