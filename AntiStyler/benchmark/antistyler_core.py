"""AntiStyler reference implementation, extracted verbatim from
AntiStyler_Demo.ipynb (the paper's own demo notebook) for reuse in the
Table 1 benchmark harness, instead of hand-copying it a second time.

ContentLoss, gram_matrix, StyleLoss, AntiStyle, AntiStyler, get_device,
and load_image are unchanged from the notebook. The only modification is
in AntiStyler.apply(): the notebook calls matplotlib's imshow() five times
per call (input/padded/styled/raw-mask/final-mask), which is fine for a
single-image demo but unusable for a 5000-image batch eval (headless
environment, ~5000x cost). Each imshow call is now gated behind a
`visualize: bool = False` parameter -- the tensor computation itself is
untouched. `generate_attack`/`class_names`/`imshow` from the notebook
(the demo's own toy per-image "creation" attack, used only to build the
Adversarial Detection demo cell) are not needed here: Table 1's three
attacks (Google's Adversarial Patch, M-PGD, DPatch) are implemented via
ART in attacks.py instead, per the paper's Section 5.1 ("ART 1.15.1").
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image


class ContentLoss(nn.Module):
    """
    Content loss module for neural style transfer.

    Parameters:
        target (torch.Tensor): The target content feature map.

    Attributes:
        loss (torch.Tensor): Content loss value.

    """

    def __init__(self, target=None):
        super(ContentLoss, self).__init__()
        self.loss = None
        if target is not None:
            self.target = target

    def forward(self, input_feature_map):
        """
        Calculate the loss between the input image feature map and the target content feature map.
        Return the input image feature map to maintain the continuity of the model flow.

        Args:
            input_feature_map (torch.Tensor): The input image feature map.

        Returns:
            torch.Tensor: The input image feature map.
        """
        if self.target is not None:
            self.loss = F.mse_loss(input_feature_map, self.target)
        return input_feature_map


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    """
    Style loss module for neural style transfer.

    Parameters:
        target_feature (torch.Tensor): The target style feature map.

    Attributes:
        loss (torch.Tensor): Style loss value.
        target (torch.Tensor): Gram matrix of the target style feature map.

    """

    def __init__(self, target_feature_map=None, name=None):
        super(StyleLoss, self).__init__()
        self.loss = None
        self.name = name
        if target_feature_map is not None:
            self.target = gram_matrix(target_feature_map)

    def forward(self, input_feature_map):
        """
        Calculate the loss between the Gram matrix of the input image feature map and the target style Gram matrix.
        Return the input image feature map to maintain the continuity of the model flow.

        Args:
            input_feature_map (torch.Tensor): The input image feature map.

        Returns:
            torch.Tensor: The input image feature map.
        """
        if self.target is not None:
            G = gram_matrix(input_feature_map)
            self.loss = F.mse_loss(G, self.target)
        return input_feature_map


class AntiStyle:
    """
    Class for performing anti-style transfer.

    Parameters:
        backbone (torch.nn.Module): The base feature extraction convolutional neural network.
        content_layers (list): List of layer names to be considered for content loss.
        style_layers (list): List of layer names to be considered for style loss.
        content_weight (float): Weight for the content loss.
        style_weight (float): Weight for the style loss.

    Attributes:
        content_layers (list): List of layer names for content loss.
        style_layers (list): List of layer names for style loss.
        content_weight (float): Weight for the content loss.
        style_weight (float): Weight for the style loss.
        model (torch.nn.Module): anti-style transfer model.
        style_losses (list): List of style losses.
        content_losses (list): List of content losses.

    """

    def __init__(self, backbone, content_layers, style_layers, content_weight, style_weight, num_steps):
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.num_steps = num_steps
        self.model = None
        self.style_losses = None
        self.content_losses = None
        self.set_base_model(backbone)

    def set_base_model(self, backbone):
        """
        Sets up the base model by trimming unnecessary layers and adding content and style loss layers.

        Args:
            backbone (torch.nn.Module): The base feature extraction convolutional neural network.

        Raises:
            ValueError: If no ContentLoss or StyleLoss layers are found in the model.

        """
        import torchvision

        model = nn.Sequential()

        i = 0  # increment every time we see a conv layer.
        for layer in backbone.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.SiLU):
                name = 'silu_{}'.format(i)
                # layer = nn.SiLU(inplace=False)
            elif isinstance(layer, torchvision.ops.StochasticDepth):
                name = 'StochasticDepth_{}'.format(i)
                # layer = torchvision.ops.StochasticDepth(p=0,mode='row')
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            # add hidden content loss layer after content layer.
            if name in self.content_layers:
                model.add_module("content_loss_{}".format(i), ContentLoss())

            # add hidden style loss layer after style layer.
            if name in self.style_layers:
                model.add_module("style_loss_{}".format(i), StyleLoss())

        # Find the index of the last content or style loss layer.
        last_loss_layer_index = None
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], (ContentLoss, StyleLoss)):
                last_loss_layer_index = i
                break

        # Check if any content or style loss layers were found.
        if last_loss_layer_index is not None:
            # Trim off layers after the last content or style loss layer.
            trimmed_model = model[:last_loss_layer_index + 1]
            self.model = trimmed_model
        else:
            # Handle the case where no content or style loss layers were found.
            raise ValueError("No ContentLoss or StyleLoss layers found in the model.")

        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.requires_grad_(False)

    def update_targets(self, content_image, style_image):
        """
        Update the target feature maps for content and style images.

        Args:
            content_image (torch.Tensor): Content image.
            style_image (torch.Tensor): Style image.

        """
        import torchvision

        content_losses = []
        style_losses = []

        content_image_tensor = content_image.detach()
        style_img_tensor = style_image.detach()

        i = 0  # increment every time we see a conv layer.
        for layer in self.model.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
            elif isinstance(layer, nn.SiLU):
                name = 'silu_{}'.format(i)
            elif isinstance(layer, torchvision.ops.StochasticDepth):
                name = 'StochasticDepth_{}'.format(i)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            elif isinstance(layer, ContentLoss) or isinstance(layer, StyleLoss):
                continue
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            content_image_tensor = layer(content_image_tensor).detach()
            style_img_tensor = layer(style_img_tensor).detach()

            # update hidden content loss layers:
            if name in self.content_layers:
                content_loss = ContentLoss(content_image_tensor)
                setattr(self.model, f'content_loss_{i}', content_loss)
                content_losses.append(content_loss)

            # update hidden style loss layers:
            if name in self.style_layers:
                style_loss = StyleLoss(style_img_tensor)
                setattr(self.model, f'style_loss_{i}', style_loss)
                style_losses.append(style_loss)

        self.style_losses = style_losses
        self.content_losses = content_losses

    def apply(self, content_image, style_image, input_img):
        """
        Apply the anti-style transfer to the input image.

        Args:
            content_image (torch.Tensor): Content image.
            style_image (torch.Tensor): Style image.
            input_img (torch.Tensor): Input image to be optimized.

        Returns:
            torch.Tensor: Optimized input image.

        """
        self.update_targets(content_image, style_image)

        input_img.requires_grad_(True)

        optimizer = optim.Adam([input_img])

        def anti_style_step():
            optimizer.zero_grad()
            self.model(input_img)
            style_loss = -sum(sl.loss for sl in self.style_losses)
            style_loss.backward()
            return style_loss

        def content_step():
            optimizer.zero_grad()
            self.model(input_img)
            content_loss = sum(cl.loss for cl in self.content_losses)
            style_loss = -sum(sl.loss for sl in self.style_losses)
            total_loss = content_loss * self.content_weight + style_loss * self.style_weight
            total_loss.backward()
            return total_loss

        # Perform optimization steps.
        for _ in range(0, self.num_steps):
            optimizer.step(anti_style_step)
            optimizer.step(content_step)
            # Clip pixel values to valid range.
            input_img.data.clamp_(0, 1)

        return input_img.detach()


class AntiStyler:

    def __init__(self, backbone, content_layers, style_layers, content_weight, style_weight, num_steps):
        self.anti_style = AntiStyle(backbone, content_layers, style_layers, content_weight, style_weight, num_steps)
        self.image_size = 600
        self.padding = 10
        self.tau = 0.99
        self.thres = 0.5

    def resize(self, image, size):
        return transforms.Resize(size, antialias=True)(image)

    def erode(self, mask, kernel_size=11):
        eroded_mask = F.max_pool2d(1 - mask.float(), kernel_size, stride=1, padding=kernel_size // 2)
        return 1 - eroded_mask

    def dilate(self, mask, kernel_size=11):
        dilated_mask = F.max_pool2d(mask.float(), kernel_size, stride=1, padding=kernel_size // 2)
        return dilated_mask

    def apply_mean_kernel(self, mask, kernel_size=11):
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device) / (kernel_size * kernel_size)
        mean_filtered_mask = F.conv2d(mask.unsqueeze(0), kernel, padding=kernel_size // 2)
        return mean_filtered_mask.squeeze(0)

    def apply(self, image, device, visualize: bool = False):
        if visualize:
            from .notebook_imshow import imshow
            imshow(image, title="Input Image")
        original_size = (image.shape[2], image.shape[3])
        content_image = torch.zeros((1, 3, self.image_size, self.image_size), device=device)
        content_image[:, :, :self.padding, :] = torch.rand((1, 3, self.padding, self.image_size))
        content_image[:, :, -self.padding:, :] = torch.rand((1, 3, self.padding, self.image_size))
        content_image[:, :, :, :self.padding] = torch.rand((1, 3, self.image_size, self.padding))
        content_image[:, :, :, -self.padding:] = torch.rand((1, 3, self.image_size, self.padding))
        content_image[:, :, self.padding:-self.padding, self.padding:-self.padding] = self.resize(image, (self.image_size - 2 * self.padding, self.image_size - 2 * self.padding))

        content_image.to(device, torch.float)
        if visualize:
            imshow(content_image, title="Padded Input Image")
        style_image = torch.randn_like(content_image, device=device, dtype=torch.float)
        anti_styled_image = self.anti_style.apply(content_image, style_image, content_image.clone())
        if visualize:
            imshow(anti_styled_image, title="AntiStyled Image")

        difference_image = torch.abs(anti_styled_image - content_image)
        difference_image = difference_image.mean(dim=1)
        flattened_values = difference_image.flatten()
        k = int(self.tau * flattened_values.shape[0])
        threshold_value = torch.kthvalue(flattened_values, k, dim=0).values.item()
        difference_image = (difference_image >= threshold_value).float()
        difference_image = difference_image[:, self.padding: -self.padding, self.padding: -self.padding]
        if visualize:
            temp = torch.stack([difference_image] * 3, dim=1)
            imshow(temp, title="Raw Mask")

        # Apply erosion and dilation
        difference_image = self.dilate(difference_image)
        difference_image = self.erode(difference_image)
        difference_image = self.apply_mean_kernel(difference_image, kernel_size=51)
        difference_image = (difference_image >= self.thres).float()
        difference_image = self.dilate(difference_image)

        if visualize:
            temp = torch.stack([difference_image] * 3, dim=1)
            imshow(temp, title="Final Mask")

        if torch.sum(difference_image).item():
            difference_image = torch.stack([difference_image] * 3, dim=1)
            anti_styled_image = anti_styled_image[:, :, self.padding: -self.padding, self.padding: -self.padding]
            final_image = anti_styled_image - (difference_image * anti_styled_image)
            final_image = self.resize(final_image, original_size)
        else:
            final_image = image

        return final_image


def get_device():
    """
    Get the available computation device.

    Checks whether CUDA (GPU) is available via PyTorch and returns
    the appropriate device object.

    Returns
    -------
    torch.device
        'cuda' if a GPU is available, otherwise 'cpu'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def load_image(image_path, device, file_type='jpg'):
    """
    Load an image from the given path, convert it to RGB if it's grayscale, and return it as a PyTorch tensor.

    Args:
        image_path (str): The path to the image file.
        device (torch.device): The device to which the image tensor should be moved.
        file_type (str): The file type.

    Returns:
        torch.Tensor: The image as a PyTorch tensor.
    """
    import numpy as np

    if file_type == 'npy':
        image = np.load(image_path) / 255.0
    else:
        # Deviation from the notebook (which did `Image.open(image_path)`
        # with no explicit mode conversion): COCO val2017 contains a
        # handful of grayscale/CMYK/palette images, and the notebook's
        # own grayscale fallback below assumes a specific tensor shape
        # that doesn't actually produce (1, 3, H, W) for a true 1-channel
        # tensor. Converting to RGB up front sidesteps that entirely and
        # is required for a dataset-scale batch load (the notebook only
        # ever loaded one hand-picked RGB demo image).
        image = Image.open(image_path).convert("RGB")
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    if image_tensor.shape[1] == 1:
        image_tensor = torch.stack([image_tensor.squeeze(0)] * 3, dim=1)
    return image_tensor.to(device, torch.float)
