"""AntiStyler's own hyperparameters, exactly as set in AntiStyler_Demo.ipynb's
"defense = AntiStyler(...)" cell. Not derivable from antistyler_core.py alone
since that module only defines the classes -- this is the actual
instantiation config the notebook uses, extracted the same way as
antistyler_core.py (read off the notebook, not re-invented).
"""
from torchvision.models import VGG19_Weights, vgg19

CONTENT_LAYERS = ["conv_4"]
STYLE_LAYERS = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1000
NUM_STEPS = 1


def build_backbone(device):
    return vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval().to(device)
