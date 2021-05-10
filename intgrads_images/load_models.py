"""
Interface for loading the supported image classification models.
"""

import torch
import torchvision as tv

from .models.bit_model import KNOWN_MODELS
from .models.lanet_model import NetworkCIFAR, gen_code_from_list, translator

# dictionary with target value for each classification
classification_dict = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
    }

def load_bit_model(model_path, device):
    """
    Load the pretrained BiT model states and prepare the model for image classification.

    Paramters
    ---------
    model_path: str
        Path to the pretrained model states binary file.
    device: torch.device
        Device to load the model on.

    Returns
    -------
    model: ResNetv2
        Model with the loaded pretrained states.
    transforms: tv.transforms
        Set of torchvision transforms to apply to the input image.
    """
    state_dict = torch.load(model_path) # model should already be on CPU
    model = KNOWN_MODELS["BiT-S-R101x3"](head_size=10, zero_head=False)

    for key in list(state_dict.keys()): # remove 'module' from every state_dict key
        new_name = key[7:]
        state_dict[new_name] = state_dict[key]
        del state_dict[key]

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    transforms = tv.transforms.Compose([
        tv.transforms.Resize((32, 32)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return(model, transforms)


def load_lanet_model(model_path, device):
    """
    Load the pretrained LaNet model states and prepare the model for image classification.

    Parameters
    ----------
    model_path: str
        Path to the pretrained model states file.
    device: torch.device:
        Device to load the model on.

    Returns
    -------
    model: NetworkCIFAR
        Model with the loaded pretrained states and correct architecture.
    transforms: tv.transforms
        Set of torchvision transforms to apply to the input image.
    """
    # best architecture from LaNet model finetuning
    net = [2, 2, 0, 2, 1, 2, 0, 2, 2, 3, 2, 1, 2, 0, 0, 1, 1, 1, 2, 1, 1, 0, 3, 4, 3, 0, 3, 1]

    # set up the model architecture
    code = gen_code_from_list(net, node_num=int((len(net) / 4)))
    genotype = translator([code, code], max_node=int((len(net) / 4)))

    # load in the model and weights
    model = NetworkCIFAR(128, 10, 24, True, genotype)
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.drop_path_prob = 0.1

    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    transforms = tv.transforms.Compose([
        tv.transforms.Resize((32, 32)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return(model, transforms)
    

def load_models(device, bit_path, lanet_path):
    """
    Load the models and image transforms and return them in a dictionary.

    Parameters
    ----------
    device: torch.device
        Device to load the model on.
    bit_path: str or None
        Path to the pretrained BiT model states binary file.
    lanet_path: str or None
        Path to the pretrained LaNet model states binary file.

    Returns
    -------
    model_dict: dict
        Dictionary storing the model, model name, and model's image transformation process.
        Current keys are 'model_name', 'model', 'transforms'.
    """
    if bit_path is not None:
        bit_model, bit_transforms = load_bit_model(str(bit_path), device)
        return {"model_name": "bit", "model": bit_model, "transforms": bit_transforms}
    elif lanet_path is not None:
        lanet_model, lanet_transforms = load_lanet_model(str(lanet_path), device)
        return {"model_name": "lanet", "model": lanet_model, "transforms": lanet_transforms}
    # add additional models here
    else:
        return None
