"""
Run the model with integrated and intermediate gradients.
"""

from captum.attr import IntegratedGradients
import logging
from PIL import Image
import torch

from intermediate_gradients.intermediate_gradients import IntermediateGradients


def prepare_input(img_array, transforms):
    """
    Crop and prepare the input for modeling.

    Parameters
    ----------
    img_array: np.array(3,x,x)
        RGB array of the photo to obtain classification from.
    transforms: torchvision.transforms
        Set of transformation steps to apply to the input image.
    Returns
    -------
    input_tensor: torch.tensor
        Transformed image ready to be passed into the model.
    """
    im = Image.fromarray(img_array)
    input_tensor = transforms(im)
    return input_tensor.reshape(1,3,32,32)


def bit_sequence_forward_func(inputs, model):
    """
    Passes forward the inputs.

    Parameters
    ----------
    inputs: torch.tensor(1, 3, 32, 32), dtype=torch.float32
        Preprocessed and cropped image tensor.
    model: torch.nn.Module
        Model to run the inputs on.
    Returns
    -------
    outputs: torch.tensor(1, 10), dtype=torch.float32
        Output classifications for the model.
    """
    outputs = model(inputs)
    return outputs


def lanet_sequence_forward_func(inputs, model):
    """
    Passes forward the inputs.

    Parameters
    ----------
    inputs: torch.tensor(1, 3, 32, 32), dtype=torch.float32
        Preprocessed and cropped image tensor.
    model: torch.nn.Module
        Model to run the inputs on.
    Returns
    -------
    outputs: torch.tensor(1, 10), dtype=torch.float32
        Output classifications for the model.
    """
    outputs = model(inputs) #assuming already reshaped to have 2 exam
    return outputs[0]


def run_models(model_name, model, transforms, img, device, target_class):
    """
    Run Integrated and Intermediate gradients on the model with the designated input.

    Parameters
    ----------
    model_name: str
        Name of the model that is being run.
        Currently supported are "Bert" or "XLNet"
    model: torch.nn.Module
        Module to run 
    img: np.array(3,x,x)
        RGB array of the photo to obtain classification from.
    transforms: torchvision.transforms
        Set of transformation steps to apply to the input image.
    device: torch.device
        Device that models are stored on.
    target_class: int
        Target class value to compute gradients in respect to.  Value between 0 and 9.
    Returns
    -------
    gradients_dict: dict
        Dictionary containing the gradient tensors with the following keys:
        "integrated_gradients", "intermediate_gradients", "step_sizes", "output",
        "target_class", and "intermediates".
    """
    input_img = prepare_input(img, transforms).to(device)
    baseline = input_img * 0 # using a 0 baseline... could use random noise instead

    # set up gradients and the baseline ids
    # split up by model because not sure if layer or full gradients
    if model_name == "bit":
        idg = IntermediateGradients(bit_sequence_forward_func)
        ig = IntegratedGradients(bit_sequence_forward_func)
        output = bit_sequence_forward_func(input_img, model)
    elif model_name == "lanet":
        idg = IntermediateGradients(lanet_sequence_forward_func)
        ig = IntegratedGradients(lanet_sequence_forward_func)
        output = lanet_sequence_forward_func(input_img, model)
    grads, step_sizes, intermediates = idg.attribute(inputs=input_img,
                                                     baselines=baseline,
                                                     additional_forward_args=(
                                                         model,
                                                     ),
                                                     target=target_class,
                                                     method="gausslegendre",
                                                     n_steps=50) # maybe pass n_steps as CLI argument

    integrated_grads = ig.attribute(inputs=input_img,
                                     baselines=baseline,
                                     additional_forward_args=(
                                         model,
                                     ),
                                     target=target_class,
                                     n_steps=50)

    grads_dict = {"integrated_directional_grads": grads.to("cpu"),
                  "step_sizes": step_sizes.to("cpu"),
                  "intermediates": intermediates.to("cpu"),
                  "integrated_grads": integrated_grads.to("cpu"),
                  "output": output.to("cpu"),
                  "target_class": target_class}

    return grads_dict
