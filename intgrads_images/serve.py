"""
Server for obtaining integrated and integrated directional gradients from specific image models.
"""

import click
from collections import OrderedDict
from flask import Flask, request, send_file
import gzip
from io import BytesIO
import numpy as np
import shutil
import torch


from . import cli_main
from .load_models import load_models, classification_dict
from .run_models import run_models

app = Flask(__name__)
MODEL_DICT = {}
DEVICE = None

@app.route("/model/", methods=["POST"])
def run_model():
    """
    Obtain the gradients from running the specified model on the input image tensor.
    The outputs are saved as a gzipped dictionary with the keys:
    integrated_gradients, integrated_directional_gradients, step_sizes, intermediates,
    output, target_class.

    """
    if request.method == 'POST':
        #data = request.get_json(force=True)
        print(request)
        data = request.json

        img_data = data["img_data"]
        img_classification = data["target_class"]
        target_class = classification_dict[img_classification]

        img = np.array(img_data, dtype=np.uint8)

        grads_dict = run_models(MODEL_DICT["model_name"],
                                MODEL_DICT["model"],
                                MODEL_DICT["transforms"],
                                img,
                                DEVICE,
                                target_class)

        temp_bytes, temp_gzip = BytesIO(), BytesIO()

        torch.save(grads_dict, temp_bytes)
        temp_bytes.seek(0)

        with gzip.GzipFile(fileobj=temp_gzip, mode='wb') as f_out:
            shutil.copyfileobj(temp_bytes, f_out)

        temp_gzip.seek(0)

        return send_file(temp_gzip, as_attachment=True, mimetype="/application/gzip", attachment_filename="returned_gradients.gzip")


@cli_main.command(help="Start a server and initialize the models for calculating gradients.")
@click.option(
    "-h",
    "--host",
    required=False,
    default="localhost",
    help="Host to bind to. Default localhost"
)
@click.option(
    "-p",
    "--port",
    default=8888,
    required=False,
    help="Port to bind to. Default 8888"
)
@click.option(
    "--cuda/--cpu",
    required=True,
    default=True,
    help="Whether or not to run models on CUDA."
)
@click.option(
    "--bit-path",
    "-bp",
    required=False,
    help="Path to the BiT finetuned model. Specify only one model path.",
    default=None,
)
@click.option(
    "--lanet-path",
    "-lp",
    required=False,
    help="Path to the pretrained LaNet model. Specifiy only one model path.",
    default=None,
)
def serve(
        host,
        port,
        cuda,
        bit_path=None,
        lanet_path=None
):
    global MODEL_DICT, DEVICE

    if cuda:
        DEVICE = torch.device("cuda:0")
        # will always load inputs on this device even if model parallel option used
    else:
        DEVICE = torch.device("cpu")

    try:
        MODEL_DICT = load_models(DEVICE, bit_path, lanet_path)
    except Exception as e:
        print("An Error occurred: ", e)
        raise e

    app.run(host=host, port=port, debug=True)
