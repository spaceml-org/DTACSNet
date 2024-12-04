import math
from typing import Tuple, List, Optional, Callable
import torch
import segmentation_models_pytorch as smp
import numpy as np
from dtacs import nn
from dtacs.download_weights import download_weights
import os

BANDS_S2_L1C =  ["B01", "B02","B03", "B04", "B05", "B06",
                 "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
BANDS_S2_L2A = ["B01", "B02","B03", "B04", "B05", "B06",
                "B07", "B08", "B8A", "B09", "B11", "B12"]

MODELS ={
    "CNN_corrector": "https://github.com/spaceml-org/DTACSNet/releases/download/v1.0/CNN_corrector.pt",
    "CNN_corrector_phisat2": "https://github.com/spaceml-org/DTACSNet/releases/download/v1.0/CNN_corrector_phisat2.pt",
    "CNN_corrector_planetscope": "https://github.com/spaceml-org/DTACSNet/releases/download/v1.0/CNN_corrector_planetscope.pt",
    "CNN_corrector_probav": "https://github.com/spaceml-org/DTACSNet/releases/download/v1.0/CNN_corrector_probav.pt",
    "Unet_corrector": "https://github.com/spaceml-org/DTACSNet/releases/download/v1.0/Unet_corrector.pt",
    "Unet_corrector_phisat2": "https://github.com/spaceml-org/DTACSNet/releases/download/v1.0/Unet_corrector_phisat2.pt",
    "Unet_corrector_planetscope": "https://github.com/spaceml-org/DTACSNet/releases/download/v1.0/Unet_corrector_planetscope.pt",
    "Unet_corrector_probav": "https://github.com/spaceml-org/DTACSNet/releases/download/v1.0/Unet_corrector_probav.pt",
    "Linear": "https://github.com/spaceml-org/DTACSNet/releases/download/v1.0/linear.pt", # run_SimpleCNN
}

def find_padding(v:int, divisor=32) -> Tuple[Tuple[int, int],slice]:
    v_divisible = max(divisor, int(divisor * math.ceil(v / divisor)))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    slice_rows = slice(pad_1, None if pad_2 <= 0 else -pad_2)
    return (pad_1, pad_2), slice_rows

# ~/.dtacs/
DIR_MODELS_LOCAL = os.path.expanduser("~/.dtacs/")

class ACModel:
    def __init__(self, model_name:str, input_bands:Optional[List[str]]=None, output_bands:Optional[List[str]]=None, 
                 device:torch.device=torch.device("cpu"), dir_models:Optional[str]=DIR_MODELS_LOCAL):
        super().__init__()

        if model_name not in MODELS:
            raise ValueError(f"Model {model_name} not found in MODELS. Available models: {list(MODELS.keys())}")
        
        if model_name == "Linear":
            model_to_load = "SimpleCNN"
            corrector = False
        else:
            model_to_load = model_name.split("_")[0]
            corrector = True
        if input_bands is None and output_bands is None:
            if "phisat2" in model_name:
                # B2, B3, B4, B5, B6, B7, and B8
                input_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08"]
                output_bands = list(input_bands)
            elif "planetscope" in model_name:
                # B2, B3, B4, and B8
                input_bands = ["B02", "B03", "B04", "B08"]
                output_bands = list(input_bands)
            elif "probav" in model_name:
                input_bands = ["B02", "B04", "B08", "B11"]
                output_bands = list(input_bands)
            else:
                input_bands = list(BANDS_S2_L1C)
                output_bands = list(BANDS_S2_L2A)
        elif input_bands is None or output_bands is None:
            raise ValueError("Both input_bands and output_bands must be provided or none of them")

        self.input_bands = input_bands
        self.output_bands = output_bands
        self.model_name = model_name
        self.model_to_load = model_to_load
        self.dir_models = dir_models
        self.corrector = corrector

        self.model = nn.load_model(model_to_load,
                                   input_bands, output_bands,
                                   corrector=corrector)
        self.device = device
        self.model.eval()
        self.model.to(self.device)

    def load_weights(self, path:Optional[str]=None):
        if path is None:
            path = os.path.join(self.dir_models, self.model_name+".pt")
            if not os.path.exists(path):
                os.makedirs(self.dir_models, exist_ok=True)
                download_weights(path, MODELS[self.model_name])
        
        with open(path,"rb") as fh:
            weights =  torch.load(fh, map_location=self.device)
        
        self.model.load_state_dict(weights["state_dict"])

    def predict(self, tensor: np.array) -> np.array:
        """
            tensor: np.array (`in_channels`, H, W) with reflectances multiplied by 10000

        Returns:
            np.uint16 np.array (H, W) with surface reflectances multiplied by 10000
        """

        pad_r,slice_rows = find_padding(tensor.shape[-2])
        pad_c, slice_cols= find_padding(tensor.shape[-1])

        tensor_padded = np.pad(
            tensor, ((0, 0), (pad_r[0], pad_r[1]), (pad_c[0], pad_c[1])), "reflect"
        )

        tensor_padded /= 10_000  # assume values are as downloaded
        tensor_padded = torch.tensor(tensor_padded, device=self.device)[None]  # Add batch dim

        with torch.no_grad():
            pred_padded = self.model(tensor_padded)[0]
            pred_cont = torch.clamp(pred_padded[(slice(None), slice_rows, slice_cols)], 0, 2)
            pred_cont *= 10_000

        return np.array(pred_cont.cpu()).astype(np.uint16)


class CDModel(torch.nn.Module):
    def __init__(self, device:torch.device=torch.device("cpu") ,model:Optional[Callable]=None, in_channels:int=13):
        super().__init__()
        if model is None:
            self.model = smp.Unet(
                encoder_name="mobilenet_v2",
                encoder_weights=None,
                in_channels=in_channels,
                classes=4
            )
        else:
            self.model = model
        self.in_channels = in_channels
        self.device = device
        self.model.eval()
        self.model.to(self.device)

    def load_weights(self, path:str):
        with open(path, "rb") as fh:
            weights =  torch.load(fh, map_location=self.device)
        self.load_state_dict(weights["state_dict"])

    def predict(self, tensor: np.array) -> np.array:
        """
            tensor: np.array (`in_channels`, H, W) with reflectances multiplied by 10000

        Returns:
            np.uint8 np.array (H, W) with interpretation {0: clear, 1: Thick cloud, 2: thin cloud, 3: cloud shadow}
        """

        pad_r,slice_rows = find_padding(tensor.shape[-2])
        pad_c, slice_cols= find_padding(tensor.shape[-1])

        tensor_padded = np.pad(
            tensor, ((0, 0), (pad_r[0], pad_r[1]), (pad_c[0], pad_c[1])), "reflect"
        )

        tensor_padded /= 10_000  # assume values are as downloaded
        tensor_padded = torch.tensor(tensor_padded, device=self.device)[None]  # Add batch dim

        with torch.no_grad():
            pred_padded = self.model(tensor_padded)[0]
            pred_cont = pred_padded[(slice(None), slice_rows, slice_cols)]
            pred_discrete = torch.argmax(pred_cont, dim=0).type(torch.uint8)

        return np.array(pred_discrete.cpu())