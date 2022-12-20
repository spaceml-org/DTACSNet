import math
from typing import Tuple, List, Optional, Callable
import torch
import segmentation_models_pytorch as smp
import numpy as np
from dtacs import nn


def find_padding(v:int, divisor=32) -> Tuple[Tuple[int, int],slice]:
    v_divisible = max(divisor, int(divisor * math.ceil(v / divisor)))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    slice_rows = slice(pad_1, None if pad_2 <= 0 else -pad_2)
    return (pad_1, pad_2), slice_rows


class ACModel:
    def __init__(self, input_bands:List[str], output_bands:List[str], device:torch.device=torch.device("cpu"),):
        super().__init__()

        self.model = nn.load_model("CNN",
                                   input_bands, output_bands,
                                   corrector=True)
        self.device = device
        self.model.eval()
        self.model.to(self.device)

    def load_weights(self, path:str):
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