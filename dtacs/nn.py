import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
import segmentation_models_pytorch as smp

def load_model(model_name:str, input_bands, output_bands,corrector:bool=True, downscale_factor_corrector:int=1,
               backbone_unet:str="mobilenet_v2") -> nn.Module:
    input_channel_size = len(input_bands)
    output_channel_size = len(output_bands)

    if model_name == 'SimpleCNN':
        model = SimpleCNN(input_channel_size=input_channel_size, output_channel_size=output_channel_size)
    elif model_name == 'SimpleCNNAE':
        assert input_channel_size == output_channel_size
        model = SimpleConvolutionalAutoencoder(channel_size=input_channel_size)
    elif model_name == 'SimpleAE':
        assert input_channel_size == output_channel_size
        model = SimpleAutoencoder(channel_size=input_channel_size)
    elif model_name == 'CNNAE':
        assert input_channel_size == output_channel_size
        model = ConvolutionalAutoencoder(channel_size=input_channel_size)
    elif model_name == 'CNN':
        model = CNN(channel_size=input_channel_size, output_channel_size=output_channel_size)
    elif model_name == 'ComplexCNN_small':
        assert input_channel_size == output_channel_size
        model = ComplexCNN_small(channel_size=input_channel_size)
    elif model_name == 'ComplexCNN_medium':
        assert input_channel_size == output_channel_size
        model = ComplexCNN_medium(channel_size=input_channel_size)
    elif model_name == 'ComplexCNN_big':
        assert input_channel_size == output_channel_size
        model = ComplexCNN_big(channel_size=input_channel_size)
    elif model_name == 'CNNAE_2':
        assert input_channel_size == output_channel_size
        model = ConvolutionalAutoencoder_1(channel_size=input_channel_size)
    elif model_name == 'CNNAE_2_w_dropout':
        assert input_channel_size == output_channel_size
        model = ConvolutionalAutoencoder_1_w_dropout(channel_size=input_channel_size, dropout=0.2)
    elif model_name == 'CNNAE_3':
        assert input_channel_size == output_channel_size
        model = ConvolutionalAutoencoder_2(channel_size=input_channel_size)
    elif model_name == 'SimpleRSNet':
        model = SimpleRSNet(input_channel_size, output_channel_size)
    elif model_name == 'Unet':
        model = smp.Unet(encoder_name=backbone_unet, encoder_weights=None, in_channels=input_channel_size,
                         classes=output_channel_size)
    elif model_name == 'DeepLabV3':
        # 'timm-mobilenetv3_large_100'
        model = smp.DeepLabV3(encoder_name=backbone_unet, encoder_weights=None,
                              in_channels=input_channel_size, classes=output_channel_size)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    if corrector:
        channels_input_in_output = [input_bands.index(b) for b in output_bands]

        model = Corrector(model, channels_input_in_output=channels_input_in_output,
                          downscale_factor=downscale_factor_corrector)

    return model


def load_cloud_detection_model():
    return smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=13,
        classes=4,
    )

class Corrector(nn.Module):
    def __init__(self, model:nn.Module, channels_input_in_output:List[int], downscale_factor:int=1):
        super(Corrector,self).__init__()
        self.model = model
        self.channels_input_in_output = channels_input_in_output
        self.downscale_factor = downscale_factor

    def forward(self, x):
        if self.downscale_factor > 1:
            x_model = F.interpolate(x, scale_factor=1 / self.downscale_factor, mode="bilinear", antialias=True)
        else:
            x_model = x

        correction = self.model(x_model)
        if self.downscale_factor > 1:
            correction = F.interpolate(correction, size=x.shape[-2:], mode="bilinear")

        input_to_output = x[:, self.channels_input_in_output, ...]

        return input_to_output - correction


class SimpleCNN(nn.Module):
    def __init__(self,input_channel_size=13, output_channel_size=12):
        super(SimpleCNN,self).__init__()
        self.cnn_layers=torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=input_channel_size,
                                  out_channels=output_channel_size,
                                  kernel_size=1,
                                  stride=1))
    def forward(self, x):
        x = self.cnn_layers(x)
        return x

class CNN(nn.Module):
    def __init__(self,channel_size=3, output_channel_size=None):
        if output_channel_size is None:
            output_channel_size = channel_size

        super(CNN,self).__init__()
        self.cnn_layers=torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=channel_size,
                                        out_channels=int(channel_size*4),
                                        kernel_size=1,
                                        stride=1),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(in_channels=int(channel_size*4),
                                        out_channels=output_channel_size,
                                        kernel_size=1,
                                        stride=1)
        )
    def forward(self,x):
        x = self.cnn_layers(x)
        return x

class ComplexCNN_small(nn.Module):
    def __init__(self,channel_size=3):
        super(ComplexCNN_small,self).__init__()
        self.cnn_layers=torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=channel_size,
                                        out_channels=int(channel_size*2),
                                        kernel_size=3,
                                        stride=1),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(in_channels=int(channel_size*2),
                                        out_channels=int(channel_size*4),
                                        kernel_size=6,
                                        stride=1),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(in_channels=int(channel_size*4),
                                        out_channels=int(channel_size*2),
                                        kernel_size=6,
                                        stride=1),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(in_channels=int(channel_size*2),
                                        out_channels=channel_size,
                                        kernel_size=3,
                                        padding=7,
                                        stride=1)
        )
    def forward(self,x):
        x = self.cnn_layers(x)
        return x

class ComplexCNN_medium(nn.Module):
    def __init__(self,channel_size=3):
        super(ComplexCNN_medium,self).__init__()
        self.cnn_layers=torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=channel_size,
                                        out_channels=int(channel_size*4),
                                        kernel_size=3,
                                        stride=1),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(in_channels=int(channel_size*4),
                                        out_channels=int(channel_size*8),
                                        kernel_size=7,
                                        stride=1),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(in_channels=int(channel_size*8),
                                        out_channels=int(channel_size*4),
                                        kernel_size=7,
                                        stride=1),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(in_channels=int(channel_size*4),
                                        out_channels=channel_size,
                                        kernel_size=3,
                                        padding=8,
                                        stride=1)
        )
    def forward(self,x):
        x = self.cnn_layers(x)
        return x

class ComplexCNN_big(nn.Module):
    def __init__(self,channel_size=3):
        super(ComplexCNN_big,self).__init__()
        self.cnn_layers=torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=channel_size,
                                        out_channels=int(channel_size*4),
                                        kernel_size=3,
                                        stride=1),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(in_channels=int(channel_size*4),
                                        out_channels=int(channel_size*8),
                                        kernel_size=3,
                                        stride=1),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(in_channels=int(channel_size*8),
                                        out_channels=int(channel_size*16),
                                        kernel_size=7,
                                        stride=1),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(in_channels=int(channel_size*16),
                                        out_channels=int(channel_size*8),
                                        kernel_size=7,
                                        stride=1),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(in_channels=int(channel_size*8),
                                        out_channels=int(channel_size*4),
                                        kernel_size=3,
                                        stride=1),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(in_channels=int(channel_size*4),
                                        out_channels=channel_size,
                                        kernel_size=3,
                                        padding=10,
                                        stride=1)
        )
    def forward(self,x):
        x = self.cnn_layers(x)
        return x

###### Linear Autoencoder ######
class SimpleAutoencoder(nn.Module):
    def __init__(self, image_shape=(509, 509), channel_size=3):
        self.image_shape=image_shape
        self.channel_size=channel_size
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(image_shape[0]*image_shape[1]*channel_size, 509),
            torch.nn.ReLU(),
            torch.nn.Linear(509, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 509),
            torch.nn.ReLU(),
            torch.nn.Linear(509, image_shape[0]*image_shape[1]*channel_size),
        )

    def forward(self, x):
        x=x.reshape(-1,self.image_shape[0]*self.image_shape[1]*self.channel_size)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.reshape(x.shape[0],self.channel_size,self.image_shape[0],self.image_shape[1])


###### Convolutional Autoencoders ######
class SimpleConvolutionalAutoencoder(nn.Module):
    def __init__(self, channel_size=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channel_size,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64,
                      out_channels=32,
                      kernel_size=3,
                      stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32,
                      out_channels=16,
                      kernel_size=2,
                      stride=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16,
                               out_channels=32,
                               kernel_size=2,
                               stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=64,
                               kernel_size=2,
                               stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=channel_size,
                               kernel_size=4,
                               stride=1)
        )
    def forward(self, x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, channel_size=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channel_size,
                      out_channels=128,
                      kernel_size=3,
                      stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64,
                      out_channels=32,
                      kernel_size=2,
                      stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32,
                      out_channels=16,
                      kernel_size=1,
                      stride=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16,
                               out_channels=32,
                               kernel_size=2,
                               stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=64,
                               kernel_size=2,
                               stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=channel_size,
                               kernel_size=4,
                               stride=1)
        )
    def forward(self, x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x

class ConvolutionalAutoencoder_1(nn.Module):
    def __init__(self, channel_size=3):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(channel_size, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128,3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,256,7),
            nn.ReLU()
        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(256,128,7),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,12,3,stride=2,padding=1,output_padding=1),
        )
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ConvolutionalAutoencoder_1_w_dropout(nn.Module):
    def __init__(self, channel_size=3, dropout=0.2):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(channel_size, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(64,128,3,stride=2,padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(128,256,7),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(256,128,7),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ConvTranspose2d(64,12,3,stride=2,padding=1,output_padding=1),
        )
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ConvolutionalAutoencoder_2(nn.Module):
    def __init__(self, channel_size=3):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(channel_size, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128,256,5,stride=1,padding=1),
            nn.ReLU()
            )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(256,128,5),
            nn.ReLU(),
            nn.ConvTranspose2d(128,12,3,stride=1,padding=2),
        )
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class DSConv2d(torch.nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
        super(DSConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = torch.nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

def SimpleRSNet(num_channels=13, out_channels=12):
    return torch.nn.Sequential(
        DSConv2d(num_channels, 64, kernel_size=5, padding=2),
        torch.nn.ReLU(inplace=True),
        DSConv2d(64, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        DSConv2d(64, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        DSConv2d(32, out_channels, kernel_size=3, padding=1)
    )

class SimpleRSNetPredict(torch.nn.Module):
        def __init__(self, num_channels=13, out_channels=12):
            super().__init__()
            self.model = SimpleRSNet(num_channels, out_channels)
        def forward(self, x):
            return self.model(x)
        
