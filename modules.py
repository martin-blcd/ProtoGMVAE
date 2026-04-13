import torch
from torch import nn
from resnet import ResNet18Enc, ResNet18Dec
import torch.nn.functional as F

"""
====================================================================================
Used repositories:
link:
https://github.com/sghalebikesabi/gmm-vae-clustering-pytorch

link:
https://github.com/RuiShu/vae-clustering/blob/master/gmvae.py

link:
https://github.com/insdout/MDS-Thesis-RULPrediction/blob/main/models/tshae_models.py
====================================================================================
"""


class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class Qy_x(nn.Module):
    """Conditional distribution q(y|x) represented by a neural network.

    Args:
        encoder (nn.Module): The encoder module used to process the input data.
        enc_out_dim (int): The output dimension of the encoder module.
        k (int): Number of components in the Gaussian mixture prior.

    Attributes:
        h1 (nn.Module): The encoder module used to process the input data.
        qy_logit (nn.Linear): Linear layer for predicting the logit of q(y|x).
        qy (nn.Softmax): Softmax activation function for q(y|x).

    """
    def __init__(self, encoder, enc_out_dim, k):
        super(Qy_x, self).__init__()
        self.h1 = encoder
        #self.h2 = nn.Sequential(
        #    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
        #    nn.ReLU(),
        #    nn.BatchNorm2d(512),

        #    nn.Flatten(),
        #    nn.Linear(512* int(32 / 32) ** 2, enc_out_dim),
        #    nn.ReLU(),
        #    )
        self.qy_logit = nn.Sequential(
                #nn.Linear(enc_out_dim, enc_out_dim),
                #nn.ReLU(),
                #nn.Linear(enc_out_dim, enc_out_dim),
                #nn.ReLU(),
                nn.Linear(enc_out_dim, k)
                )
        self.qy = nn.Softmax(dim=1)

    def forward(self, x):
        """Perform the forward pass for q(y|x).

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            tuple: A tuple containing the logit and softmax outputs of q(y|x).
        """
        h1 = self.h1(x)
        #h2 = self.h2(h1)
        qy_logit = self.qy_logit(h1)
        qy = self.qy(qy_logit)
        return qy_logit, qy


class Qz_xy(nn.Module):
    """Conditional distribution q(z|x, y) represented by a neural network.

    Args:
(zmean[0] + zmean2[0])/2        k (int): Number of components in the Gaussian mixture prior.
        encoder (nn.Module): The encoder module used to process the input data.
        enc_out_dim (int): The output dimension of the encoder module.
        hidden_size (int): Number of units in the hidden layer.
        latent_dim (int): Dimensionality of the latent space.

    Attributes:
        h1 (nn.Module): The encoder module used to process the input data.
        h2 (nn.Sequential): The hidden layers of the neural network.
        z_mean (nn.Linear): Linear layer for predicting the mean of q(z|x, y).
        zlogvar (nn.Linear): Linear layer for predicting
            the log variance of q(z|x, y).

    """
    def __init__(self, k, encoder, enc_out_dim, hidden_size, latent_dim):
        super(Qz_xy, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.h1 = encoder
        #self.h1_2 = nn.Sequential(   
        #    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
        #    nn.ReLU(),
        #    nn.BatchNorm2d(512),

        #    nn.Flatten(),
        #    nn.Linear(512* int(32 / 32) ** 2, hidden_size),
        #    nn.ReLU(),
        #    )

        self.h2 = nn.Sequential(
            nn.Linear(enc_out_dim + k, hidden_size),
            nn.ReLU(inplace=True),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.z_mean = nn.Linear(hidden_size, latent_dim)
        self.zlogvar = nn.Linear(hidden_size, latent_dim)

    def gaussian_sample(self, z_mean, z_logvar):
        z_std = torch.sqrt(torch.exp(z_logvar))

        eps = torch.randn_like(z_std)
        z = z_mean + eps*z_std

        return z

    def forward(self, x, y):
        """Perform the forward pass for q(z|x, y).

        Args:
            x (torch.Tensor): Input data tensor.
            y (torch.Tensor): One-hot encoded tensor representing
                the class labels.

        Returns:
            tuple: A tuple containing the latent variables, mean,
                and log variance of q(z|x, y).
        """
        h1 = self.h1(x)
        #h1_2 = self.h1_2(h1)
        xy = torch.cat((h1, y), dim=1)
        h2 = self.h2(xy)
        # q(z|x, y)
        z_mean = self.z_mean(h2)
        #zlogvar = self.zlogvar(h2)
        #zlogvar = torch.zeros(z_mean.shape[0],z_mean.shape[1]).to('cuda')
        zlogvar = torch.log(0.1*torch.ones(z_mean.shape[0],z_mean.shape[1])).to('cuda')
        z = self.gaussian_sample(z_mean, zlogvar)
        return z, z_mean, zlogvar


class Px_z(nn.Module):
    """Conditional distribution p(x|z) represented by a neural network.

    Args:
        decoder (nn.Module): The decoder module used to reconstruct the data.
        k (int): Number of components in the Gaussian mixture prior.

    Attributes:
        decoder (nn.Module): The decoder module used to reconstruct the data.
        decoder_hidden (int): Number of units in the hidden layer
            of the decoder.
        latent_dim (int): Dimensionality of the latent space.
        z_mean (nn.Linear): Linear layer for predicting the mean of p(z|y).
        zlogvar (nn.Linear): Linear layer for predicting the log variance
            of p(z|y).

    """
    def __init__(self, decoder, k):
        super(Px_z, self).__init__()
        self.decoder = decoder
        self.decoder_hidden = self.decoder.hidden_size
        self.latent_dim = self.decoder.latent_dim
        self.z_mean = nn.Sequential(nn.Linear(k, self.latent_dim),
                #nn.ReLU(),
                #nn.Linear(self.latent_dim, self.latent_dim),
                #nn.ReLU(),
                #nn.Linear(self.latent_dim, self.latent_dim)
                )
        self.zlogvar = nn.Sequential(nn.Linear(k, self.latent_dim),
            #nn.ReLU(),
            #nn.Linear(self.latent_dim, self.latent_dim),
            #nn.ReLU(),
            #nn.Linear(self.latent_dim, self.latent_dim)
            )

    def forward(self, z, y):
        """Perform the forward pass for p(x|z) and p(z|y).

        Args:
            z (torch.Tensor): Latent variable tensor.
            y (torch.Tensor): One-hot encoded tensor representing
                the class labels.

        Returns:
            tuple: A tuple containing the prior mean, prior log variance,
                and reconstructed data.
        """
        # p(z|y)
        z_mean = self.z_mean(y)
        #zlogvar = self.zlogvar(y)
        #zlogvar = torch.zeros(z_mean.shape[0],z_mean.shape[1]).to('cuda')
        zlogvar = torch.log(0.1*torch.ones(z_mean.shape[0],z_mean.shape[1])).to('cuda')
        # p(x|z)
        x_hat = self.decoder(z)
        return z_mean, zlogvar, x_hat


class EncoderFC(nn.Module):
    """Fully connected encoder module.

    Args:
        input_size (int): Dimensionality of the input data.
        hidden_size (int): Number of units in the hidden layer.
        dropout (float): Dropout probability.

    Attributes:
        input_size (int): Dimensionality of the input data.
        hidden_size (int): Number of units in the hidden layer.
        p (float): Dropout probability.
        enc_block (nn.Sequential): Sequential neural network layers
            for the encoder.

    """
    def __init__(self, input_size, hidden_size, dropout):
        super(EncoderFC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.p = dropout
        self.enc_out_dim = hidden_size

        self.enc_block = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=self.p),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=self.p)
        )

    def forward(self, x):
        """Perform the forward pass for the encoder module.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Encoded tensor after passing through
                the encoder layers.
        """
        h = self.enc_block(x)
        return h


class DecoderFC(nn.Module):
    def __init__(self, input_size, hidden_size,
                 latent_dim, return_probs=True):
        """Fully connected decoder module.

        Args:
            input_size (int): Dimensionality of the input data.
            hidden_size (int): Number of units in the hidden layer.
            latent_dim (int): Dimensionality of the latent space.
            return_probs (bool, optional): Whether to apply a sigmoid
                activation for output probabilities. Defaults to True.

        Attributes:
            input_size (int): Dimensionality of the input data.
            hidden_size (int): Number of units in the hidden layer.
            latent_dim (int): Dimensionality of the latent space.
            return_probs (bool): Whether to apply a sigmoid activation
                for output probabilities.
            dec_block (nn.Sequential): Sequential neural network layers
                for the decoder.
            sigmoid (nn.Sigmoid): Sigmoid activation function.

        """

        super(DecoderFC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.return_probs = return_probs

        self.dec_block = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        """Perform the forward pass for the decoder module.

        Args:
            z (torch.Tensor): Latent variable tensor.

        Returns:
            torch.Tensor: Reconstructed data tensor after passing
                through the decoder layers.
        """
        out = self.dec_block(z)
        if self.return_probs:
            out = self.sigmoid(out)
        return out

class EncoderCONV(nn.Module):
    """Fully connected encoder module.

    Args:
        input_size (int): Dimensionality of the input data.
        hidden_size (int): Number of units in the hidden layer.
        dropout (float): Dropout probability.

    Attributes:
        input_size (int): Dimensionality of the input data.
        hidden_size (int): Number of units in the hidden layer.
        p (float): Dropout probability.
        enc_block (nn.Sequential): Sequential neural network layers
            for the encoder.

    """
    def __init__(self, input_size, hidden_size, dropout, input_channels=1):
        super(EncoderCONV, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.p = dropout
        self.input_channels = input_channels
        self.enc_out_dim = hidden_size

        self.enc_block = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),

            #nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.BatchNorm2d(32),
            #nn.MaxPool2d(2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),

            nn.Dropout2d(p=self.p),
            nn.Flatten(),
            nn.Linear(64 * int(self.input_size / 4) ** 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.p),
            nn.Linear(hidden_size, hidden_size),
        )


    def forward(self, x):
        """Perform the forward pass for the encoder module.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Encoded tensor after passing through
                the encoder layers.
        """
        h = self.enc_block(x)
        return h


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)

class DecoderCONV(nn.Module):
    def __init__(self, input_size, hidden_size,
                 latent_dim, dropout, output_channels=1, return_probs=True):
        """Fully connected decoder module.

        Args:
            input_size (int): Dimensionality of the input data.
            hidden_size (int): Number of units in the hidden layer.
            latent_dim (int): Dimensionality of the latent space.
            return_probs (bool, optional): Whether to apply a sigmoid
                activation for output probabilities. Defaults to True.

        Attributes:
            input_size (int): Dimensionality of the input data.
            hidden_size (int): Number of units in the hidden layer.
            latent_dim (int): Dimensionality of the latent space.
            return_probs (bool): Whether to apply a sigmoid activation
                for output probabilities.
            dec_block (nn.Sequential): Sequential neural network layers
                for the decoder.
            sigmoid (nn.Sigmoid): Sigmoid activation function.

        """

        super(DecoderCONV, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.return_probs = return_probs
        self.latent_size = int(input_size / 4)
        self.output_channels = output_channels
        self.p = dropout

        self.dec_block = nn.Sequential(
            nn.Linear(self.latent_dim, 64 * self.latent_size ** 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64 * self.latent_size ** 2),
            View((-1, 64, self.latent_size, self.latent_size)),

            #nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.BatchNorm2d(64),

            #nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm2d(32),
            ResizeConv2d(in_channels=64, out_channels=32, kernel_size=3, scale_factor=2),
            #nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=self.p),

            #nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm2d(32),
            ResizeConv2d(in_channels=32, out_channels=32, kernel_size=3, scale_factor=2),
            #nn.ConvTranspose2d(32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=self.p),

            nn.Conv2d(in_channels=32, out_channels=self.output_channels, kernel_size=3, stride=1, padding=1),
            #nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, z):
        """Perform the forward pass for the decoder module.

        Args:
            z (torch.Tensor): Latent variable tensor.

        Returns:
            torch.Tensor: Reconstructed data tensor after passing
                through the decoder layers.
        """
        out = self.dec_block(z)
        if self.return_probs:
            out = self.sigmoid(out)
        return out

class EncoderRGB(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, input_channels=3):
        super(EncoderRGB, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.p = dropout
        self.input_channels = input_channels
        self.enc_out_dim = hidden_size

        self.enc_block = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),
            #nn.Dropout2d(p=self.p),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),
            #nn.Dropout2d(p=self.p),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),
            #nn.Dropout2d(p=self.p),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(2, stride=2),
            #nn.Dropout2d(p=self.p),
            
            nn.Flatten(),
            nn.Linear(256* int(self.input_size / 16) ** 2, hidden_size),
            nn.ReLU(),
            #nn.Dropout(p=self.p),
        )

    def forward(self, x):
        """Perform the forward pass for the encoder module.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Encoded tensor after passing through
                the encoder layers.
        """
        h = self.enc_block(x)
        return h


class DecoderRGB(nn.Module):
    def __init__(self, input_size, hidden_size,
                 latent_dim, dropout, output_channels=3, return_probs=True):
        super(DecoderRGB, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.return_probs = return_probs
        self.latent_size = int(input_size / 16)
        self.output_channels = output_channels
        self.p = dropout

        self.dec_block = nn.Sequential(
            nn.Linear(self.latent_dim, 256 * self.latent_size ** 2),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.BatchNorm1d(256 * self.latent_size ** 2),
            View((-1, 256, self.latent_size, self.latent_size)),
            
            ResizeConv2d(in_channels=256, out_channels=256, kernel_size=3, scale_factor=2),
            #nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=self.p),

            ResizeConv2d(in_channels=128, out_channels=128, kernel_size=3, scale_factor=2),
            #nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=self.p),
            
            ResizeConv2d(in_channels=64, out_channels=64, kernel_size=3, scale_factor=2),
            #nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=self.p),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=self.p),
            
            #nn.ConvTranspose2d(64, out_channels=32, kernel_size=4, stride=2, padding=1),
            #nn.ReLU(),
            #nn.BatchNorm2d(32),     
            
            ResizeConv2d(in_channels=32, out_channels=32, kernel_size=3, scale_factor=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=self.p),
            nn.Conv2d(32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=self.p),

            nn.Conv2d(in_channels=32, out_channels=self.output_channels, kernel_size=3, stride=1, padding=1)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        """Perform the forward pass for the decoder module.

        Args:
            z (torch.Tensor): Latent variable tensor.

        Returns:
            torch.Tensor: Reconstructed data tensor after passing
                through the decoder layers.
        """
        out = self.dec_block(z)
        if self.return_probs:
            out = self.sigmoid(out)
        return out
        
class EncoderCIFAR(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, input_channels=3):
        super(EncoderCIFAR, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.p = dropout
        self.input_channels = input_channels
        self.enc_out_dim = 128

        self.enc_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2), ##16x16
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2), ##8x8
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2), ##4x4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(2, stride=2), ##2x2

            nn.Flatten(),
            nn.Linear(256 * int(self.input_size / 16) ** 2, hidden_size),
        )

    def forward(self, x):
        """Perform the forward pass for the encoder module.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Encoded tensor after passing through
                the encoder layers.
        """
        h = self.enc_block(x)
        return h
        
        
class DecoderCIFAR(nn.Module):
    def __init__(self, input_size, hidden_size,
                 latent_dim, dropout, output_channels=3, return_probs=True):
        super(DecoderCIFAR, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.return_probs = return_probs
        self.latent_size = int(input_size / 16)
        self.output_channels = output_channels
        self.p = dropout

        self.dec_block = nn.Sequential(
            #nn.Linear(self.latent_dim, self.latent_dim),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm1d(self.latent_dim),
            
            nn.Linear(self.latent_dim, 256 * self.latent_size ** 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256 * self.latent_size ** 2),
            
            View((-1, 256,self.latent_size,self.latent_size)),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            ResizeConv2d(in_channels=128, out_channels=128, kernel_size=3, scale_factor=2),
            #nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),    ##8x8

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            ResizeConv2d(in_channels=64, out_channels=64, kernel_size=3, scale_factor=2),
            #nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),  ##16x16

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            ResizeConv2d(in_channels=32, out_channels=32, kernel_size=3, scale_factor=2),
            #nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),  ##32x32

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            ResizeConv2d(in_channels=32, out_channels=32, kernel_size=3, scale_factor=2),
            #nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        """Perform the forward pass for the decoder module.

        Args:
            z (torch.Tensor): Latent variable tensor.

        Returns:
            torch.Tensor: Reconstructed data tensor after passing
                through the decoder layers.
        """
        out = self.dec_block(z)
        if self.return_probs:
            out = self.sigmoid(out)
        return out
        
        
class EncoderResnet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, input_channels=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.p = dropout
        self.input_channels = input_channels
        self.enc_out_dim = hidden_size
        self.enc_block = ResNet18Enc(nc=self.input_channels, hidden_size=self.hidden_size, input_size=self.input_size)
        #self.enc_block = resnet18_features(hidden_size=self.hidden_size)
        
    def forward(self, x):
        h = self.enc_block(x)
        return h
        
class DecoderResnet(nn.Module):
    def __init__(self, input_size, hidden_size,
                 latent_dim, dropout, output_channels=3, return_probs=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.return_probs = return_probs
        #self.latent_size = int(input_size / 16)
        self.output_channels = output_channels
        self.p = dropout
        self.sigmoid = nn.Sigmoid()
        self.dec_block = ResNet18Dec(latent_dim = self.latent_dim, nc=self.output_channels, input_size=self.input_size)
        #self.dec_block = resnet18dec_features(latent_dim=self.latent_dim)
        
    def forward(self, z):
        out = self.dec_block(z)
        if self.return_probs:
            out = self.sigmoid(out)
        return(out)
        
        
class EncoderTNFA(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, input_channels=1):
        super(EncoderTNFA, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.p = dropout
        self.input_channels = input_channels
        self.enc_out_dim = hidden_size

        self.enc_block = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, stride=2),

            #nn.Dropout2d(p=self.p),
            nn.Flatten(),
            nn.Linear(512 * int(self.input_size / 32) ** 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.p),
            nn.Linear(hidden_size,hidden_size),
            
        )
            
    def forward(self,x):
         h = self.enc_block(x)
         return h
         
class DecoderTNFA(nn.Module):
    def __init__(self, input_size, hidden_size,
                 latent_dim, dropout, output_channels=1, return_probs=True):
        """Fully connected decoder module.

        Args:
            input_size (int): Dimensionality of the input data.
            hidden_size (int): Number of units in the hidden layer.
            latent_dim (int): Dimensionality of the latent space.
            return_probs (bool, optional): Whether to apply a sigmoid
                activation for output probabilities. Defaults to True.

        Attributes:
            input_size (int): Dimensionality of the input data.
            hidden_size (int): Number of units in the hidden layer.
            latent_dim (int): Dimensionality of the latent space.
            return_probs (bool): Whether to apply a sigmoid activation
                for output probabilities.
            dec_block (nn.Sequential): Sequential neural network layers
                for the decoder.
            sigmoid (nn.Sigmoid): Sigmoid activation function.

        """

        super(DecoderTNFA, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.return_probs = return_probs
        self.latent_size = int(input_size / 32)
        self.output_channels = output_channels
        self.p = dropout

        self.dec_block = nn.Sequential(
        
            nn.Linear(self.latent_dim, 512 * self.latent_size ** 2), #512
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512 * self.latent_size ** 2),
            nn.Dropout(p=self.p),
            View((-1, 512, self.latent_size, self.latent_size)),
            
            #nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm2d(256),
            ResizeConv2d(in_channels=512, out_channels=256, kernel_size=3, scale_factor=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            
            #nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm2d(128),
            ResizeConv2d(in_channels=256, out_channels=128, kernel_size=3, scale_factor=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            #nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm2d(64),
            ResizeConv2d(in_channels=128, out_channels=64, kernel_size=3, scale_factor=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),            

            #nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm2d(32),
            ResizeConv2d(in_channels=64, out_channels=32, kernel_size=3, scale_factor=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            #nn.Dropout2d(p=self.p),

            #nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm2d(32),
            ResizeConv2d(in_channels=32, out_channels=32, kernel_size=3, scale_factor=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            #nn.Dropout2d(p=self.p),

            nn.Conv2d(in_channels=32, out_channels=self.output_channels, kernel_size=3, stride=1, padding=1),
            #nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, z):
        """Perform the forward pass for the decoder module.

        Args:
            z (torch.Tensor): Latent variable tensor.

        Returns:
            torch.Tensor: Reconstructed data tensor after passing
                through the decoder layers.
        """
        out = self.dec_block(z)
        if self.return_probs:
            out = self.sigmoid(out)
        return out
