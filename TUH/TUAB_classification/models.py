
import math
import copy

import torch
from torch import nn
from torchvision import models

import numpy as np
import matplotlib.pyplot as plt

from braindecode.models import ShallowFBCSPNet, Deep4Net, EEGResNet, EEGNetv4, TCN


def get_model(cfg):
    """Return the dataset class with the given name
    
    Args:
        cfg
    """
    if cfg.args.model_name not in globals():
        raise NotImplementedError("model not found: {}".format(cfg.args.model_name))

    model_fn = globals()[cfg.args.model_name]

    return model_fn(cfg)

class deep4(nn.Module):
    """ The DEEP4 model
    This is from the Braindecode package:
        https://github.com/braindecode/braindecode
    
    Args:
        cfg
    Attributes:
        forward pass
    """
    
    def __init__(self, cfg):
        super(deep4, self).__init__()


        self.model = Deep4Net(
            in_chans = cfg.data.in_chans,
            n_classes = cfg.data.n_classes,
            input_window_samples = cfg.data.input_window_samples,
            final_conv_length='auto',
            n_filters_time=32,
            n_filters_spat=32,
            filter_time_length=10,
            pool_time_length=3,
            pool_time_stride=3,
            n_filters_2=64,
            filter_length_2=10,
            n_filters_3=128,
            filter_length_3=10,
            n_filters_4=256,
            filter_length_4=10
        )

        # # Delete undesired layers
        # self.classifier = copy.deepcopy(self.model.conv_classifier)
        # del self.model.conv_classifier
        # del self.model.softmax
        # del self.model.squeeze
        
    def forward(self, input):

        # Forward pass
        out = self.model(input)
        # features = self.model(input.permute((0, 2, 1)))
        # out = self.classifier(features)

        # # Remove all extra dimension and Add the time prediction dimension
        # out, features = torch.flatten(out, start_dim=1), torch.flatten(features, start_dim=1)
        # out, features = out.unsqueeze(1), features.unsqueeze(1)

        return out#, features

class EEGNet(nn.Module):
    """ The EEGNet model
    This is a really small model ~3k parameters.
    This is from the Braindecode package:
        https://github.com/braindecode/braindecode
    
    Args:
        cfg
    Attributes:
        forward pass
    """
    
    def __init__(self, cfg):
        super(EEGNet, self).__init__()
 
        scale = 1
        self.model = EEGNetv4(
            in_chans = cfg.data.in_chans,
            n_classes = cfg.data.n_classes,
            input_window_samples = cfg.data.input_window_samples,
            final_conv_length='auto',
            F1=8*scale,
            D=2*scale,
            F2=16*scale*scale, #usually set to F1*D (?)
            kernel_length=64*scale,
            third_kernel_size=(8, 4),
            drop_prob=cfg.args.drop_prob,
        )

        
    def forward(self, input):

        # Forward pass
        out = self.model(input)

        return out 

class Sim_CNN(nn.Module):
    """ Hand-tuned architecture for extracting representation from MNIST images
    This was adapted from :
        https://github.com/facebookresearch/DomainBed
    In our context, it is used to extract the representation from the images which are fed to a recurrent model such as an LSTM
    Args:
        dataset (Multi_Domain_Dataset): dataset that we will be training on
        model_hparams (dict): The hyperparameters for the model.
        input_size (int, optional): The size of the input to the model. Defaults to None. If None, the input size is calculated from the dataset.
    """
    def __init__(self, cfg):
        super(Sim_CNN, self).__init__()

        # Make CNN
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, (16,16), 1, padding=1),
            # nn.Tanh(),
            nn.BatchNorm2d(64, momentum=0.01, affine=True, eps=1e-3),
            nn.Conv2d(64, 32, (8,8), stride=1, padding=1),
            # nn.Tanh(),
            nn.BatchNorm2d(32, momentum=0.01, affine=True, eps=1e-3),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(32, 32, (4,8), 1, padding=1),
            # nn.Tanh(),
            nn.BatchNorm2d(32, momentum=0.01, affine=True, eps=1e-3),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(32, 64, (4,8), 1, padding=1),
            # nn.SELU(),
            nn.BatchNorm2d(64, momentum=0.01, affine=True, eps=1e-3),
            ## more layer 
            nn.Conv2d(64, 1, (2,8), 1, padding=1),
            # nn.SELU(),
            nn.BatchNorm2d(1, momentum=0.01, affine=True, eps=1e-3),
        )

        # Make FCC layers
        self.FCC = nn.Sequential(
            nn.Linear(2*170, 2), #cfg.data.n_classes),
            # nn.Tanh(),
            # nn.Linear(64, 32),
            # nn.Mish(),
        )

        self.log_sm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """ Forward pass through the model
        Args:
            x (torch.Tensor): The input to the model.
        Returns:
            torch.Tensor: The output representation of the model.
        """
        x = torch.unsqueeze(x, dim=1)
        x = self.conv(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.FCC(x)
        x = self.log_sm(x)
        return x

class LSTM(nn.Module):
    """ A simple LSTM model
    Args:
        dataset (Multi_Domain_Dataset): dataset that we will be training on
        model_hparams (dict): The hyperparameters for the model.
        input_size (int, optional): The size of the input to the model. Defaults to None. If None, the input size is calculated from the dataset.
    Attributes:
        state_size (int): The size of the hidden state of the LSTM.
        recurrent_layers (int): The number of recurrent layers stacked on each other.
        hidden_depth (int): The number of hidden layers of the classifier MLP (after LSTM).
        hidden_width (int): The width of the hidden layers of the classifier MLP (after LSTM).
    
    Notes:
        All attributes need to be in the model_hparams dictionary.
    """
    def __init__(self, dataset, model_hparams, input_size=None):
        super(LSTM, self).__init__()

        ## Save stuff
        # Model parameters
        self.state_size = model_hparams['state_size']
        self.hidden_depth = model_hparams['hidden_depth']
        self.hidden_width = model_hparams['hidden_width']
        self.recurrent_layers = model_hparams['recurrent_layers']

        # Dataset parameters
        self.input_size = np.prod(dataset.INPUT_SHAPE) if input_size is None else input_size
        self.output_size = dataset.OUTPUT_SIZE

        ## Recurrent model
        self.lstm = nn.LSTM(self.input_size, self.state_size, self.recurrent_layers, batch_first=True)

        ## Classification model
        layers = []
        if self.hidden_depth == 0:
            layers.append( nn.Linear(self.state_size, self.output_size) )
        else:
            layers.append( nn.Linear(self.state_size, self.hidden_width) )
            for i in range(self.hidden_depth-1):
                layers.append( nn.Linear(self.hidden_width, self.hidden_width) )
            layers.append( nn.Linear(self.hidden_width, self.output_size) )
        
        seq_arr = []
        for i, lin in enumerate(layers):
            seq_arr.append(lin)
            if i != self.hidden_depth:
                seq_arr.append(nn.ReLU(True))
        self.classifier = nn.Sequential(*seq_arr)

    def forward(self, input, time_pred):
        """ Forward pass of the model
        Args:
            input (torch.Tensor): The input to the model.
            time_pred (torch.Tensor): The time prediction of the input.
        Returns:
            torch.Tensor: The output of the model.
        """

        # Setup array
        hidden = self.initHidden(input.shape[0], input.device)

        # Forward propagate LSTM
        input = input.view(input.shape[0], input.shape[1], -1)
        features, hidden = self.lstm(input, hidden)

        # Make prediction with fully connected
        all_out = torch.zeros((input.shape[0], time_pred.shape[0], self.output_size)).to(input.device)
        all_features = torch.zeros((input.shape[0], time_pred.shape[0], features.shape[-1])).to(input.device)
        for i, t in enumerate(time_pred):
            output = self.classifier(features[:,t,:])
            all_out[:,i,...] = output
            all_features[:,i,...] = features[:,t,...]

        return all_out, all_features

    def initHidden(self, batch_size, device):
        """ Initialize the hidden state of the LSTM with a normal distribution
        Args:
            batch_size (int): The batch size of the model.
            device (torch.device): The device to use.
        """
        return (torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device), 
                torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device))


    # model = Deep4Net(
    #         in_chans = batch_X.shape[1],
    #         n_classes = 2,
    #         input_window_samples=batch_X.shape[2],
    #         final_conv_length='auto',
    #         # n_filters_time=64,
    #         # # n_filters_spat=32,
    #         # filter_time_length=8,
    #         # pool_time_length=4,
    #         # # pool_time_stride=3,
    #         # # n_filters_2=64,
    #         # # filter_length_2=10,
    #         # n_filters_3=256,
    #         # # filter_length_3=10,
    #         # # n_filters_4=256,
    #         # # filter_length_4=10
    #     )
    # model = TCN(
    #         n_in_chans=batch_X.shape[1] ,
    #         n_outputs=2,
    #         n_filters=55,
    #         n_blocks=5,
    #         kernel_size=8,
    #         drop_prob=0.05270154233150525,
    #         add_log_softmax=True
    #     )
    #
    #EEGNetv4
    # model = EEGNetv4(
    #     in_chans = batch_X.shape[1],
    #     n_classes = 2,
    #     input_window_samples=batch_X.shape[2],
    #     final_conv_length='auto', 
    #     drop_prob=0.25,
    # )  

if __name__ == '__main__':
    x = input = torch.randn([1024, 21, 750])
    # x = torch.unsqueeze(x, dim=1)
    print(x.shape)
    cfg = 10
    model = Sim_CNN(cfg)
    y = model(x)
    print(y.shape)
