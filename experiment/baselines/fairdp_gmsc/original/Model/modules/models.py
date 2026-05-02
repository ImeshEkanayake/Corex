import torch
from torch import nn
from Model.modules.spectral_norm import spectral_norm

class NN(nn.Module):

    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, n_layer:int, dropout:float=None, clip:bool=False):
        super(NN, self).__init__()
        self.n_layers = n_layer
        if self.n_layers > 1:
            self.n_hid = n_layer - 2
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(in_features=input_dim, out_features=hidden_dim))
            for i in range(self.n_hid):
                layer = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
                nn.init.kaiming_normal(layer.weight, nonlinearity="relu")
                self.layers.append(layer)
            self.out_layer = nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=False)
        else:
            self.out_layer = nn.Linear(in_features=input_dim, out_features=output_dim, bias=False)
        self.activation = torch.nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x, mode:str='normal'):
        if self.n_layers > 1:
            h = x
            for i in range(0, self.n_layers-1):
                h = self.layers[i](h)
                h = self.activation(h)
                if self.dropout is not None:
                    h = self.dropout(h)
            if mode == 'normal':       
                h = self.out_layer(h)
                return h
            else:
                x_in = h
                x_out = self.out_layer(h)
                return x_in, x_out
        else:
            h = x
            if mode == 'normal':       
                h = self.out_layer(h)
                return h
            else:
                x_in = h
                x_out = self.out_layer(h)
                return x_in, x_out
        
class CNN(nn.Module):

    def __init__(self, channel:list, hid_dim:list, img_size:int, channel_in:int, out_dim:int, 
                 debug:int=1, kernel_size:int=5, padding:int=0, stride:int=1, dropout:float=0.2, clip:bool=False):
        super(CNN, self).__init__()

        # general
        self.layer_name = []
        self.lay_out_size = {}
        self.inter_act = nn.ReLU()
        if debug:
            self.img_slist = []

        # conv part
        img_s = img_size
        if debug:
            self.img_slist.append(img_s)
        self.cnn_layers = nn.ModuleList()

        self.cnn_layers.append(nn.Conv2d(channel_in, channel[0], kernel_size=kernel_size, padding=padding, stride=stride))
        img_s = int((img_s + 2*padding - 1*(kernel_size - 1) - 1) / stride + 1)
        self.num_trans = 1
        if debug:
            self.img_slist.append(img_s)
        self.layer_name.append('conv_1')
        self.lay_out_size['conv_1'] = img_s*img_s*channel[0]

        self.cnn_layers.append(nn.MaxPool2d(2))
        self.layer_name.append(f'pool_1')
        img_s = int((img_s + 2*padding - 1*(2 - 1) - 1) / 2 + 1)
        if debug:
            self.img_slist.append(img_s)

        for i in range(1, len(channel)):
            self.cnn_layers.append(nn.Conv2d(channel[i-1], channel[i], kernel_size=kernel_size, padding=padding, stride=stride))
            self.num_trans += 1
            self.layer_name.append(f'conv_{i+1}')
            img_s = int((img_s + 2*padding - 1*(kernel_size - 1) - 1) / stride + 1)
            self.lay_out_size[f'conv_{i+1}'] = img_s*img_s*channel[0]
            if debug:
                self.img_slist.append(img_s)
            self.cnn_layers.append(nn.MaxPool2d(2))
            self.layer_name.append(f'pool_{i+1}')
            img_s = int((img_s + 2*padding - 1*(2 - 1) - 1) / 2 + 1)
            if debug:
                self.img_slist.append(img_s)

        self.flatten_size = img_s*img_s*channel[-1]

        # linear part
        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(nn.Linear(self.flatten_size, hid_dim[0]))
        self.layer_name.append('linr_1')
        self.num_trans += 1
        for i in range(1, len(hid_dim)):
            self.linear_layers.append(nn.Linear(hid_dim[i-1], hid_dim[i]))
            self.num_trans += 1
            self.layer_name.append(f'linr_{i+1}')
        if clip:
            self.out_layer = spectral_norm(nn.Linear(hid_dim[-1], out_dim, bias=False), n_power_iterations=100)
        else:
            self.out_layer = nn.Linear(hid_dim[-1], out_dim, bias=False)
        self.conv_drop = nn.Dropout2d(dropout)
        self.linr_drop = nn.Dropout(dropout)
        self.debug = debug

    def forward(self, x, mode:str='normal'):
        h = x
        for i, layer in enumerate(self.cnn_layers):
            name = self.layer_name[i]
            if ('conv' in name) & (i > 0):
                self.conv_drop(h)
            h = layer(h)
            if ('pool' in name) or (i==0):
                h = self.inter_act(h)

        h = h.view(-1, self.flatten_size)
        for i, layer in enumerate(self.linear_layers):
            self.linr_drop(h)
            h = layer(h)
        h = self.linr_drop(h)
        if mode == 'normal':       
            h = self.out_layer(h)
            return h
        else:
            x_in = h
            x_out = self.out_layer(h)
            return x_in, x_out