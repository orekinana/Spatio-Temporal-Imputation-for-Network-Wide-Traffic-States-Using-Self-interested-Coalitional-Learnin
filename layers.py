import math
from random import sample
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import nn
from torch.nn import functional as F

class GraphConvolution(nn.Module): 

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(support, adj)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class VariationalAutoencoder(nn.Module):
    def __init__(self, node_num, hidden_features, vi_feature):
        super(VariationalAutoencoder, self).__init__()
        self.nodeNum = node_num

        self.encoder_net = nn.Sequential()
        self.encoder_net.add_module('fc_e', nn.Linear(node_num,hidden_features[0]))
        self.encoder_net.add_module('activation_e', nn.ReLU())
        for i in range(len(hidden_features)-1):
            self.encoder_net.add_module('fc_e'+str(i), nn.Linear(hidden_features[i],hidden_features[i+1]))
            self.encoder_net.add_module('activation_e'+str(i), nn.ReLU())

        self.mu_fc = nn.Linear(hidden_features[-1], vi_feature)
        self.logvar_fc = nn.Linear(hidden_features[-1], vi_feature)

        self.decoder_net = nn.Sequential()
        for i in range(len(hidden_features)-1):
            self.decoder_net.add_module('fc_d'+str(i), nn.Linear(hidden_features[-i-1],hidden_features[-i-2]))
            self.decoder_net.add_module('activation_d'+str(i), nn.ReLU())
        self.decoder_net.add_module('fc_d', nn.Linear(hidden_features[0],node_num))
        self.decoder_net.add_module('activation_d', nn.ReLU())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, input, mode):
        latent = self.encoder_net(input)
        mu = F.relu(self.mu_fc(latent))
        logvar = F.relu(self.logvar_fc(latent))
        sample_latent = self.reparameterize(mu, logvar)
        if mode == 'train':
            input_re = self.decoder_net(sample_latent)
        else:
            input_re = self.decoder_net(mu)
        return input_re, mu, logvar

class Discriminator(nn.Module):

    def __init__(self, node_num, kernel_size, vi_feature, hidden_features):
        super(Discriminator, self).__init__()
        self.node_num = node_num
        self.hidden_features = hidden_features

        self.x_conv1d = Convolutinon1D(in_features=node_num, kernal_size=kernel_size)
        self.x_gcn = GraphConvolution(in_features=node_num, out_features=node_num)

        self.s_conv1d = Convolutinon1D(in_features=node_num, kernal_size=kernel_size)
        self.s_gcn = GraphConvolution(in_features=node_num, out_features=node_num)

        self.fc_x = nn.Linear(node_num, node_num)
        self.fc_s = nn.Linear(node_num, node_num)

        self.w = nn.Parameter(torch.rand(node_num, node_num)*2-1)

        self.bias1 = Parameter(torch.rand(node_num)*2-1)
        self.bias2 = Parameter(torch.rand(node_num)*2-1)

        self.fc_logvar = nn.Linear(vi_feature, node_num)

        self.fusion = nn.Linear(2*node_num, node_num)

        self.encoder_net = nn.Sequential()
        self.encoder_net.add_module('fc_e', nn.Linear(node_num,hidden_features[0]))
        self.encoder_net.add_module('activation_e', nn.ReLU())
        for i in range(len(hidden_features)-1):
            self.encoder_net.add_module('fc_e'+str(i), nn.Linear(hidden_features[i],hidden_features[i+1]))
            self.encoder_net.add_module('activation_e'+str(i), nn.ReLU())

        self.mu_fc = nn.Linear(hidden_features[-1], vi_feature)
        self.logvar_fc = nn.Linear(hidden_features[-1], vi_feature)

        self.decoder_net = nn.Sequential()
        for i in range(len(hidden_features)-1):
            self.decoder_net.add_module('fc_d'+str(i), nn.Linear(hidden_features[-i-1],hidden_features[-i-2]))
            self.decoder_net.add_module('activation_d'+str(i), nn.ReLU())
        self.decoder_net.add_module('fc_d', nn.Linear(hidden_features[0],node_num))
        self.decoder_net.add_module('activation_d', nn.ReLU())

        self.transpose_conv = ConvolutinonTranspose1D(in_features=node_num, kernal_size=kernel_size)


    def forward(self, delta_x, support_emb, adj, logvar):

        delta_x_conv = self.x_conv1d(delta_x.permute(0, 2, 1))
        delta_x_gcn = self.x_gcn(torch.squeeze(delta_x_conv, -1), adj)

        s_conv = self.s_conv1d(support_emb.permute(0, 2, 1))
        s_gcn = self.s_gcn(torch.squeeze(s_conv, -1), adj)

        h_delta_x = F.relu(self.fc_x(delta_x_gcn))
        h_support = F.relu(self.fc_s(s_gcn))

        output = torch.cat((h_delta_x, h_support), -1)

        output = F.relu(self.fusion(output))

        output = self.encoder_net(output)
        output = self.decoder_net(output)
        
        output = self.transpose_conv(torch.unsqueeze(output, -1))
        output = output.permute(0, 2, 1)
        output = F.sigmoid(output) 
      
        return output

class Convolutinon1D(nn.Module):

    def __init__(self, in_features, kernal_size):
        super(Convolutinon1D, self).__init__()
        self.in_features = in_features
        self.kernal_size = kernal_size

        self.W = nn.Parameter(torch.rand(in_features, kernal_size))

    def forward(self, input):
        output = (input * self.W).sum(-1)
        return output

class ConvolutinonTranspose1D(nn.Module):
    def __init__(self, in_features, kernal_size):
        super(ConvolutinonTranspose1D, self).__init__()
        self.in_features = in_features
        self.kernal_size = kernal_size

        self.W = nn.Parameter(torch.rand(in_features, kernal_size)*2-1)
        self.bias = nn.Parameter(torch.rand(kernal_size)*2-1)

    def forward(self, input):
        output = torch.mul(input, self.W) + self.bias
        return output
