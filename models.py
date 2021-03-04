from math import exp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import layers

class MTGC(nn.Module):

    def __init__(self, args, adj, node_num, re_hidden, dis_hidden, vi_feature, kernel_size, alphi1, alphi2, alphi3):
        super(MTGC, self).__init__()
        # Mask Temporal Graph Convolution
        self.adj = adj
        self.node_num = node_num
        self.re_hidden = re_hidden
        self.dis_hidden = dis_hidden
        self.vi_feature = vi_feature
        self.kernel_size = kernel_size

        self.alphi1 = alphi1
        self.alphi2 = alphi2
        self.alphi3 = alphi3

        # feature extraction

        self.s_conv = layers.Convolutinon1D(in_features=node_num, kernal_size=self.kernel_size)
        self.s_gcn_e = layers.GraphConvolution(in_features=self.node_num, out_features=self.node_num)
        self.x_conv = layers.Convolutinon1D(in_features=node_num, kernal_size=self.kernel_size)
        self.x_gcn_e = layers.GraphConvolution(in_features=self.node_num, out_features=self.node_num)

        self.x_vae_tras = nn.Linear(self.kernel_size * self.node_num, self.node_num)
        self.s_vae_tras = nn.Linear(self.kernel_size * self.node_num, self.node_num)

        self.fusion = nn.Linear(self.node_num*2, self.node_num)
        
        # reconstruction network

        self.vae = layers.VariationalAutoencoder(node_num=self.node_num, hidden_features=self.re_hidden, vi_feature=self.vi_feature)

        self.x_vae_traspose = nn.Linear(self.node_num, self.kernel_size * self.node_num)

        # feature extraction
        self.gcn_d = layers.GraphConvolution(in_features=self.node_num, out_features=self.node_num)
        self.transpose_conv = layers.ConvolutinonTranspose1D(in_features=node_num, kernal_size=self.kernel_size)
        self.gcn_d_std = layers.GraphConvolution(in_features=self.node_num, out_features=self.node_num)
        self.transpose_conv_std = layers.ConvolutinonTranspose1D(in_features=node_num, kernal_size=self.kernel_size)

        # Discriminator
        self.mask_discriminator = layers.Discriminator(node_num=self.node_num, kernel_size=kernel_size, vi_feature=self.vi_feature, hidden_features=self.dis_hidden)

        self.args = args

    def set_requires_grad(self, *args, flag):
        for module in args:
            if module:
                for p in module.parameters():
                    p.requires_grad = flag

    def set_generator_requires_grad(self, flag):
        self.set_requires_grad(
            self.s_conv,
            self.s_gcn_e,
            self.x_conv,
            self.x_gcn_e,
            self.fusion,
            self.vae,
            self.gcn_d,
            self.transpose_conv,            
            flag=flag
        )

    def set_discriminator_requires_grad(self, flag):
        self.set_requires_grad(
            self.mask_discriminator,
            flag=flag
        )

    def forward(self, x, support, mode):

        x_he1 = self.x_conv(x.permute(0, 2, 1))
        s_he1 = self.s_conv(support.permute(0, 2, 1))

        x_he2 = self.x_gcn_e(x_he1.squeeze(-1), self.adj)
        s_he2 = self.s_gcn_e(s_he1.squeeze(-1), self.adj)
        
        he2 = torch.cat((x_he2, s_he2), -1)
        
        if self.args.model == 'MVAE':
            he2 = torch.cat((x_he1, s_he1), -1)

        
        he3 = F.relu(self.fusion(he2))

        hed, mu, logvar = self.vae(he3, mode)

        hd1 = self.gcn_d(hed, self.adj)
        
        x_re = self.transpose_conv(torch.unsqueeze(hd1, -1))
        x_re = x_re.permute(0, 2, 1)

        if self.args.model == 'MVAE':
            x_re = self.transpose_conv(torch.unsqueeze(hed, -1))
            x_re = x_re.permute(0, 2, 1)

        delta_x = (x_re - x)**2

        if self.args.freeze:
            delta_x = delta_x.detach()

        mask_re = self.mask_discriminator(delta_x, x, self.adj, logvar)

        if mode == 'benchmark':
            return {
                're_x': x_re,
                're_mask': mask_re
            }
        elif mode == 'test':
            return x_re, mask_re

        return x_re, mask_re, mu, logvar

    def loss(self, x, x_re, mask, mask_re, mu, logvar, filled_index, freeze, device):

        MSE = torch.nn.MSELoss(reduce=False, size_average=False)
        BCE = torch.nn.BCELoss(reduce=False, size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        boosting_weight = 1 / mask_re[filled_index]
        boosting_weight2 = -1 / (1-mask_re[torch.where(mask == 0)])


        if self.args.freeze:
            boosting_weight = boosting_weight.detach() 
            boosting_weight2 = boosting_weight2.detach()

        boosting_weight = torch.FloatTensor(np.clip(boosting_weight.cpu().detach().numpy(), 0,10)).to(device)
        boosting_weight2 = torch.FloatTensor(np.clip(boosting_weight2.cpu().detach().numpy(), 0,10)).to(device)
        
        l1 = self.alphi1 * MSE(x_re[filled_index], x[filled_index])
        OB = boosting_weight * MSE(x_re[filled_index], x[filled_index])
        UB = boosting_weight2 * MSE(x_re[torch.where(mask == 0)], x[torch.where(mask == 0)])

        l3 = self.alphi3 * KLD
        l2 = self.alphi2 * BCE(mask_re, mask)

        if self.args.freeze:
            return  l1.mean(), l2.mean(), l3, OB.mean(), 100000*(l1.sum() + l2.sum() + l3 + OB.sum() + UB.sum())
        
        if self.args.multi:
            return  l1.mean(), l2.mean(), l3, OB.sum(), 100000*(l1.sum() + l3 + l2.sum())
        
        if self.args.dropd:
            return  l1.mean(), l2.mean(), l3, OB.sum(), 100000*(l1.sum() + l3)
        
        if self.args.gan:
            return  l1.mean(), l2.mean(), l3, OB.sum(), 100000*(l2.sum())

    
