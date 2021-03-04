import time
import argparse
import numpy as np
import os
import random
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models import MTGC
from layers import VariationalAutoencoder
import JNdata as JN
import model_configs

from code.utils import setup_random_seed
from code.utils import random_cover
from code.evaluate import test

NOISE = 0
([],[],[])

class Trainer():
    def __init__(self, model, args):
        self.model = model
        self.args = args

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.initial_lr, weight_decay=0.1)
        print("Trainer initial finish!")

    def train(self, train_dataloader, test_dataloader):
        print("Training Start!")
        t_start = time.time()
        total_tr_loss = 0
        loss_trend = []
        for epoch in range(self.args.epochs):
            tr_loss = 0
            tr_mse, tr_bce, tr_kld, tr_ob = 0, 0, 0, 0

            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.initial_lr, weight_decay=0.1)
            for _, (x, support, historcal_x) in enumerate(train_dataloader):
                x, support, historcal_x = x.to(self.args.device), support.to(self.args.device), historcal_x.to(self.args.device)
                self.model.train()
                historcal_x = (0.1**0.5) * torch.normal(mean=torch.rand(1), std=torch.rand(historcal_x.shape)).to(self.args.device)

                if NOISE == 1:
                    noise_x, _ = random_cover(x, self.args.train_drop_rate, mask_value=0)
                    historical_filled_x = torch.where(x == 0, historcal_x, noise_x)
                else:
                    historical_filled_x = torch.where(x == 0, historcal_x, x)
                historical_filled_x = x

                filled_index = torch.where(x != 0)
                re_x, re_mask, mu, logvar = self.model(historical_filled_x, support, 'train')

                mask = torch.ones(support.shape).to(self.args.device)
                mask = torch.where(support > 0, mask, torch.zeros(mask.shape).to(self.args.device))

                mse, bce, kld, ob, loss = self.model.loss(x, re_x, mask, re_mask, mu, logvar, filled_index, args.freeze, self.args.device)

                loss = loss.sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tr_mse += mse.item()
                tr_bce += bce.item()
                tr_kld += kld.item()
                tr_ob += ob.item()
            if epoch == 150:
                print('====> Test Average loss:', runer.test(test_dataloader, self.args.test_drop_rate))
            print('====> Epoch: {} x reconstruction loss: {:.4f}, mask reconstruction loss: {:.4f}, kld loss: {:.4f}, ob loss: {:.4f}'.format(epoch, tr_mse/ len(train_dataloader), tr_bce/ len(train_dataloader), tr_kld/ len(train_dataloader), tr_ob/ len(train_dataloader)))
            total_tr_loss += tr_loss / len(train_dataloader)
            loss_trend.append(tr_mse/ len(train_dataloader))
        print("Training Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_start))
        # if self.args.freeze:
        #     np.save(f'{os.getcwd()}/data/model/freeze_trend.npy' + '',np.array(loss_trend))
        # if self.args.multi:
        #     np.save(f'{os.getcwd()}/data/model/multi_trend.npy' + '',np.array(loss_trend))
        # if self.args.dropd:
        #     np.save(f'{os.getcwd()}/data/model/dropd_trend.npy' + '',np.array(loss_trend))



    def test(self, dataloader, drop_rate):
        self.args.output_dir = f'xxx'
        return test(self.args, self.model, dataloader, drop_rate)

        # print(drop_rate)
        # total_loss = 0
        # for _, (x, support, historcal_x) in enumerate(dataloader):

        #     x, support, historcal_x = x.to(self.args.device), support.to(self.args.device), historcal_x.to(self.args.device)

        #     self.model.eval()

        #     noise_x, noise_index = random_cover(x, drop_rate, mask_value=0)
        #     historical_filled_x = torch.where(x == 0, historcal_x, noise_x)

        #     filled_index = torch.where(noise_x != 0)
        #     historical_filled_x = (0.1**0.5)*torch.normal(mean=torch.rand(1), std=torch.rand(historical_filled_x.shape)).to(self.args.device)
        #     historical_filled_x = noise_x
            
        #     re_x, re_mask, _, _, = self.model(historical_filled_x, support, adj, 'test')

        #     mask = torch.ones(support.shape).to(self.args.device)
        #     mask = torch.where(support > 0, mask, torch.zeros(mask.shape).to(self.args.device))

        #     print('re_mask filld number:', len(torch.where(re_mask > 0.5)[0]), 're_mask missing number:', len(torch.where(re_mask < 0.5)[0]))
        #     print('mask filled number:', len(torch.where(mask > 0)[0]), 'mask missing number:', len(torch.where(mask == 0)[0]))

        #     print('ob accuracy:', len(torch.where(mask[torch.where(re_mask < 0.5)] == 1)[0]), 'ub accuracy:', len(torch.where(mask[torch.where(re_mask < 0.5)] == 0)[0]))

        #     print('re_mask lower than 0.5 and support number higher than 1:', len(torch.where(support[torch.where(re_mask < 0.5)] > 0.0589)[0]))
        #     print('re_mask lower than 0.5 and support number lower than 1:', len(torch.where(support[torch.where(re_mask < 0.5)] < 0.0589)[0]))
        #     print('support number higher than 1:', len(torch.where(support > 0.0589)[0]))
        #     print('support number lower than 1:', len(torch.where(support < 0.0589)[0]))

        #     if self.args.freeze:
        #         np.save(f'{os.getcwd()}/data/model/freeze_re_mask.npy' + '',re_mask.detach().cpu().numpy())
        #         np.save(f'{os.getcwd()}/data/model/freeze_re_x.npy' + '',re_x.detach().cpu().numpy())
        #         np.save(f'{os.getcwd()}/data/model/freeze_noise_index.npy' + '', torch.stack(noise_index).numpy())
        #     if self.args.dropd:
        #         np.save(f'{os.getcwd()}/data/model/dropd_re_mask.npy' + '',re_mask.detach().cpu().numpy())
        #         np.save(f'{os.getcwd()}/data/model/dropd_re_x.npy' + '',re_x.detach().cpu().numpy())
        #         np.save(f'{os.getcwd()}/data/model/dropd_noise_index.npy' + '',torch.stack(noise_index).numpy())

        #     if self.args.multi:
        #         np.save(f'{os.getcwd()}/data/model/multi_re_mask.npy' + '',re_mask.detach().cpu().numpy())
        #         np.save(f'{os.getcwd()}/data/model/multi_re_x.npy' + '',re_x.detach().cpu().numpy())
        #         np.save(f'{os.getcwd()}/data/model/multi_noise_index.npy' + '',torch.stack(noise_index).numpy())
            
        #     if self.args.model == 'MVAE':
        #         np.save(f'{os.getcwd()}/data/model/mvae_re_mask.npy' + '',re_mask.detach().cpu().numpy())
        #         np.save(f'{os.getcwd()}/data/model/mvae_re_x.npy' + '',re_x.detach().cpu().numpy())
        #         np.save(f'{os.getcwd()}/data/model/mvae_noise_index.npy' + '',torch.stack(noise_index).numpy())
        [
                [],
                [],
                [],

        ]

        #     loss = F.mse_loss(re_x[noise_index], x[noise_index]).sum().item()  
        #     def get_index(s,e):
        #         d1 = noise_index[0][(noise_index[0]>12*s) & (noise_index[0]<12*e)]
        #         d2 = noise_index[1][(noise_index[0]>12*s) & (noise_index[0]<12*e)]
        #         d3 = noise_index[2][(noise_index[0]>12*s) & (noise_index[0]<12*e)]
        #         return (d1, d2, d3)
        #     m_loss = F.mse_loss(re_x[get_index(6, 10)], x[get_index(6, 10)]).sum().item()  
        #     e_loss = F.mse_loss(re_x[get_index(17, 20)], x[get_index(17, 20)]).sum().item()  
        #     f_loss = F.mse_loss(re_x[get_index(10, 17)], x[get_index(10, 17)]).sum().item()  
        #     n_loss = F.mse_loss(re_x[get_index(0, 6)], x[get_index(0, 6)]).sum().item()  
            
        #     # total_loss += loss.sum().item()            

        # # print('====> Test Average loss:', total_loss)
        # return loss, m_loss, e_loss, f_loss, n_loss

    

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

if __name__ == "__main__": 

    # Training settings

    parser = argparse.ArgumentParser(description='MTGC training params')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=777, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--initial_lr', type=int, default=0.001, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--resume-last', action='store_true',
                        help='LOAD TRAIN PATH')
    parser.add_argument('--test-only', action='store_true',
                        help='test only')
    
    parser.add_argument('--freeze', action='store_true',
                        help='freeze')
    parser.add_argument('--multi', action='store_true',
                        help='multi-task version')
    parser.add_argument('--gan', action='store_true',
                        help='multi-task version')
    parser.add_argument('--dropd', action='store_true',
                        help='multi-task version')
    
    parser.add_argument('--train-drop-rate', type=float, default=0, metavar='N',
                        help='train_drop_rate')
    parser.add_argument('--test-drop-rate', type=float, default=0.1, metavar='N',
                        help='test_drop_rate')
    parser.add_argument('--model', type=str, default='MTGC', metavar='N',
                        help='choose the model will train')

    parser.add_argument('--traning-op', type=str, default='20170903')
    parser.add_argument('--traning-ed', type=str, default='20170930')
    parser.add_argument('--testing-op', type=str, default='20170901')
    parser.add_argument('--testing-ed', type=str, default='20170902')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    setup_random_seed(seed=args.seed)

    torch.manual_seed(args.seed)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Load static adj matrix
    adj = JN.load_adj().to(args.device)

    # defalt JN1
    model = MTGC(args, adj, **model_configs.MODEL_CONFIGS['JN2'])
    model = model.to(args.device)
    runer = Trainer(model, args)

    savemodels = os.listdir(f'{os.getcwd()}/data/model/')
    latestmodel = f'{os.getcwd()}/data/model/' + max(savemodels)
    # latestmodel = f'{os.getcwd()}/data/model/' + 'MTGC_2020-11-18 18:25:31.model'
    print('latestmodel:', latestmodel)

    train_dataloader = DataLoader(JN.JN(args, mode='train', drop_rate=args.train_drop_rate), batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(JN.JN(args, mode='test', drop_rate=args.test_drop_rate), batch_size=args.batch_size, shuffle=False)



    if args.test_only:
        runer.load(latestmodel)
        loss, m_loss, e_loss, f_loss, n_loss = runer.test(test_dataloader, args.test_drop_rate)
        print('====> Test Average loss:', loss, m_loss, e_loss, f_loss, n_loss)
    else:
        if args.resume_last:
            runer.load(latestmodel)
        runer.train(train_dataloader, test_dataloader)
        runer.save(f'{os.getcwd()}/data/model/MTGC_' + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())) + '.model')
        loss, m_loss, e_loss, f_loss, n_loss = runer.test(test_dataloader, args.test_drop_rate)
        print('====> Test Average loss:', loss, m_loss, e_loss, f_loss, n_loss)


    
