import numpy as np
import torch
import torch.nn.functional as F
import os
from collections import defaultdict

from .utils import random_cover


def imputate(args, model, dataloader, drop_rate):
    outputs = defaultdict(list)
    for _, (x, support, historcal_x) in enumerate(dataloader):
        # x:           B x L x D
        # support:     B x L x D
        # historcal_x: B x L x D
        x, support, historcal_x = x.to(args.device), support.to(args.device), historcal_x.to(args.device)
        model.eval()
        # noise_x:     B x L x D
        # noise_index: B x L x D
        noise_x, noise_index = random_cover(x, drop_rate, mask_value=0)
        historical_filled_x = torch.where(x == 0, historcal_x, noise_x)
        historical_filled_x = (0.1**0.5)*torch.normal(mean=torch.rand(1), std=torch.rand(historical_filled_x.shape)).to(args.device)
        historical_filled_x = noise_x
        output = model(historical_filled_x, support, 'benchmark')
        for k, v in output.items():
            outputs[k].append(v)
        outputs['x'].append(x)
        outputs['noise_index'].append(noise_index)
    return { k: torch.cat(v, dim=0) for k, v in outputs.items() }

        

def test(args, model, dataloader, drop_rate):
    imputated_data = imputate(args, model, dataloader, drop_rate)
    x = imputated_data['x']
    re_x = imputated_data['re_x']
    noise_index = imputated_data['noise_index']
    loss = F.mse_loss(re_x[noise_index], x[noise_index]).sum().item()  

    def get_index(s,e):
        new_index = noise_index.clone()
        new_index[:12 * s, :, :] = False
        new_index[12 * e:, :, :] = False
        return new_index

    m_loss = F.mse_loss(re_x[get_index(6, 10)], x[get_index(6, 10)]).sum().item()  
    e_loss = F.mse_loss(re_x[get_index(17, 20)], x[get_index(17, 20)]).sum().item()  
    f_loss = F.mse_loss(re_x[get_index(10, 17)], x[get_index(10, 17)]).sum().item()  
    n_loss = F.mse_loss(re_x[get_index(0, 6)], x[get_index(0, 6)]).sum().item()

    return loss, m_loss, e_loss, f_loss, n_loss



    print(drop_rate)
    total_loss = 0
    for _, (x, support, historcal_x) in enumerate(dataloader):

        x, support, historcal_x = x.to(args.device), support.to(args.device), historcal_x.to(args.device)

        model.eval()

        noise_x, noise_index = random_cover(x, drop_rate, mask_value=0)
        historical_filled_x = torch.where(x == 0, historcal_x, noise_x)

        filled_index = torch.where(noise_x != 0)
        historical_filled_x = (0.1**0.5)*torch.normal(mean=torch.rand(1), std=torch.rand(historical_filled_x.shape)).to(args.device)
        historical_filled_x = noise_x
        
        re_x, re_mask = model(historical_filled_x, support, 'benchmark')

        mask = torch.ones(support.shape).to(args.device)
        mask = torch.where(support > 0, mask, torch.zeros(mask.shape).to(args.device))

        print('re_mask filld number:', len(torch.where(re_mask > 0.5)[0]), 're_mask missing number:', len(torch.where(re_mask < 0.5)[0]))
        print('mask filled number:', len(torch.where(mask > 0)[0]), 'mask missing number:', len(torch.where(mask == 0)[0]))

        print('ob accuracy:', len(torch.where(mask[torch.where(re_mask < 0.5)] == 1)[0]), 'ub accuracy:', len(torch.where(mask[torch.where(re_mask < 0.5)] == 0)[0]))

        print('re_mask lower than 0.5 and support number higher than 1:', len(torch.where(support[torch.where(re_mask < 0.5)] > 0.0589)[0]))
        print('re_mask lower than 0.5 and support number lower than 1:', len(torch.where(support[torch.where(re_mask < 0.5)] < 0.0589)[0]))
        print('support number higher than 1:', len(torch.where(support > 0.0589)[0]))
        print('support number lower than 1:', len(torch.where(support < 0.0589)[0]))

        if args.freeze:
            np.save(f'{args.output_dir}/freeze_re_mask.npy' + '',re_mask.detach().cpu().numpy())
            np.save(f'{args.output_dir}/freeze_re_x.npy' + '',re_x.detach().cpu().numpy())
            np.save(f'{args.output_dir}/freeze_noise_index.npy' + '', torch.stack(noise_index).numpy())
        if args.dropd:
            np.save(f'{args.output_dir}/dropd_re_mask.npy' + '',re_mask.detach().cpu().numpy())
            np.save(f'{args.output_dir}/dropd_re_x.npy' + '',re_x.detach().cpu().numpy())
            np.save(f'{args.output_dir}/dropd_noise_index.npy' + '',torch.stack(noise_index).numpy())

        if args.multi:
            np.save(f'{args.output_dir}/multi_re_mask.npy' + '',re_mask.detach().cpu().numpy())
            np.save(f'{args.output_dir}/multi_re_x.npy' + '',re_x.detach().cpu().numpy())
            np.save(f'{args.output_dir}/multi_noise_index.npy' + '',torch.stack(noise_index).numpy())
        
        if args.model == 'MVAE':
            np.save(f'{args.output_dir}/mvae_re_mask.npy' + '',re_mask.detach().cpu().numpy())
            np.save(f'{args.output_dir}/mvae_re_x.npy' + '',re_x.detach().cpu().numpy())
            np.save(f'{args.output_dir}/mvae_noise_index.npy' + '',torch.stack(noise_index).numpy())


        loss = F.mse_loss(re_x[noise_index], x[noise_index]).sum().item()  
        def get_index(s,e):
            d1 = noise_index[0][(noise_index[0]>12*s) & (noise_index[0]<12*e)]
            d2 = noise_index[1][(noise_index[0]>12*s) & (noise_index[0]<12*e)]
            d3 = noise_index[2][(noise_index[0]>12*s) & (noise_index[0]<12*e)]
            return (d1, d2, d3)
        m_loss = F.mse_loss(re_x[get_index(6, 10)], x[get_index(6, 10)]).sum().item()  
        e_loss = F.mse_loss(re_x[get_index(17, 20)], x[get_index(17, 20)]).sum().item()  
        f_loss = F.mse_loss(re_x[get_index(10, 17)], x[get_index(10, 17)]).sum().item()  
        n_loss = F.mse_loss(re_x[get_index(0, 6)], x[get_index(0, 6)]).sum().item()  
        
        # total_loss += loss.sum().item()            

    # print('====> Test Average loss:', total_loss)
    return loss, m_loss, e_loss, f_loss, n_loss