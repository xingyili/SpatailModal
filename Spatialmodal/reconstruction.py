# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import scanpy as sc
import torch.optim as optim
import os
from torch.backends import cudnn
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


loss1 = nn.L1Loss()
loss2 = nn.MSELoss()


class Encoder(nn.Module):
    def __init__(self, gene_number, X_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(gene_number, 500)
        self.fc1_bn = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 50)
        self.fc2_bn = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50, 10)
        self.fc3_bn = nn.BatchNorm1d(10)
        self.fc4 = nn.Linear(10, X_dim)

    def forward(self, input, relu):
        h1 = F.relu(self.fc1_bn(self.fc1(input)))
        h2 = F.relu(self.fc2_bn(self.fc2(h1)))
        h3 = F.relu(self.fc3_bn(self.fc3(h2)))
        if relu:
            return F.relu(self.fc4(h3))
        else:
            return self.fc4(h3)


class Decoder(nn.Module):
    def __init__(self, gene_number, X_dim):
        super(Decoder, self).__init__()
        self.fc6 = nn.Linear(X_dim, 50)
        self.fc6_bn = nn.BatchNorm1d(50)
        self.fc7 = nn.Linear(50, 500)
        self.fc7_bn = nn.BatchNorm1d(500)
        self.fc8 = nn.Linear(500, 1000)
        self.fc8_bn = nn.BatchNorm1d(1000)
        self.fc9 = nn.Linear(1000, gene_number)

    def forward(self, z, relu):
        h6 = F.relu(self.fc6_bn(self.fc6(z)))
        h7 = F.relu(self.fc7_bn(self.fc7(h6)))
        h8 = F.relu(self.fc8_bn(self.fc8(h7)))
        if relu:
            return F.relu(self.fc9(h8))
        else:
            return self.fc9(h8)


class RF(nn.Module):
    def __init__(self,
                 adata,
                 epochs=5,
                 device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu'),
                 seed=1234,
                 learning_rate=0.001,
                 step_size=500,
                 gamma=1,
                 relu='True',
                 w_recon=0.1,
                 w_w=0.1,
                 w_l1=0.1,
                 cs=77,
                 X_dim=2
                 ):
        super(RF, self).__init__()
        self.adata = adata
        self.in_feature = adata.obsm['emb'].shape[1]
        self.X_dim = X_dim
        self.out_feature = adata.obsm['feat'].shape[1]
        # self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.random_seed = seed
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.relu = relu
        self.w_recon = w_recon
        self.w_w = w_w
        self.w_l1 = w_l1
        self.epochs = epochs
        self.cs = cs
        self.emb = torch.FloatTensor(adata.obsm['emb']).to(self.device)
        self.feat = torch.FloatTensor(adata.obsm['feat']).to(self.device)
        fix_seed(self.random_seed)
        self.coord = adata.obsm['coord']
        self.coord_norm = MinMaxScaler().fit_transform(self.coord)

    def train(self, mo, z=None):
        self.encoder = Encoder(self.in_feature, self.X_dim).to(self.device)
        self.decoder = Decoder(self.out_feature, self.X_dim).to(self.device)
        if mo=='2D':
            self.pos = torch.FloatTensor(self.coord_norm).to(self.device)
        else:
            self.pos = torch.FloatTensor(np.hstack((self.coord,z))).to(self.device)

        enc_optim = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        dec_optim = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        enc_sche = optim.lr_scheduler.StepLR(enc_optim, step_size=self.step_size, gamma=self.gamma)
        dec_sche = optim.lr_scheduler.StepLR(dec_optim, step_size=self.step_size, gamma=self.gamma)

        self.encoder.train()
        self.decoder.train()

        with tqdm(range(self.epochs), total=self.epochs, desc='Epochs') as epoch:
            for j in epoch:
                train_loss = []
                train_lc_loss = []
                train_re_loss = []
                torch.set_grad_enabled(True)

                enc_optim.zero_grad()
                dec_optim.zero_grad()

                latent = self.encoder(self.emb, self.relu)
                latent = latent.view(-1, self.X_dim)
                latent_loss = loss1(latent, self.pos)

                xrecon = self.decoder(latent, self.relu)
                recon_loss = loss2(xrecon, self.feat)
                total_loss = 0.1 * latent_loss + 0.1 * self.w_recon * recon_loss
                total_loss.backward()

                enc_optim.step()
                dec_optim.step()
                enc_sche.step()
                dec_sche.step()

                # 记录损失值
                train_lc_loss.append(latent_loss.item())
                train_re_loss.append(recon_loss.item())
                train_loss.append(total_loss.item())

                epoch_info = 'la:%.4f,rec:%.4f,to:%.4f' % \
                             (torch.mean(torch.FloatTensor(train_lc_loss)),
                              torch.mean(torch.FloatTensor(train_re_loss)),
                              torch.mean(torch.FloatTensor(train_loss)))
                epoch.set_postfix_str(epoch_info)

                # pbar.update(1)

        self.decoder.eval()
        if mo=='3D':
            new_pos = np.hstack((self.coord,z))
            z_values = new_pos[:, 2]  # 或 z_normalized.flatten()
            diffs = np.diff(z_values)
            common_diff = max(diffs)-min(diffs)
            new_pos = new_pos[new_pos[:,2]>np.min(new_pos[:,2])]
            new_pos[:,2] = new_pos[:,2]-common_diff/2
            new_pos = torch.FloatTensor(new_pos).to(self.device)
            all_pos = torch.cat((self.pos, new_pos), dim=0)
            xrecon = self.decoder(all_pos, self.relu).cpu().detach().numpy()
            adata_re = sc.AnnData(xrecon)
            adata_re.obsm['coord'] = all_pos.cpu().numpy()
            adata_re.uns['raw_slide'] = self.coord
            adata_re.var_names = self.adata.var_names
            return adata_re
        else:
            xrecon = self.decoder(self.pos, self.relu).cpu().detach().numpy()
            self.adata.obsm['rec'] = xrecon
            return self.adata




