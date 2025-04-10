import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class DishTS(nn.Module):
    def __init__(self, args):
        super().__init__()
        init = 'standard'   # args.dish_init #'standard', 'avg' or 'uniform'
        activate = True
        n_series = args.enc_in    #args.n_series # number of series
        lookback = args.seq_len    #args.seq_len # lookback length

        self.device = self._acquire_device()

        if init == 'standard':
            self.reduce_mlayer = nn.Parameter(torch.rand(n_series, lookback, 2)/lookback)
        elif init == 'avg':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2)/lookback)
        elif init == 'uniform':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2)/lookback+torch.rand(n_series, lookback, 2)/lookback)
        self.gamma, self.beta = nn.Parameter(torch.ones(n_series)), nn.Parameter(torch.zeros(n_series))
        self.activate = activate

    def forward(self, batch_x, mode='forward', dec_inp=None):
        if mode == 'forward':
            # batch_x: B*L*D || dec_inp: B*?*D (for xxformers)
            self.preget(batch_x)
            batch_x = self.forward_process(batch_x)
            dec_inp = None if dec_inp is None else self.forward_process(dec_inp)
            return batch_x, dec_inp
        elif mode == 'inverse':
            # batch_x: B*H*D (forecasts)
            batch_y = self.inverse_process(batch_x)
            return batch_y

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def preget(self, batch_x):
        x_transpose = batch_x.permute(2,0,1)
        reduce_mlayer = self.reduce_mlayer.to(self.device)
        theta = torch.bmm(x_transpose, reduce_mlayer).permute(1,2,0)
        if self.activate:
            theta = F.gelu(theta)
        self.phil, self.phih = theta[:,:1,:], theta[:,1:,:] 
        self.xil = torch.sum(torch.pow(batch_x - self.phil,2), axis=1, keepdim=True) / (batch_x.shape[1]-1)
        self.xih = torch.sum(torch.pow(batch_x - self.phih,2), axis=1, keepdim=True) / (batch_x.shape[1]-1)

    def forward_process(self, batch_input):
        #print(batch_input.shape, self.phil.shape, self.xih.shape)
        temp = (batch_input - self.phil)/torch.sqrt(self.xil + 1e-8)
        gamma = self.gamma.to(self.device)
        beta = self.beta.to(self.device)
        rst = temp.mul(gamma) + beta
        return rst
    
    def inverse_process(self, batch_input):
        gamma = self.gamma.to(self.device)
        beta = self.beta.to(self.device)
        return ((batch_input - beta) / gamma) * torch.sqrt(self.xih + 1e-8) + self.phih



