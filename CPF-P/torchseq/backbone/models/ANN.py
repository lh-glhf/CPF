import torch
import torch.nn as nn
import os


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_size = args.seq_len * args.enc_in
        self.args = args
        self.device = self._acquire_device()
        output_size = args.pred_len * args.enc_in
        hidden_size = args.hidden_size
        super().__init__()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

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

    def forward(self, x, x_mark, dec_inp, y_mark):
        B, L, D = x.shape
        x = x.view(B, L * D)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x.view(B, -1, D)



