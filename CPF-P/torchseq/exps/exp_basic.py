import os
import torch
import numpy as np
from ..pre_process.process_factory import process_provider


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.pre_pipeline, self.post_pipeline = self._build_process_pipeline()
        self.pre_pipeline.to(self.device)
        self.post_pipeline.to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _build_process_pipeline(self):
        pre_pipeline, post_pipeline = process_provider(self.args)
        return pre_pipeline, post_pipeline
    
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

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
