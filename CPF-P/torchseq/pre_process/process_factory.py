from torch import nn
from .BN import BN
from .LN import LN
from .MN import MN
from .RevIN import RevIN
from .NaNFill import NaNFill
from torchseq.post_process.smooth import GaussianSmoothing, MovingAverageSmoothing, ExponentialMovingAverage
from .DishTS import DishTS

process_dict = {
    "BN": BN,
    "LN": LN,
    "MN": MN,
    "RIN": RevIN,
    "NaNFill": NaNFill,
    "GS": GaussianSmoothing,
    "MS": MovingAverageSmoothing,
    "DishTS": DishTS,
    "ES": ExponentialMovingAverage
}


class process_pipeline(nn.Module):
    rin_unit = None
    dts_unit = None
    def __init__(self, args, isPro, modules):
        super(process_pipeline, self).__init__()
        self.args = args
        self.isPro = isPro
        self.modules = modules

    def forward(self, x):
        for module in self.modules:
            process_class = process_dict[module]
            unit = process_class(self.args)
            if module == 'RIN':
                if self.isPro:
                    x = unit(x, 'norm')
                    process_pipeline.rin_unit = unit
                else:
                    if process_pipeline.rin_unit is not None:
                        x = process_pipeline.rin_unit(x, 'denorm')
                    else:
                        raise ValueError("ReVIN is not defined")
            elif module == 'DishTS':
                if self.isPro:
                    x, _ = unit(x, 'forward')
                    process_pipeline.dts_unit = unit
                else:
                    if process_pipeline.dts_unit is not None:
                        x = process_pipeline.dts_unit(x, 'inverse')
                    else:
                        raise ValueError("DishTS is not defined")
            else:
                x = unit(x)
        return x


def process_provider(args):
    pre_processes = args.pre_process  # ["BN"]
    post_processes = args.post_process
    pre_pipeline = process_pipeline(args, isPro=True, modules=pre_processes)
    post_pipeline = process_pipeline(args, isPro=False, modules=post_processes)
    return pre_pipeline, post_pipeline

