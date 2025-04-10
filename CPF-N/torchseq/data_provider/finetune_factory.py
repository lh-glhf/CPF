import numpy as np
from .finetune_loader import Dataset_text_ruptures
from torch.utils.data import DataLoader
import torch
finetune_dict = {
    'text_ruptures': Dataset_text_ruptures
}


def text_collate_fn(batch):
    texts, change_points, dates = zip(*batch)
    texts = np.array([item.tolist() for item in texts])
    dates = np.array(dates)
    return texts, torch.tensor(np.array(change_points)).float(), dates

def data_provider(args, flag):
    Data = finetune_dict[args.data]
    data_set = Data(
        root_path=args.root_path,
        change_path=args.change_path,
        flag=flag,
        delay=args.delay,
        size=args.size,
    )
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=text_collate_fn,
    )
    return data_set, data_loader

