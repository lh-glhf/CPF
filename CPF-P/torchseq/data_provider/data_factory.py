# from .data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Solar, Dataset_PEMS, Dataset_M4, PSMSegLoader,\
#     MSLSegLoader, SMAPSegLoader,SWATSegLoader, SMDSegLoader, UEAloader, Dataset_ETT_hour_TimesNet, Dataset_Custom_TimesNet, Dataset_ETT_minute_TimesNet, GLUONTSDataset
from .data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Solar, Dataset_PEMS, Dataset_M4, PSMSegLoader,\
    MSLSegLoader, SMAPSegLoader,SWATSegLoader, SMDSegLoader, Dataset_ETT_hour_TimesNet, Dataset_Custom_TimesNet, Dataset_ETT_minute_TimesNet, GLUONTSDataset
from .uea import collate_fn
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh1_TimesNet': Dataset_ETT_hour_TimesNet,
    'ETTh2': Dataset_ETT_hour,
    'ETTh2_TimesNet': Dataset_ETT_hour_TimesNet,
    'ETTm1': Dataset_ETT_minute,
    'ETTm1_TimesNet': Dataset_ETT_minute_TimesNet,
    'ETTm2': Dataset_ETT_minute,
    'ETTm2_TimesNet': Dataset_ETT_minute_TimesNet,
    'custom': Dataset_Custom,
    'custom_TimesNet': Dataset_Custom_TimesNet,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'm4': Dataset_M4,
    'PSMSeg': PSMSegLoader,
    'MSLSeg': MSLSegLoader,
    'SMAPSeg': SMAPSegLoader,
    'SMDSeg': SMDSegLoader,
    'SWATSeg': SWATSegLoader,
    'GLUONTS': GLUONTSDataset
}


def data_provider(args, flag):
    if args.model == "TimesNet":
        Data = data_dict[args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        shuffle_flag = False if flag == 'test' else True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

    else:
        Data = data_dict[args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = False
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader


def random_subset(dataset, pct, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=generator)
    return Subset(dataset, idx[:int(len(dataset) * pct)].long().numpy())


def data_provider_pretrain(args, config, flag, ddp=False):  # args,
    ddp = False
    Data = data_dict[config['data']]
    timeenc = 0 if config['embed'] != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        if 'anomaly_detection' in config['task_name']:  # working on one gpu
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if 'gluonts' in config['data']:
        # process gluonts dataset:
        data_set = Data(
            dataset_name=config['dataset_name'],
            size=(config['seq_len'], config['label_len'], config['pred_len']),
            path=config['root_path'],
            # Don't set dataset_writer
            features=config["features"],
            flag=flag,
        )
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

        return data_set, data_loader

    timeenc = 0 if config['embed'] != 'timeF' else 1

    if 'anomaly_detection' in config['task_name']:
        drop_last = False
        data_set = Data(
            root_path=config['root_path'],
            win_size=config['seq_len'],
            flag=flag,
        )
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)
        print("ddp mode is set to false for anomaly_detection", ddp, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            num_workers=args.num_workers,
            sampler=DistributedSampler(data_set) if ddp else None,
            drop_last=drop_last)
        return data_set, data_loader
    elif 'classification' in config['task_name']:
        drop_last = False
        data_set = Data(
            root_path=config['root_path'],
            flag=flag,
        )
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            sampler=DistributedSampler(data_set) if ddp else None,
            collate_fn=lambda x: collate_fn(x, max_len=config['seq_len'])
        )
        return data_set, data_loader
    else:
        if config['data'] == 'm4':
            drop_last = False
        data_set = Data(
            root_path=config['root_path'],
            data_path=config['data_path'],
            flag=flag,
            size=[config['seq_len'], config['label_len'], config['pred_len']],
            features=config['features'],
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=config['seasonal_patterns'] if config['data'] == 'm4' else None
        )
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            num_workers=args.num_workers,
            sampler=DistributedSampler(data_set) if ddp else None,
            drop_last=drop_last)
        return data_set, data_loader
