import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta


class Dataset_text_ruptures(Dataset):
    def __init__(self, root_path, change_path, flag='train', delay=7, size=20):
        self.root_path = root_path
        self.change_path = change_path
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.delay = delay
        self.size = size
        self.__read_data__()

    def __read_data__(self):
        self.changePoints = pd.read_csv(self.change_path)['0'].values
        self.changePoints = np.array([datetime.strptime(point, '%Y/%m/%d %H:%M:%S') for point in self.changePoints])
        texts = os.listdir(self.root_path)
        data_days = []
        for text_path in texts:
            data_path = os.path.join(self.root_path, text_path)
            df_data = pd.read_csv(data_path)
            titles = df_data['title'].values
            date = datetime.strptime(df_data['date'].values[0], '%Y/%m/%d %H:%M:%S')
            date_range = [date + timedelta(days=i) for i in range(self.delay)]
            result = np.array([1 if day in self.changePoints else 0 for day in date_range])
            titles = titles[:self.size * (len(titles) // self.size)]
            titles = titles.reshape(-1, self.size)
            for title in titles:
                data_days.append((title, result, date))
            num_train = int(len(data_days) * 0.7)
            num_test = int(len(data_days) * 0.2)
            num_vali = len(data_days) - num_train - num_test
            border1s = [0, num_train, len(data_days) - num_test]
            border2s = [num_train, num_train + num_vali, len(data_days)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            self.data = data_days[border1: border2]

    def __getitem__(self, index):
        text_x, result_y, date = self.data[index]
        return text_x, result_y, date

    def __len__(self):
        return len(self.data)



