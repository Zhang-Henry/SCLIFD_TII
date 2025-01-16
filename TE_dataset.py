import torch
from torch.utils.data import Dataset,Subset
import numpy as np
import pandas as pd
import random

class MyDataset(Dataset):

    def __init__(self,  x_dir, y_dir, transform = None):
        self.transform = transform
        data = np.load(x_dir)
        self.X = data
        data = torch.from_numpy(data)
        # data = torch.tensor(data, dtype=torch.float32)
        data = data.clone().detach().to(torch.float32)
        labels = np.load(y_dir)
        self.Y  =  labels
        labels = torch.from_numpy(labels)
        # labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.clone().detach().to(torch.long)
        labels = np.array(labels)

        self.df = pd.DataFrame()
        self.df['data'] = pd.Series(list(data))
        self.df['labels'] = labels

        self.data = self.df['data']
        self.labels = self.df['labels']



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.df.loc[index, 'data'], self.df.loc[index, 'labels']


        if self.transform is not None:
            img = self.transform(img)

        return index, img, target

    def __len__(self):
        return len(self.X)

    def getTargets(self):
        return set(self.labels)

    # test
    def get_indices(self, labels):
        return list(self.df[self.df['labels'].isin(labels)].index)

    def split_classes(self, n_splits=5, seed=None, dictionary_of='dataframes'):
        if dictionary_of not in ['dataframes','indices']:
            raise ValueError("'dictionary_of' must be equal to 'dataframes' or 'indices'")

        all_classes = list(self.df['labels'].value_counts().index)
        all_classes = sorted(all_classes)
        dictionary = {}
        random.seed(seed)
#        random.shuffle(all_classes)
        split_size = int(len(all_classes)/n_splits)
        for j in range(n_splits):
            if ((j+1)*split_size < len(all_classes)):
                split_end = (j+1)*split_size
            else:
                split_end = None
            subgroup = all_classes[j*split_size:split_end]
            if dictionary_of == 'dataframes':
                dictionary[j] = self.df[self.df['labels'].isin(subgroup)]
            elif dictionary_of == 'indices':
                dictionary[j] = list(self.df[self.df['labels'].isin(subgroup)].index)
        return dictionary

    def split_groups_in_train_validation(self, groups, ratio=0.5, seed=None):
        groups_train_val = dict()
        for k, subdf in groups.items():
            train_indexes = []
            val_indexes = []
            split_labels = list(subdf['labels'].value_counts().index)
            for l in split_labels:
                indexes_to_sample = list(subdf[subdf['labels'] == l].index)
                # random.seed(seed)
                train_samples = random.sample(indexes_to_sample, int(len(indexes_to_sample)*ratio))
                train_indexes = train_indexes + train_samples
                val_indexes = val_indexes + list(set(indexes_to_sample).difference(set(train_samples)))
            groups_train_val[k] = {
                'train': train_indexes,
                'val': val_indexes
            }
        return groups_train_val

    def split_in_train_val_groups(self, n_splits=5, ratio=0.5, seed=None):
        groups = self.split_classes(n_splits=n_splits, seed=seed)
        return self.split_groups_in_train_validation(groups, ratio=ratio, seed=seed)

    # given a tensors returns an image (used in exemplars)
    #def tensorToImg(self, tensor):
    #   return Variable(transform(Image.fromarray(img)), volatile=True)


class CustomSubset(Subset):
     '''A custom subset class'''
     def __init__(self, dataset, indices):
         super().__init__(dataset, indices)
         self.X = dataset.X # 保留targets属性
         self.Y = dataset.Y # 保留classes属性

     def __getitem__(self, idx): #同时支持索引访问操作
         _, x, y = self.dataset[self.indices[idx]]
         return _, x, y

     def __len__(self): # 同时支持取长度操作
         return len(self.indices)