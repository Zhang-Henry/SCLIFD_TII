# -*- coding: utf-8 -*-
"""
Created on Sat May  7 16:31:13 2022

@author: zhr
"""


"""
    This class implements the main model of iCaRL
    and all the methods regarding the exemplars
    from delivery: iCaRL is made up of 2 components
    - feature extractor (a convolutional NN) => resnet32 optimized on cifar100
    - classifier => a FC layer OR a non-parametric classifier (NME)
"""
from tkinter.messagebox import NO
from torchvision import transforms as T
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Subset, DataLoader, TensorDataset
from torch.backends import cudnn
import copy
import gc #extensive use in order to manage memory issues

import utils
import math
import random
import pandas as pd
# new classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from tqdm import tqdm
from imblearn.ensemble import BalancedRandomForestClassifier

class CustomSubset(Subset):
     '''A custom subset class'''
     def __init__(self, dataset, indices):
         super().__init__(dataset, indices)
         self.X = dataset.X # 保留targets属性
         self.Y = dataset.Y # 保留classes属性

     def __getitem__(self, idx): #同时支持索引访问操作
         x, y = self.dataset[self.indices[idx]]
         return x, y

     def __len__(self): # 同时支持取长度操作
         return len(self.indices)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count


        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),0)

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        logits = logits * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        #
        #
        # od over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
#        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def _ICaRL__item_reorder(item_seq):
#    print(item_seq.shape)
    item_seq_len =  item_seq.shape[0]
#    print(item_seq_len)
    beta = 0.6
    num_reorder = math.floor(item_seq_len * beta)
    reorder_begin = random.randint(0, item_seq_len - num_reorder)
    reordered_item_seq = item_seq.clone()
    shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
    random.shuffle(shuffle_index)
    reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
    return reordered_item_seq

def _ICaRL__Gaussian_Noise(img):
    """
    :param img: 输入的图像
    :param pos: 图像截取的位置,类型为元组，包含(x, y)
    :param size: 图像截取的大小
    :return: 返回截取后的图像
    """
    m = img.shape
    noise = torch.randn(m)
    return img + 0.001*noise


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [x, self.transform(x)]



class SCLDataset(Dataset):
    def __init__(self, x, y, indices, transform=None):
        self.X = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self.indices[idx]  # 使用提供的索引
        x = self.X[idx]
        label = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, label


def auto_loss_rebalancing(n_known, n_classes, loss_type):
  alpha = n_known/n_classes

  if loss_type == 'class':
    return 1-alpha
  return alpha

def get_rebalancing(rebalancing=None):
  if rebalancing is None:
    return lambda n_known, n_classes, loss_type: 1
  if rebalancing in ['auto', 'AUTO']:
    return auto_loss_rebalancing
  if callable(rebalancing):
    return rebalancing

# feature_size: 2048, why?
# n_classes: 10 => 100
class ICaRL(nn.Module):
  def __init__(self, args, reverse_index = None, class_loss_criterion='bce', dist_loss_criterion='bce', loss_rebalancing='auto', lambda0=1):
    super(ICaRL, self).__init__()
    if args.classify_net=='TE':
        # from nets.ResNet import resnet18
        from nets.Classify_Net import mynet
    elif args.classify_net=='MFF':
        from nets.Classify_Net_CMFF import mynet
    self.net = mynet()
    self.net.fc = nn.Linear(self.net.fc.in_features, args.n_classes)
    self.feature_extractor = mynet()
    self.feature_extractor.fc = nn.Sequential()
    self.n_classes = args.n_classes
    self.n_known = 0
    # Hyper-parameters from iCaRL
    self.BATCH_SIZE = args.BATCH_SIZE
    self.WEIGHT_DECAY  = args.WEIGHT_DECAY
    self.LR = args.LR
    self.GAMMA = args.GAMMA # this allow LR to become 1/5 LR after MILESTONES epochs
    self.NUM_EPOCHS = args.NUM_EPOCHS
    self.DEVICE = args.DEVICE
    self.MILESTONES = args.MILESTONES # when the LR decreases, according to icarl
    self.MOMENTUM = args.MOMENTUM
    self.K = args.K
    self.reverse_index=reverse_index
    self.optimizer, self.scheduler = utils.getOptimizerScheduler(self.LR, self.MOMENTUM, self.WEIGHT_DECAY, self.MILESTONES, self.GAMMA, self.parameters())
    gc.collect()

    # List containing exemplar_sets
    # Each exemplar_set is a np.array of N images
    self.exemplar_sets = []
    self.exemplar_sets_indices = []
    self.all_features=None
    self.labels=[]
    # for the classification/distillation loss we have two alternatives
    # 1- BCE loss with Logits (reduction could be mean or sum)
    # 2- BCE loss + sigmoid
    # actually we use just one loss as explained on the forum

    self.class_loss, self.dist_loss = self.build_loss(class_loss_criterion, dist_loss_criterion, loss_rebalancing, lambda0=lambda0)

    # Means of exemplars (cntroids)
    self.compute_means = True
    self.exemplar_means = []
    self.exemplar_mean_nn = [] # means not normalized

    self.herding = args.herding # random choice of exemplars or icarl exemplars strategy?

    # this is used as explained in the forum to compute the exemplar mean in a more accurate way
    # populated during construct exemplar set and used in the classify step
    self.data_from_classes = []
    self.means_from_classes = []

    # Knn, svc classification
    self.model = None

  # increment the number of classes considered by the net
  # incremental learning approach, 0,10..100
  def increment_classes(self, n):
        gc.collect()

        in_features = self.net.fc.in_features
        out_features = self.net.fc.out_features
        weights = self.net.fc.weight.data
        bias = self.net.fc.bias.data

        self.net.fc = nn.Linear(in_features, out_features + n) #add 10 classes to the fc last layer
        self.net.fc.weight.data[:out_features] = weights
        self.net.fc.bias.data[:out_features] = bias
        self.n_classes += n #icrement #classes considered

  # computes the mean of each exemplar set
  def computeMeans(self):
    torch.no_grad()
    torch.cuda.empty_cache()

    feature_extractor = self.feature_extractor.to(self.DEVICE)
    feature_extractor.train(False)

    # new mean mgmt
    tensors_mean = []
    exemplar_mean_nn=[]
    with torch.no_grad():
      for tensor_set in self.data_from_classes:
        features = []
        for tensor, _ in tensor_set:

          tensor = tensor.to(self.DEVICE)
          feature = feature_extractor(tensor)

          feature.data = feature.data / feature.data.norm() # Normalize
          features.append(feature)

          # cleaning
          torch.no_grad()
          torch.cuda.empty_cache()

        features = torch.stack(features) #(num_exemplars,num_features)
        mean_tensor = features.mean(0)
        exemplar_mean_nn.append(mean_tensor.to('cpu'))
        mean_tensor.data = mean_tensor.data / mean_tensor.data.norm() # Re-normalize
        mean_tensor = mean_tensor.to('cpu')
        tensors_mean.append(mean_tensor)

    self.exemplar_means = tensors_mean  # nb the mean is computed over all the imgs
    self.exemplar_mean_nn= exemplar_mean_nn # exemplars means not normalized

    # cleaning
    torch.no_grad()
    torch.cuda.empty_cache()

  # train procedure common for KNN and SVC classifier (save a lot of training time)
  def modelTrain(self, method, K_nn = None):
    torch.no_grad()
    torch.cuda.empty_cache()

    feature_extractor = self.feature_extractor.to(self.DEVICE)
    feature_extractor.train(False)

    # -- train a SVC classifier
    X_train, y_train = [], []

    for exemplar_set in self.exemplar_sets:
          for exemplar, label in  exemplar_set:
            exemplar = exemplar.to(self.DEVICE)
            feature = feature_extractor(exemplar)
            feature = feature.squeeze()
            feature.data = feature.data / feature.data.norm() # Normalize
            X_train.append(feature.cpu().detach().numpy())
            y_train.append(label)

    if method == 'KNN':
      model = KNeighborsClassifier(n_neighbors = K_nn)
    elif method == 'SVC':
      model = LinearSVC()
    elif method == 'RF':
      model = BalancedRandomForestClassifier(n_estimators=200, random_state=42, sampling_strategy='auto',replacement=True, bootstrap=True)


    self.model = model.fit(X_train, y_train)


  def modelTrainFC(self):
    # torch.no_grad()
    # torch.cuda.empty_cache()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(self.net.fc.parameters(), lr=0.001)
    # for param in self.net.encoder.parameters():
    #     param.requires_grad = False

    self.net = self.net.to(self.DEVICE)
    feature_extractor = self.feature_extractor.to(self.DEVICE)
    feature_extractor.eval()

    # -- train a SVC classifier
    X_train, y_train = [], []

    with torch.no_grad():
      for exemplar_set in self.exemplar_sets:
            for exemplar, label in  exemplar_set:
              exemplar = exemplar.to(self.DEVICE)
              feature = feature_extractor(exemplar)
              feature = feature.squeeze()
              feature.data = feature.data / feature.data.norm() # Normalize
              X_train.append(feature)
              y_train.append(label)
      # Assuming features and labels are collected
    X_train_tensor = torch.stack(X_train)  # Convert list of tensors to a single tensor
    y_train_tensor = torch.tensor(y_train)  # Convert list to a tensor
    # Create a TensorDataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    for epoch in tqdm(range(500)):
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = self.net.fc(inputs)  # Use the FC layer directly
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()

  # common classify function
  def KNN_SVC_classify(self, images):
    torch.no_grad()
    torch.cuda.empty_cache()

    # --- prediction
    X_pred = []
    images = images.to(self.DEVICE)
    feature_extractor = self.feature_extractor.to(self.DEVICE)
    feature_extractor.train(False)

    features = feature_extractor(images)
    for feature in features:
      feature = feature.squeeze()
      feature.data = feature.data / feature.data.norm() # Normalize
      X_pred.append(feature.cpu().detach().numpy())

    preds = self.model.predict(X_pred)
    # --- end prediction
    return torch.tensor(preds)


  #   return torch.FloatTensor(preds).to(self.DEVICE)
  def COS_classify(self, batch_imgs):
    torch.no_grad()
    torch.cuda.empty_cache()
    batch_imgs_size = batch_imgs.size(0)
    feature_extractor = self.feature_extractor.to(self.DEVICE)
    feature_extractor.train(False)

    means_exemplars = torch.cat(self.exemplar_mean_nn, dim=0)
    means_exemplars = torch.stack([means_exemplars] * batch_imgs_size)
    means_exemplars = means_exemplars.transpose(1, 2) # means no normalized

    feature = feature_extractor(batch_imgs) # features no normalized

    feature=feature.to('cpu')
    means_exemplars = means_exemplars.to('cpu')

    preds=[]
    for a in feature:
      a=a.detach().numpy()
      aa=np.linalg.norm(a)
      res=[]
      for b in means_exemplars:
        b=b.detach().numpy()
        bb=np.linalg.norm(b)
        dot = np.dot(a, b)
        cos = dot / (aa * bb)
        res.append(cos)
      preds.append(np.argmax(np.array(res)))

    # cleaning
    torch.no_grad()
    torch.cuda.empty_cache()
    gc.collect()

    return torch.FloatTensor(preds).to(self.DEVICE)

  # classification via fc layer (similar to lwf approach)
  # def FCC_classify(self, images):
  #   self.net = self.net.cuda()
  #   _, preds = torch.max(torch.softmax(self.net(images), dim=1), dim=1, keepdim=False)
  #   return preds
  def FCC_classify(self, images):
    self.net = self.net.cuda()
    _, preds = torch.max(self.net(images), dim=1, keepdim=False)
    return preds


  def classify(self, batch_imgs):
      """Classify images by nearest-mean-of-exemplars
      Args:
          batch_imgs: input image batch
      Returns:
          preds: Tensor of size (batch_size,)
      """
      torch.no_grad()
      torch.cuda.empty_cache()

      batch_imgs_size = batch_imgs.size(0)
      feature_extractor = self.feature_extractor.to(self.DEVICE)
      feature_extractor.train(False)

      # update exemplar_means with the mean
      # of all the train data for a given class

      means_exemplars = torch.cat(self.exemplar_means, dim=0)
      means_exemplars = torch.stack([means_exemplars] * batch_imgs_size)
      means_exemplars = means_exemplars.transpose(1, 2)

      feature = feature_extractor(batch_imgs)
      aus_normalized_features = []
      for el in feature: # Normalize
          el.data = el.data / el.data.norm()
          aus_normalized_features.append(el)

      feature = torch.stack(aus_normalized_features,dim=0)

      feature = feature.unsqueeze(2)
      feature = feature.expand_as(means_exemplars)

      means_exemplars = means_exemplars.to(self.DEVICE)

      # Nearest prototype
      preds = torch.argmin((feature - means_exemplars).pow(2).sum(1),dim=1)


      # WD=SinkhornDistance(eps=0.1, max_iter=100)
      # wd,_,_=WD(feature, means_exemplars)
      # preds = torch.argmin(wd)
      # cleaning
      torch.no_grad()
      torch.cuda.empty_cache()
      gc.collect()

      return preds

  # implementation of alg. 4 of icarl paper
  # iCaRL ConstructExemplarSet
  def construct_exemplar_set(self, tensors, m, label):
    """
      Args:
          tensors: train_subset containing a single label
          m: number of exemplars allowed/exemplar set (class)
          label: considered class
    """
    torch.no_grad()
    torch.cuda.empty_cache()
    gc.collect()
    exemplar_set_indices = set()
    exemplar_list_indices = []
    exemplar_set = []
    if self.herding=="icarl":
      print('icarl_herding')
      feature_extractor = self.feature_extractor.to(self.DEVICE)
      feature_extractor.train(False)
      # Compute and cache features for each example
      features = []
      labels_all = []
      loader = DataLoader(tensors,batch_size=self.BATCH_SIZE,shuffle=True,drop_last=False)
      with torch.no_grad():
        for _, images, labels in loader:
          images = images.to(self.DEVICE)
          labels = labels.to(self.DEVICE)
          feature = feature_extractor(images)
          feature = feature / np.linalg.norm(feature.cpu()) # Normalize
          features.append(feature)
          labels_all.append(labels)
      features_s = torch.cat(features)
      labels_all = torch.cat(labels_all)
      class_mean = features_s.mean(0)
      class_mean = class_mean / np.linalg.norm(class_mean.cpu()) # Normalize
      class_mean = torch.stack([class_mean]*features_s.size()[0])
      summon = torch.zeros(1,features_s.size()[1]).to(self.DEVICE) #(1,num_features)
      for k in range(1, (m + 1)):
          S = torch.cat([summon]*features_s.size()[0]) # second addend, features in the exemplar set
          results = pd.DataFrame(list((class_mean-(1/k)*(features_s + S)).pow(2).sum(1).detach().cpu()), columns=['result']).sort_values('result')
          results['index'] = results.index
          results = results.to_numpy()
          # select argmin not included in exemplar_set_indices
          for i in range(results.shape[0]):
            index = results[i, 1]
            exemplar_k_index = tensors[index][0]
            if exemplar_k_index not in exemplar_set_indices:
              exemplar_k = tensors[index][1].unsqueeze(dim = 0) # take the image from the tuple (index, img, label)
              exemplar_set.append((exemplar_k, label))
              exemplar_k_index = tensors[index][0] # index of the img on the real dataset
              exemplar_list_indices.append(exemplar_k_index)
              exemplar_set_indices.add(exemplar_k_index)
              break
          # features of the exemplar k
          phi = feature_extractor(exemplar_k.to(self.DEVICE)) #feature_extractor(exemplar_k.to(self.DEVICE))
          summon += phi # update sum of features
    elif self.herding=="MES":
      print('MES')
      feature_extractor = self.feature_extractor.to(self.DEVICE)
      feature_extractor.train(False)
      # Compute and cache features for each example
      features = []
      labels_all = []
      loader = DataLoader(tensors,batch_size=self.BATCH_SIZE,shuffle=True,drop_last=False)
      with torch.no_grad():
        for _, images, labels in loader:
          images = images.to(self.DEVICE)
          labels = labels.to(self.DEVICE)
          feature = feature_extractor(images)
          feature = feature / np.linalg.norm(feature.cpu()) # Normalize
          features.append(feature)
          labels_all.append(labels)
      features_s = torch.cat(features)
      labels_all = torch.cat(labels_all)
      class_mean = features_s.mean(0)
      class_mean = class_mean / np.linalg.norm(class_mean.cpu()) # Normalize
      class_mean = torch.stack([class_mean]*features_s.size()[0])
      summon = torch.zeros(1,features_s.size()[1]).to(self.DEVICE) #(1,num_features)
      for k in range(1, (m + 1)):
          S = torch.cat([summon]*features_s.size()[0]) # second addend, features in the exemplar set
          results = pd.DataFrame(list((class_mean-(1/k)*(features_s + S)).pow(2).sum(1).detach().cpu()), columns=['result']).sort_values('result', ascending=False)
          results['index'] = results.index
          results = results.to_numpy()
          # select argmax not included in exemplar_set_indices
          for i in range(results.shape[0]):
            index = results[i, 1]
            exemplar_k_index = tensors[index][0]
            if exemplar_k_index not in exemplar_set_indices:
              exemplar_k = tensors[index][1].unsqueeze(dim = 0) # take the image from the tuple (index, img, label)
              exemplar_set.append((exemplar_k, label))
              exemplar_k_index = tensors[index][0] # index of the img on the real dataset
              exemplar_list_indices.append(exemplar_k_index)
              exemplar_set_indices.add(exemplar_k_index)
              break
          # features of the exemplar k
          phi = feature_extractor(exemplar_k.to(self.DEVICE)) #feature_extractor(exemplar_k.to(self.DEVICE))
          summon += phi # update sum of features
    elif self.herding=="mixed":
          # 初始化步骤，特征提取器和缓存特征相同
          feature_extractor = self.feature_extractor.to(self.DEVICE)
          feature_extractor.train(False)

          features = []
          labels_all = []
          loader = DataLoader(tensors, batch_size=self.BATCH_SIZE, shuffle=True, drop_last=False)

          with torch.no_grad():
              for _, images, labels in loader:
                  images = images.to(self.DEVICE)
                  labels = labels.to(self.DEVICE)
                  feature = feature_extractor(images)
                  feature = feature / np.linalg.norm(feature.cpu())  # Normalize
                  features.append(feature)
                  labels_all.append(labels)

          features_s = torch.cat(features)
          labels_all = torch.cat(labels_all)

          class_mean = features_s.mean(0)
          class_mean = class_mean / np.linalg.norm(class_mean.cpu())  # Normalize
          class_mean = torch.stack([class_mean] * features_s.size()[0])

          summon = torch.zeros(1, features_s.size()[1]).to(self.DEVICE)  # (1, num_features)

          # 将m分成两部分，一半用icarl，一半用MES
          half_m = m // 2
          for k in range(1, m + 1):
              S = torch.cat([summon] * features_s.size()[0])  # second addend, features in the exemplar set

              if k <= half_m:  # 使用 icarl 策略 (选择最小距离)
                  results = pd.DataFrame(list((class_mean - (1 / k) * (features_s + S)).pow(2).sum(1).detach().cpu()), columns=['result']).sort_values('result')
              else:  # 使用 MES 策略 (选择最大距离)
                  results = pd.DataFrame(list((class_mean - (1 / k) * (features_s + S)).pow(2).sum(1).detach().cpu()), columns=['result']).sort_values('result', ascending=False)

              results['index'] = results.index
              results = results.to_numpy()

              # 选择未包含在 exemplar_set_indices 中的样本
              for i in range(results.shape[0]):
                  index = int(results[i, 1])
                  exemplar_k_index = tensors[index][0]
                  if exemplar_k_index not in exemplar_set_indices:
                      exemplar_k = tensors[index][1].unsqueeze(dim=0)
                      exemplar_set.append((exemplar_k, label))
                      exemplar_list_indices.append(exemplar_k_index)
                      exemplar_set_indices.add(exemplar_k_index)
                      break

              # 更新召唤特征向量
              phi = feature_extractor(exemplar_k.to(self.DEVICE))
              summon += phi  # 累加选中样本的特征

    elif self.herding=="random":
      print('random herding')
      tensors_size = len(tensors)
      unique_random_indexes = random.sample(range(0, tensors_size), min(m,tensors_size)) # random sample without replacement k exemplars
      i = 0
      for k in range(1, (min(m,tensors_size) + 1)):
        index = unique_random_indexes[i]
        exemplar_k = tensors[index][1].unsqueeze(dim = 0)
        exemplar_k_index = tensors[index][0]
        exemplar_set.append((exemplar_k, label))
        exemplar_set_indices.add(exemplar_k_index)
        exemplar_list_indices.append(exemplar_k_index)
        i = i + 1
    # --- new ---
    tensor_set = []
    for i in range(0, len(tensors)):
      t = tensors[i][1].unsqueeze(dim = 0)
      tensor_set.append((t, label))

    self.exemplar_sets.append(exemplar_set) #update exemplar sets with the updated exemplars images
    self.exemplar_sets_indices.append(exemplar_list_indices)
    # print(exemplar_list_indices)
    # this is used to compute more accurately the means of the exemplar (see also computeMeans and classify)
    self.data_from_classes.append(tensor_set)

    # cleaning
    torch.cuda.empty_cache()

  # build a exemplar dataset as a subset of the train dataset
  def build_exemplars_dataset(self, train_dataset): #complete train dataset
    all_exemplars_indices = []
    for exemplar_set_indices in self.exemplar_sets_indices:
        all_exemplars_indices.extend(exemplar_set_indices)
    # print('all_exemplars_indices:',all_exemplars_indices)
    exemplars_dataset = CustomSubset(train_dataset, all_exemplars_indices)
    return exemplars_dataset

  def update_representation(self, dataset, train_dataset_big, new_classes):
    # 1 - retrieve the classes from the dataset (which is the current train_subset)
    # 2 - retrieve the new classes
    # 1,2 are done in the main_icarl
    #gc.collect()

    # 3 - increment classes
    #          (add output nodes)
    #          (update n_classes)
    # 5        store network outputs with pre-update parameters
    self.increment_classes(len(new_classes))

    # 4 - combine current train_subset (dataset) with exemplars
    #     to form a new augmented train dataset
    # join the datasets
    exemplars_dataset = self.build_exemplars_dataset(train_dataset_big)
    #
    if len(exemplars_dataset) > 0:
      augmented_dataset = ConcatDataset(dataset, exemplars_dataset)
      #augmented_dataset = utils.joinSubsets(train_dataset_big, [dataset, exemplars_dataset])
    else:
      augmented_dataset = dataset # first iteration

    # 6 - run network training, with loss function

    net = self.net
    optimizer = optim.Adam(net.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY,)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.MILESTONES, gamma=self.GAMMA)
    criterion =  SupConLoss()

    cudnn.benchmark # Calling this optimizes runtime
    net = net.to(self.DEVICE)
    train_transform = T.Lambda(lambda img:_ICaRL__item_reorder(img))
    data_train_SCL =  SCLDataset(augmented_dataset.X, augmented_dataset.Y, augmented_dataset.indices, transform=TwoCropTransform(train_transform))

    # define the loader for the augmented_dataset
#    loader = DataLoader(data_train_SCL, batch_size=self.BATCH_SIZE,sampler=BalancedBatchSampler(data_train_SCL.X,data_train_SCL.y),drop_last = True)
    loader = DataLoader(data_train_SCL, batch_size=self.BATCH_SIZE,shuffle=True)
    if len(self.exemplar_sets) > 0:
      old_net = copy.deepcopy(net)
      old_net.eval()
    bar=tqdm(range(1, self.NUM_EPOCHS+1))
    total_loss=[]
    for epoch in bar:
        for images, labels in loader:
            # Bring data over the device of choice
            net.train()

            images = torch.cat([images[0], images[1]], dim=0)
            images=images.to(self.DEVICE)
            labels=labels.to(self.DEVICE)
            bsz = labels.shape[0]
            features = net.encoder(images)
#            print(features.shape)
            features = F.normalize(features, dim=1)
            # features = self.attention(features)


            if len(self.exemplar_sets) > 0:
                new_classes_tensor = torch.tensor(new_classes).to(self.DEVICE)

                # 1. 找到不属于新类的标签的索引
                past_classes_idx = ~labels.unsqueeze(1).eq(new_classes_tensor).any(1)

                past_classes_idx_all = torch.concat((past_classes_idx,past_classes_idx))

                features1_prev_task = features[past_classes_idx_all]

                features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T),1)
                logits_mask = torch.scatter(torch.ones_like(features1_sim),1,torch.arange(features1_sim.size(0)).view(-1, 1).to(self.DEVICE),0)
                logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
                features1_sim = features1_sim - logits_max1.detach()
                row_size = features1_sim.size(0)
                logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) #torch.Size([64, 2, 10])
            loss = criterion(features,labels) #监督对比学习分类损失


            if len(self.exemplar_sets) > 0:
                with torch.no_grad():
                    features2_prev_task = old_net.encoder(images)[past_classes_idx_all]

                    features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), 1)

                    logits_max2, _ = torch.max(features2_sim*logits_mask, dim=1, keepdim=True)
                    features2_sim = features2_sim - logits_max2.detach()
                    logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) /  torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

                loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
                loss = loss + loss_distill

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        scheduler.step()

    self.net = copy.deepcopy(net)
    self.net.fc =  nn.Linear(self.net.fc.in_features,  self.n_classes)
    self.feature_extractor = copy.deepcopy(net)
    self.feature_extractor.fc = nn.Sequential()

    #cleaning
    del net
    torch.cuda.empty_cache()


  def extract_features(self, dataloader):
    self.eval()
    features = []
    labels = []
    for _, images, labels_batch in dataloader:
      images = images.to(self.DEVICE)
      with torch.no_grad():
        feature = self.feature_extractor(images).cpu()
      features.append(feature)
      labels.extend(labels_batch)
    features = torch.cat(features, dim=0)
    return features, torch.tensor(labels)

  def build_loss(self, class_loss_criterion, dist_loss_criterion, rebalancing=None, lambda0=1):
    class_loss_func = None
    dist_loss_func = None

    if class_loss_criterion in ['l2', 'L2']:
      class_loss_func = self.l2_class_loss
    elif class_loss_criterion in ['bce', 'BCE']:
      class_loss_func = self.bce_class_loss
    elif class_loss_criterion in ['ce', 'CE']:
      class_loss_func = self.ce_class_loss

    if dist_loss_criterion in ['l2', 'L2']:
      dist_loss_func = self.l2_dist_loss
    elif dist_loss_criterion in ['bce', 'BCE']:
      dist_loss_func = self.bce_dist_loss
    elif dist_loss_criterion in ['ce', 'CE']:
      dist_loss_func = self.ce_dist_loss

    rebalancing = get_rebalancing(rebalancing)

    def class_loss(outputs, labels, row_start=None, row_end=None, col_start=None, col_end=None):
      alpha = rebalancing(self.n_known, self.n_classes, 'class')
      return alpha*class_loss_func(outputs, labels, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end)

    def dist_loss(outputs, labels, row_start=None, row_end=None, col_start=None, col_end=None):
      alpha = rebalancing(self.n_known, self.n_classes, 'dist')
      return lambda0*alpha*dist_loss_func(outputs, labels, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end)

    return class_loss, dist_loss

  def bce_class_loss(self, outputs, labels, row_start=None, row_end=None, col_start=None, col_end=None):
    return self.bce_loss(outputs, labels, encode=True, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end)

  def bce_dist_loss(self, outputs, labels, row_start=None, row_end=None, col_start=None, col_end=None):
    return self.bce_loss(outputs, labels, encode=False, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end)

  def ce_class_loss(self, outputs, labels, row_start=None, row_end=None, col_start=None, col_end=None):
    return self.ce_loss(outputs, self.reverse_index.getNodes(labels), decode=False, row_start=row_start, row_end=row_end, col_start=None, col_end=col_end)

  def ce_dist_loss(self, outputs, labels, row_start=None, row_end=None, col_start=None, col_end=None):
    return self.ce_loss(outputs, labels, decode=True, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end)

  def l2_class_loss(self, outputs, labels, row_start=None, row_end=None, col_start=None, col_end=None):
    return self.l2_loss(outputs, labels, encode=True, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end)

  def l2_dist_loss(self, outputs, labels, row_start=None, row_end=None, col_start=None, col_end=None):
    return self.l2_loss(outputs, labels, encode=False, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end)


  def bce_loss(self, outputs, labels, encode=False, row_start=None, row_end=None, col_start=None, col_end=None):
    criterion = nn.BCEWithLogitsLoss(reduction = 'mean')

    if encode:
      labels = utils._one_hot_encode(labels, self.n_classes, self.reverse_index, device=self.DEVICE)
      labels = labels.type_as(outputs)

    return criterion(outputs[row_start:row_end, col_start:col_end], labels[row_start:row_end, col_start:col_end])


  def ce_loss(self, outputs, labels, decode=False, row_start=None, row_end=None, col_start=None, col_end=None):
    criterion = nn.CrossEntropyLoss()

    if decode:
      labels = torch.argmax(labels, dim=1)

    return criterion(outputs[row_start:row_end, col_start:col_end], labels[row_start:row_end])


  def l2_loss(self, outputs, labels, encode=False, row_start=None, row_end=None, col_start=None, col_end=None):
    criterion = nn.MSELoss(reduction = 'mean')

    if encode:
      labels = utils._one_hot_encode(labels, self.n_classes, self.reverse_index, device=self.DEVICE)
      labels = labels.type_as(outputs)

    loss_val = criterion(outputs[row_start:row_end, col_start:col_end], labels[row_start:row_end, col_start:col_end])
    return self.limit_loss(loss_val)

  def limit_loss(self, loss, limit=3):
    if loss <= limit:
      return loss
    denom = loss.item() / limit
    return loss / denom


  # implementation of alg. 5 of icarl paper
  # iCaRL ReduceExemplarSet
  def reduce_exemplar_sets(self, m):
  	    # i keep only the first m exemplar images
        # where m is the UPDATED K/number_classes_seen
        # the number of images per each exemplar set (class) progressively decreases
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]
        for x, P_x in enumerate(self.exemplar_sets_indices):
            self.exemplar_sets_indices[x] = P_x[:m]


# ----------
from torch.utils.data import Dataset
"""
  Merge two different datasets (train and exemplars in our case)
  format:
  train
  --------
  exemplars
  train leans on cifar100
  exemplars is managed here (exemplar_transform is performed) => changed
"""
class ConcatDataset( ):

    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.l1 = len(dataset1)
        self.l2 = len(dataset2)
        self.X = np.concatenate((dataset1.X,dataset2.X),axis=0) # 保留targets属性
        self.Y = np.concatenate((dataset1.Y,dataset2.Y),axis=0) # 保留classes属性
        self.indices = np.concatenate((dataset1.indices,dataset2.indices),axis=0) # 保留classes属性

    def __getitem__(self,index):
        if index < self.l1:
            image,label = self.dataset1[index] #here it leans on cifar100 get item
            return image,label
        else:
             image, label = self.dataset2[index - self.l1]
             return  image,label

    def __len__(self):
        return (self.l1 + self.l2)
#------------