B"""
This is a wrapper of methods used commonly across our various models
such as: finetuning, icarl, lwf.
By centralizing them we make the code more efficient and less prone to errors.
"""


from torchvision import transforms
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset, DataLoader
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module
import os, datetime, random

def build_test_splits(dataset, reverse_index):
    splits = dict()
    groups = list(reverse_index.getGroups())
    for g in groups:
        labels_of_groups = reverse_index.getLabelsOfGroup(g)
        indices = list(dataset.df[dataset.df['labels'].isin(labels_of_groups)].index)
        splits[g] = indices
    return splits

# These are the default iCaRL hyper-parameters
def getHyperparams():
	dictHyperparams = {
		"LR": 0.01,
		"MOMENTUM": 0.9,
		"WEIGHT_DECAY": 1e-5,
		"NUM_EPOCHS": 500,
		"MILESTONES": [200, 400],
		"BATCH_SIZE": 512,
		"GAMMA": 0.2,
		"SEED": 66, #use 30, 42, 16
		"LOG_FREQUENCY": 100,
		"NUM_CLASSES": 12
	}
	return dictHyperparams

def getOptimizerScheduler(LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, parameters_to_optimize):
#	optimizer = optim.SGD(parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer = optim.Adam(parameters_to_optimize, lr=LR)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA, last_epoch=-1)
    return optimizer, scheduler

# the mean and the std have been found on the web as mean and std of cifar100
# alternative (realistic): compute mean and std for the dataset
def getTransformations():
	# Define transforms for training phase
	train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), # Randomly flip the image with probability of 0.5
	                                      transforms.Pad(4), # Add padding
	                                      transforms.RandomCrop(32),# Crops a random squares of the image
	                                      transforms.ToTensor(), # Turn PIL Image to torch.Tensor
	                                      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
	])
	# Define transforms for the evaluation phase
	eval_transform = transforms.Compose([
	                                      transforms.ToTensor(),
	                                      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
	])
	return train_transform, eval_transform

# BCEWithLogits = Sigmoid + BCE, is the loss used in iCaRL
def getLossCriterion():
	criterion = nn.BCEWithLogitsLoss(reduction = 'mean') # for classification: Cross Entropy
	return criterion

def computeLoss(criterion, outputs, labels):
	return criterion(outputs, labels)

# support BCE
def _one_hot_encode(labels, n_classes, reverse_index, dtype=None, device="cpu"):
		batch_size = len(labels)
		enconded = torch.zeros(batch_size, n_classes, dtype=dtype, device=device)
		labels=map_to_outputs(labels, reverse_index)
		for i, l in enumerate(labels):
			enconded[i, l] = 1
		return enconded

def map_to_outputs(labels, reverse_index):
		if reverse_index is None:
			return labels
		if type(labels) == int:
			return int(reverse_index.getNodes(torch.tensor([labels])))
		elif type(labels) == torch.Tensor:
			return reverse_index.getNodes(labels)


def plotAccuracyTrend(method, data_plot_line, seed, out_path):
#	plt.figure(figsize=(20,7))
	accuracyDF=pd.DataFrame(data_plot_line, columns = ['Classes','Accuracy'])
	ax = sns.lineplot(x="Classes", y="Accuracy",data=accuracyDF, marker = 'o')
	ax.minorticks_on()
	ax.set_xticks(np.arange(2,12,2))
	ax.set_xlim(xmin=1, xmax=11)
	ax.set_ylim(ymin=0, ymax=1)
	plt.legend(['Accuracy {}'.format(method)])
	ax.grid(axis='y')
	plt.title("Accuracies against seen classes {} - seed: {}".format(method, seed))
	filename = "images/acc_{}_{}.jpg".format(method, seed) # ex. acc_lwf_30
	plt.savefig(os.path.join(out_path,filename), format='png', dpi=300)
	plt.show()

def plotF1Trend(method, data_plot_line, seed):
#	plt.figure(figsize=(20,7))
	accuracyDF=pd.DataFrame(data_plot_line, columns = ['Classes','F1 score'])
	ax = sns.lineplot(x="Classes", y="F1 score",data=accuracyDF, marker = 'o')
	ax.minorticks_on()
	ax.set_xticks(np.arange(2,12,2))
	ax.set_xlim(xmin=1, xmax=11)
	ax.set_ylim(ymin=0, ymax=1)
	plt.legend(['F1 score {}'.format(method)])
	ax.grid(axis='y')
	plt.title("F1 scores against seen classes {} - seed: {}".format(method, seed))

	filename = "images/f1_{}_{}.jpg".format(method, seed)
	plt.savefig(filename, format='png', dpi=300)
	plt.show()

def plotConfusionMatrix(method, confusionMatrixData, seed, out_path):
#	fig,ax=plt.subplots(figsize=(10,10))
    fig,ax=plt.subplots()
    sns.heatmap(confusionMatrixData,cmap='terrain',ax=ax)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title("Confusion Matrix {} - seed: {}".format(method, seed))
    filename = "images/cm_{}_{}.jpg".format(method, seed) # ex. cm_lwf_30
    plt.savefig(os.path.join(out_path,filename), format='png', dpi=300)
    plt.show()

# Write down the metrices (accuracy trand and confusion matrix)
# this method is a shortcut when perfoming multiple tests with different splits (random_seed)
# and allow us to plot on the same graph the data from multiple models (accuracy)
def writeMetrics(args, accuracies, confusionMatrixData):
  data = {}
  data['avg acc'] = sum(accuracies)/len(accuracies)
  data['K'] = args.K
  data['accuracies'] = []
  data['cm'] = [] #cm line
  i = 0
  for classes_seen in range(2, 12, 2): #x axis on the plot
    data['accuracies'].append({classes_seen : accuracies[i]})
    i += 1

  i = 0
  for class_num in range(0,len(confusionMatrixData)): #rows of the cm
    data['cm'].append({class_num : confusionMatrixData[i].tolist()})
    i += 1

  out = os.path.join(args.out_path,"result.txt")
  with open(out, 'a') as f:
    json.dump(data, f)
    f.write('\n')
  print(f'Saving to {out}')

def joinSubsets(dataset, subsets):
    indices = []
    for s in subsets:
        indices += s.indices
    return Subset(dataset, indices)

def L_G_dist_scalar(feat_old_1d, feat_new_1d):
	feat_old_1d = feat_old_1d/torch.norm(feat_old_1d, 2)
	feat_new_1d = feat_new_1d/torch.norm(feat_new_1d, 2)
	return 1 - feat_old_1d.dot(feat_new_1d)

def L_G_dist(feat_old, feat_new, reduce='mean'):
	result = torch.zeros(feat_old.size()[0], dtype=torch.float64, device=feat_old.device)
	for i in range(feat_old.size()[0]):
		result[i] = L_G_dist_scalar(feat_old[i,:], feat_new[i,:])
	if reduce == 'mean':
		return result.mean()
	elif reduce == 'sum':
		return result.sum()
	return result

def L_G_dist_criterion(reduce='mean'):
	def loss(feat_old, feat_new):
		return L_G_dist(feat_old, feat_new, reduce=reduce)
	return loss

class CosineNormalizationLayer(Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineNormalizationLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma

    def forward(self, input):
        #w_norm = self.weight.data.norm(dim=1, keepdim=True)
        #w_norm = w_norm.expand_as(self.weight).add_(self.epsilon)
        #x_norm = input.data.norm(dim=1, keepdim=True)
        #x_norm = x_norm.expand_as(input).add_(self.epsilon)
        #w = self.weight.div(w_norm)
        #x = input.div(x_norm)
        out = F.linear(F.normalize(input, p=2,dim=1), \
                F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out

        return out



def seed_torch(seed=66):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
