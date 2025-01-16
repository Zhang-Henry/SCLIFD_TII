import torch
from torch.utils.data import Subset, DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import math
from TE_dataset import MyDataset, CustomSubset
import utils
import pandas as pd
import os, datetime, random
from arguments import parse_arguments
from reverse_index import ReverseIndex
from icarl_scl_model import ICaRL
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.special import rel_entr
import pandas as pd
import time


args = parse_arguments()
print(args)

DEVICE = 'cuda'

train_dataset = MyDataset(args.X_train, args.y_train)
test_dataset = MyDataset(args.X_test, args.y_test)

train_splits = train_dataset.split_in_train_val_groups(ratio=1, seed=args.RANDOM_SEED)
outputs_labels_mapping = ReverseIndex(train_dataset, train_splits, device=args.DEVICE)

# performing the test split (coherent with train/val)
test_splits = utils.build_test_splits(test_dataset, outputs_labels_mapping)


train_subsets = []
val_subsets = []
test_subsets = []

for v in train_splits.values():
    train_subs = CustomSubset(train_dataset, v['train'])
    #val_subs = Subset(train_dataset, v['val'])
    train_subsets.append(train_subs)
    #val_subsets.append(val_subs)

for i in range(0,5): # for each group of classes
    v=test_splits[i]
    test_subs = CustomSubset(test_dataset, v)
    test_subsets.append(test_subs)




def computeAccuracy(method, net, loader, reverse_index, dataset, all_preds_cm, all_labels_cm,args):
  total = 0.0
  correct = 0.0
  preds_all=[]
  labels_all=[]
  for indices, images, labels in loader:
        images = images.to(args.DEVICE)
        labels = labels.to(args.DEVICE)

        if args.classifier == 'NCM':
          labels = reverse_index.getNodes(labels)
          preds = net.classify(images)
        elif args.classifier == 'FCC':
          labels = reverse_index.getNodes(labels)
          preds = net.FCC_classify(images)
        elif args.classifier == 'KNN' or args.classifier == 'SVC' or args.classifier == 'RF':
          if net.n_known>1:
            preds = net.KNN_SVC_classify(images)
            preds = preds.to(args.DEVICE)
          else:
            preds = labels.data
        elif args.classifier == 'COS':
          labels = reverse_index.getNodes(labels)
          preds = net.COS_classify(images)


        preds_all.extend(preds)
        labels_all.extend(labels)
        correct += torch.sum(preds == labels.data).data.item()

        if method == 'test':
          all_preds_cm.extend(preds.tolist())
          all_labels_cm.extend(labels.data.tolist())
  accuracy = correct/len(dataset)


  return accuracy, all_preds_cm, all_labels_cm






def incrementalTraining(icarl, train_subsets, val_subsets, test_subsets, reverse_index):
    all_accuracies = []
    group_id=1
    train_set = None
    test_set = None

    total_training_time = 0.0  # 统计总的训练时间
    total_testing_time = 0.0   # 统计总的测试时间

    #for train_subset, val_subset, test_subset in zip(train_subsets, val_subsets, test_subsets):
    for train_subset, test_subset in zip(train_subsets, test_subsets):
        all_preds_cm = []
        all_labels_cm = []
        print("GROUP: ",group_id)
        if args.all == "true":
          if train_set is None:
              train_set = train_subset
          else:
              train_set = utils.joinSubsets(train_dataset, [train_set, train_subset])
          print("Training by all dataset")
        if test_set is None:
          test_set = test_subset
          train_set_big = train_subset
        else:
          test_set = utils.joinSubsets(test_dataset, [test_set, test_subset])

        train_dataloader = DataLoader(train_subset, batch_size=args.BATCH_SIZE,shuffle=True)
        #val_dataloader = DataLoader(val_subset, batch_size=args.BATCH_SIZE,shuffle=True, num_workers=4)
        test_dataloader = DataLoader(test_set, batch_size=args.BATCH_SIZE,shuffle=True)

        ####### iCaRL implementation(following alg. 2,3,4,5 on icarl paper) ##################

        new_classes_examined = list(train_dataset.df.loc[train_subset.indices, 'labels'].value_counts().index)

        start_time = time.time()
        # 1 - update representation of the net
        #  alg. 3 icarl
        # (here the trainset will be augmented with the exemplars too)
        # (here the classes are incremented too)
        icarl.update_representation(train_subset, train_dataset, new_classes_examined)


        # 2 - update m (number of images per class in the exemplar set corresponding to that class)
        m = int(math.ceil(args.K/icarl.n_classes))


        # 3 - reduce exemplar set for all the previously seen classes
        # alg.5 icarl
        if args.all == "true":
          print("Training by all dataset")
        else:
          print("Training by reduced dataset")
          icarl.reduce_exemplar_sets(m)

        # retrieve the 10 classes in the current subset
        # NB. Here there will be exemplars too! (if i do not want that, use new_classes_examined)
        classes_current_subset = list(train_dataset.df.loc[train_subset.indices, 'labels'].value_counts().index)

        print("Constructing exemplar sets class...")

        # 4 - construct the exemplar set for the new classes
        for y in new_classes_examined: # for each class in the current subset
          # extract all the imgs in the train subset that are linked to this class
          images_current_class = train_subset.dataset.df.loc[train_dataset.df['labels'] == y, 'data']
          imgs_idxs = images_current_class.index # the indexes of all the images in the current classe being considered 0...49k
          class_train_subset = Subset(train_dataset, imgs_idxs)#subset of the train dataset where i have all the imgs of class y

          icarl.construct_exemplar_set(class_train_subset,m,y)


        # update the num classes seen so far
        icarl.n_known = icarl.n_classes #n_classes is incremented in 1: updateRepresentation

        print("Performing classification...")

        # start args.classifier
        icarl.computeMeans()

        # common training on exemplars for KNN and SVC args.classifier
        if icarl.n_known>1:
          if args.classifier == 'KNN':
            K_nn = int(math.ceil(m/2))
            icarl.modelTrain(args.classifier, K_nn)
          elif args.classifier == 'SVC' or args.classifier == 'RF':
            icarl.modelTrain(args.classifier)
        if args.classifier == 'FCC':
          icarl.modelTrainFC()
        #train accuracy
        train_accuracy, _, _ = computeAccuracy('train',icarl, train_dataloader, reverse_index, train_subset,all_preds_cm, all_labels_cm,args)
        print ('Train Accuracy (on current group): %.2f\n' % (100.0 * train_accuracy))

        end_time = time.time()
        training_time = end_time - start_time
        total_training_time += training_time

        print(f'Training time for group {group_id}: {training_time:.2f} seconds')

        start_time = time.time()
        #test
        test_accuracy, all_preds_cm, all_labels_cm = computeAccuracy('test',icarl, test_dataloader, reverse_index, test_set, all_preds_cm, all_labels_cm,args)
        all_accuracies.append(test_accuracy)
        print ('Test Accuracy (all groups seen so far): %.2f\n' % (100.0 * test_accuracy))

        print ("the model knows %d classes:\n " % icarl.n_known)

        end_time = time.time()
        testing_time = end_time - start_time
        total_testing_time += testing_time

        print(f'Testing time for group {group_id}: {testing_time:.2f} seconds')


        group_id+=1

    print(f'Total training time: {total_training_time:.2f} seconds')
    print(f'Total testing time: {total_testing_time:.2f} seconds')

    return all_accuracies, np.array(all_preds_cm), np.array(all_labels_cm)





def main():
    utils.seed_torch(args.RANDOM_SEED)

    icarl = ICaRL(args,outputs_labels_mapping)

    accuracies, all_preds_cm, all_labels_cm = incrementalTraining(icarl, train_subsets, val_subsets, test_subsets, outputs_labels_mapping)
    print([len(i) for i in icarl.exemplar_sets_indices])
    print(accuracies)
    print('Average accuracy: %.2f\n' % (100.0 * np.mean(accuracies)))

    confusionMatrixData = confusion_matrix(all_labels_cm, all_preds_cm)
    utils.writeMetrics(args, accuracies, confusionMatrixData)

    torch.save(icarl.net,os.path.join(args.out_path,"final_net.pth"))


if __name__ == '__main__':
  for i in range(args.num_test):
    print('************** Test iter {} **************'.format(i))
    main()