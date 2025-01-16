import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', default='NCM')
    parser.add_argument('--X_train', default="datasets/TE/imbalanced/X_train.npy")
    parser.add_argument('--y_train', default="datasets/TE/imbalanced/y_train.npy")
    parser.add_argument('--X_test', default="datasets/TE/imbalanced/X_test.npy")
    parser.add_argument('--y_test', default="datasets/TE/imbalanced/y_test.npy")
    parser.add_argument('--out_path', help="output directory", default="logs/visualize")
    parser.add_argument('--all')
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--herding', default='icarl')
    parser.add_argument('--classes_per_group', default=2)
    parser.add_argument('--classify_net', default='TE')

    parser.add_argument('--DEVICE', default='cuda')  # 'cuda' or 'cpu'
    parser.add_argument('--NUM_CLASSES', default=2)
    parser.add_argument('--BATCH_SIZE', default=2048)
    parser.add_argument('--LR', default=0.01)
    parser.add_argument('--MOMENTUM', default=0.9)
    parser.add_argument('--WEIGHT_DECAY', default=1e-5)
    parser.add_argument('--NUM_EPOCHS', type=int, default=5)
    parser.add_argument('--GAMMA', default=0.2)
    parser.add_argument('--LOG_FREQUENCY', default=100)
    parser.add_argument('--MILESTONES', default=[200, 400])
    parser.add_argument('--RANDOM_SEED', default=66)
    parser.add_argument('--gpu', default=1)
    parser.add_argument('--n_classes', default=0)
    parser.add_argument('--feature_size', default=10)
    parser.add_argument('--num_test', type=int, default=1)

    args = parser.parse_args()

    return args
