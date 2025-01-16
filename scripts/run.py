import os,time

timestamp=time.strftime("%Y-%m-%d-%H-%M-%S")

def run_experiment(gpu,dataset,imb,suffix):
    os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu)

    test_num = 1
    classifier = "RF" # "SVM", "RF", "KNN", "NCM"，FCC
    herds = ["MES"] # "random"，"icarl", "MES", "mixed"
    X_train = f"datasets/{dataset}/{imb}/X_train.npy"
    y_train = f"datasets/{dataset}/{imb}/y_train.npy"
    X_test = f"datasets/{dataset}/{imb}/X_test.npy"
    y_test = f"datasets/{dataset}/{imb}/y_test.npy"

    if dataset == 'TE' and imb in ['imbalanced']:
        K=100
        classes_per_group=2
        num_iter = 500
    elif dataset == 'TE' and imb in ['long_tail']:
        K=40
        classes_per_group=2
        num_iter = 500
    elif dataset == 'MFF' and imb in ['imbalanced']:
        K=10
        classes_per_group=1
        num_iter = 100
    elif dataset == 'MFF' and imb in ['long_tail']:
        K=5
        num_iter = 100
        classes_per_group=1

    for herd in herds:
        logf = f'logs/{dataset}/{imb}/{herd}/{timestamp}'
        os.makedirs(logf,exist_ok=True)

        command = f"nohup python -u icarl_scl_experiment.py \
            --classifier {classifier} \
            --X_train {X_train} \
            --y_train {y_train} \
            --X_test {X_test} \
            --y_test {y_test} \
            --herding {herd} \
            --out_path {logf} \
            --gpu {gpu} \
            --NUM_EPOCHS {num_iter} \
            --num_test {test_num} \
            --K {K} \
            --classify_net {dataset} \
            --classes_per_group {classes_per_group} \
            > {logf}/scl_{herd}_{classifier}_{K}_{suffix}.log 2>&1 &"

        os.system(command)



run_experiment(2,'TE','imbalanced','')
run_experiment(1,'TE','long_tail','')
run_experiment(3,'MFF','imbalanced','')
run_experiment(3,'MFF','long_tail','')

