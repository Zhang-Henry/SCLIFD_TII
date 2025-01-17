# SCLIFD: Class Incremental Fault Diagnosis under Limited Fault Data via Supervised Contrastive Knowledge Distillation

This is the official code for IEEE Transactions on Industrial Informatics paper [Class Incremental Fault Diagnosis under Limited Fault Data via Supervised Contrastive Knowledge Distillation](https://arxiv.org/pdf/2501.09525).

## Installation

To set up the environment for this project using conda, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Zhang-Henry/SCLIFD_TII.git
    cd SCLIFD
    ```

2. **Create a conda environment:**

    ```bash
    conda create --name sclifd python=3.8
    conda activate sclifd
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Verify the installation:**

    ```bash
    pip list
    ```

This will ensure that all necessary dependencies are installed and the environment is set up correctly.

## Running the `run.py` Script

The `run.py` script is used to execute experiments with various configurations. Below are the steps to run the script and an explanation of its usage:

1. **Activate the conda environment:**

    ```bash
    conda activate sclifd
    ```

2. **Run the script:**

    ```bash
    python scripts/run.py
    ```

### Explanation of the Script

The script sets up and runs experiments based on the specified dataset and imbalance type. Here are the key parameters and their meanings:

- **dataset**: The dataset being used (e.g., 'TE', 'MFF').
- **imb**: The type of imbalance in the dataset (e.g., 'imbalanced', 'long_tail').
- **herds**: A list of herding methods to be used in the experiments.
- **timestamp**: A timestamp to uniquely identify the log files.
- **logf**: The directory where log files will be saved.
- **command**: The command that runs the experiment using `nohup` to allow it to run in the background.

### Command Parameters

- `--classifier`: The classifier to be used.
- `--X_train`: Training data features.
- `--y_train`: Training data labels.
- `--X_test`: Testing data features.
- `--y_test`: Testing data labels.
- `--herding`: The exemplar replay method.
- `--out_path`: The output path for logs.
- `--gpu`: The GPU to be used.
- `--NUM_EPOCHS`: The number of epochs for training.
- `--num_test`: The number of tests to be performed.
- `--K`: The number of exemplars to replay.
- `--classify_net`: The dataset name.
- `--classes_per_group`: The number of classes per group.

### Example Usage

To run an experiment with specific parameters, you can modify the script or pass the parameters directly when calling the script. Ensure that the paths and parameters match your setup and requirements.
