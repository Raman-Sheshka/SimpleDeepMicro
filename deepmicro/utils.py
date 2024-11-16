# argparse
import argparse
import numpy as np
from pydantic import BaseModel, Field, validator
from typing import Optional, List

class UserChoices(BaseModel):
    data_test: Optional[str] = Field(None, description="Prefix of dataset to open", choices=[
        "abundance_Cirrhosis", "abundance_Colorectal", "abundance_IBD",
        "abundance_Obesity", "abundance_T2D", "abundance_WT2D",
        "marker_Cirrhosis", "marker_Colorectal", "marker_IBD",
        "marker_Obesity", "marker_T2D", "marker_WT2D"
    ])
    custom_data: Optional[str] = Field(None, description="Filename for custom input data under the 'data_test' folder")
    custom_data_labels: Optional[str] = Field(None, description="Filename for custom input labels under the 'data_test' folder")

    data_dir: str = Field("", description="Custom path for both '/data_test' and '/results' folders")
    dataType: str = Field("float64", description="Specify data type for numerical values", choices=["float16", "float32", "float64"])

    seed: int = Field(0, description="Random seed for train and test split")
    repeat: int = Field(5, description="Repeat experiment x times by changing random seed for splitting data")

    numFolds: int = Field(5, description="The number of folds for cross-validation in the training set")

    method: str = Field("all", description="Classifier(s) to use", choices=["all", "svm", "rf", "mlp", "svm_rf"])

    svm_cache: int = Field(1000, description="Cache size for SVM run")

    numJobs: int = Field(-2, description="The number of jobs used in parallel GridSearch")
    scoring: str = Field("roc_auc", description="Metrics used to optimize method", choices=['roc_auc', 'accuracy', 'f1', 'recall', 'precision'])

    pca: bool = Field(False, description="Run PCA")
    rp: bool = Field(False, description="Run Random Projection")

    ae: bool = Field(False, description="Run Autoencoder or Deep Autoencoder")
    vae: bool = Field(False, description="Run Variational Autoencoder")
    cae: bool = Field(False, description="Run Convolutional Autoencoder")
    save_rep: bool = Field(False, description="Write the learned representation of the training set as a file")

    ae_loss: Optional[str] = Field("mse", description="Set autoencoder reconstruction loss function", choices=['mse', 'binary_crossentropy'])
    ae_output_act: bool = Field(False, description="Output layer sigmoid activation function on/off")
    act: str = Field("relu", description="Activation function for hidden layers", choices=['relu', 'sigmoid'])
    dims: str = Field("50", description="Comma-separated dimensions for deep representation learning")
    max_epochs: int = Field(2000, description="Maximum epochs when training autoencoder")
    patience: int = Field(20, description="The number of epochs which can be executed without the improvement in validation loss")
    ae_lact: bool = Field(False, description="Latent layer activation function on/off")
    vae_beta: float = Field(1.0, description="Weight of KL term")
    vae_warmup: bool = Field(False, description="Turn on warm up")
    vae_warmup_rate: float = Field(0.01, description="Warm-up rate which will be multiplied by current epoch to calculate current beta")
    rf_rate: float = Field(0.1, description="What percentage of input size will be the receptive field (kernel) size?")
    st_rate: float = Field(0.25, description="What percentage of receptive field (kernel) size will be the stride size?")
    no_trn: bool = Field(False, description="Stop before learning representation to see specified autoencoder structure")
    no_clf: bool = Field(False, description="Skip classification tasks")
    exec_mode: str = Field(..., description="Execution mode", choices=["test", "run"])

    @validator('dims')
    def validate_dims(cls, v):
        try:
            dims = [int(dim) for dim in v.split(',')]
            if not all(dim > 0 for dim in dims):
                raise ValueError
        except ValueError:
            raise ValueError("dims must be a comma-separated list of positive integers")
        return v

    @validator('data_dir')
    def validate_data_dir(cls, v, values):
        if 'exec_mode' in values and values['exec_mode'] == 'test':
            return '/data_test'
        elif 'exec_mode' in values and values['exec_mode'] == 'run' and not v:
            raise ValueError("data_dir must be provided in 'run' mode")
        return v

def parse_args():
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser()
    parser._action_groups.pop()

    # load data_test
    load_data = parser.add_argument_group('Loading data_test')
    load_data.add_argument("-d", "--data_test", help="prefix of dataset to open (e.g. abundance_Cirrhosis)", type=str,
                        choices=["abundance_Cirrhosis", "abundance_Colorectal", "abundance_IBD",
                                 "abundance_Obesity", "abundance_T2D", "abundance_WT2D",
                                 "marker_Cirrhosis", "marker_Colorectal", "marker_IBD",
                                 "marker_Obesity", "marker_T2D", "marker_WT2D",
                                 ])
    load_data.add_argument("-cd", "--custom_data", help="filename for custom input data_test under the 'data_test' folder", type=str,)
    load_data.add_argument("-cl", "--custom_data_labels", help="filename for custom input labels under the 'data_test' folder", type=str,)
    load_data.add_argument("-p", "--data_dir", help="custom path for both '/data_test' and '/results' folders", default="")
    load_data.add_argument("-dt", "--dataType", help="Specify data_test type for numerical values (float16, float32, float64)",
                        default="float64", type=str, choices=["float16", "float32", "float64"])
    dtypeDict = {"float16": np.float16, "float32": np.float32, "float64": np.float64}

    # experiment design
    exp_design = parser.add_argument_group('Experiment design')
    exp_design.add_argument("-s", "--seed", help="random seed for train and test split", type=int, default=0)
    exp_design.add_argument("-r", "--repeat", help="repeat experiment x times by changing random seed for splitting data_test",
                        default=5, type=int)

    # classification
    classification = parser.add_argument_group('Classification')
    classification.add_argument("-f", "--numFolds", help="The number of folds for cross-validation in the tranining set",
                        default=5, type=int)
    classification.add_argument("-m", "--method", help="classifier(s) to use", type=str, default="all",
                        choices=["all", "svm", "rf", "mlp", "svm_rf"])
    classification.add_argument("-sc", "--svm_cache", help="cache size for svm run", type=int, default=1000)
    classification.add_argument("-t", "--numJobs",
                        help="The number of jobs used in parallel GridSearch. (-1: utilize all possible cores; -2: utilize all possible cores except one.)",
                        default=-2, type=int)
    parser.add_argument("--scoring", help="Metrics used to optimize method", type=str, default='roc_auc',
                        choices=['roc_auc', 'accuracy', 'f1', 'recall', 'precision'])

    # representation learning & dimensionality reduction algorithms
    rl = parser.add_argument_group('Representation learning')
    rl.add_argument("--pca", help="run PCA", action='store_true')
    rl.add_argument("--rp", help="run Random Projection", action='store_true')
    rl.add_argument("--ae", help="run Autoencoder or Deep Autoencoder", action='store_true')
    rl.add_argument("--vae", help="run Variational Autoencoder", action='store_true')
    rl.add_argument("--cae", help="run Convolutional Autoencoder", action='store_true')
    rl.add_argument("--save_rep", help="write the learned representation of the training set as a file", action='store_true')

    # detailed options for representation learning
    ## common options
    common = parser.add_argument_group('Common options for representation learning (SAE,DAE,VAE,CAE)')
    common.add_argument("--ae_loss", help="set autoencoder reconstruction loss function", type=str,
                        choices=['mse', 'binary_crossentropy'], default='mse')
    common.add_argument("--ae_output_act", help="output layer sigmoid activation function on/off", action='store_true')
    common.add_argument("-a", "--act", help="activation function for hidden layers", type=str, default='relu',
                        choices=['relu', 'sigmoid'])
    common.add_argument("-dm", "--dims",
                        help="Comma-separated dimensions for deep representation learning e.g. (-dm 50,30,20)",
                        type=str, default='50')
    common.add_argument("-e", "--max_epochs", help="Maximum epochs when training autoencoder", type=int, default=2000)
    common.add_argument("-pt", "--patience",
                        help="The number of epochs which can be executed without the improvement in validation loss, right after the last improvement.",
                        type=int, default=20)

    ## AE & DAE only
    AE = parser.add_argument_group('SAE & DAE-specific arguments')
    AE.add_argument("--ae_lact", help="latent layer activation function on/off", action='store_true')

    ## VAE only
    VAE = parser.add_argument_group('VAE-specific arguments')
    VAE.add_argument("--vae_beta", help="weight of KL term", type=float, default=1.0)
    VAE.add_argument("--vae_warmup", help="turn on warm up", action='store_true')
    VAE.add_argument("--vae_warmup_rate", help="warm-up rate which will be multiplied by current epoch to calculate current beta", default=0.01, type=float)

    ## CAE only
    CAE = parser.add_argument_group('CAE-specific arguments')
    CAE.add_argument("--rf_rate", help="What percentage of input size will be the receptive field (kernel) size? [0,1]", type=float, default=0.1)
    CAE.add_argument("--st_rate", help="What percentage of receptive field (kernel) size will be the stride size? [0,1]", type=float, default=0.25)

    # other options
    others = parser.add_argument_group('other optional arguments')
    others.add_argument("--no_trn", help="stop before learning representation to see specified autoencoder structure", action='store_true')
    others.add_argument("--no_clf", help="skip classification tasks", action='store_true')


    args = parser.parse_args()
    print(args)
    return args
