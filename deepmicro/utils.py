# argparse
import argparse
import numpy as np


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
    common.add_argument("--aeloss", help="set autoencoder reconstruction loss function", type=str,
                        choices=['mse', 'binary_crossentropy'], default='mse')
    common.add_argument("--ae_oact", help="output layer sigmoid activation function on/off", action='store_true')
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
