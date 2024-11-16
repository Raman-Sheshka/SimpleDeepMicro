# importing numpy, pandas, and matplotlib
import numpy as np
import pandas as pd
from deepmicro.plot_tools import plot_loss

# importing sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn import cluster
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# importing keras
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier # scikit-learn wrapper for keras do not use that
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from tensorflow.keras.models import Model, load_model

# importing util libraries
import datetime
import time
import math
import os
import importlib

# importing custom library
import DNN_models
import exception_handle
from utils import parse_args
#fix np.random.seed for reproducibility in numpy processing
np.random.seed(7)

class DeepMicrobiome(object):
    def __init__(self, data, seed, data_dir):
        self.t_start = time.time()
        self.filename = str(data)
        self.data = self.filename.split('.')[0]
        self.seed = seed
        self.data_dir = data_dir
        self.prefix = ''
        self.representation_only = False
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.history = {}
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.cae = None

    def load_data(self, feature_string, label_string, label_dict, dtype=None):
        # read file
        filename = self.data_dir + "data_test/" + self.filename
        if os.path.isfile(filename):
            raw = pd.read_csv(filename, sep='\t', index_col=0, header=None)
        else:
            print("FileNotFoundError: File {} does not exist".format(filename))
            exit()

        # select rows having feature index identifier string
        X = raw.loc[raw.index.str.contains(feature_string, regex=False)].T

        # get class labels
        Y = raw.loc[label_string] #'disease'
        Y = Y.replace(label_dict)

        # train and test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X.values.astype(dtype), Y.values.astype('int'), test_size=0.2, random_state=self.seed, stratify=Y.values)
        self.print_data_shapes()

    def load_custom_data(self, dtype=None):
        # read file
        filename = self.data_dir + "data_test/" + self.filename
        if os.path.isfile(filename):
            raw = pd.read_csv(filename, sep=',', index_col=False, header=None)
        else:
            print("FileNotFoundError: File {} does not exist".format(filename))
            exit()

        # load data_test
        self.X_train = raw.values.astype(dtype)

        # put nothing or zeros for y_train, y_test, and X_test
        self.y_train = np.zeros(shape=(self.X_train.shape[0])).astype(dtype)
        self.X_test = np.zeros(shape=(1,self.X_train.shape[1])).astype(dtype)
        self.y_test = np.zeros(shape=(1,)).astype(dtype)
        self.print_data_shapes(train_only=True)

    def load_custom_data_with_labels(self, label_data, dtype=None):
        # read file
        filename = self.data_dir + "data_test/" + self.filename
        label_filename = self.data_dir + "data_test/" + label_data
        if os.path.isfile(filename) and os.path.isfile(label_filename):
            raw = pd.read_csv(filename, sep=',', index_col=False, header=None)
            label = pd.read_csv(label_filename, sep=',', index_col=False, header=None)
        else:
            if not os.path.isfile(filename):
                print("FileNotFoundError: File {} does not exist".format(filename))
            if not os.path.isfile(label_filename):
                print("FileNotFoundError: File {} does not exist".format(label_filename))
            exit()

        # label data_test validity check
        if not label.values.shape[1] > 1:
            label_flatten = label.values.reshape((label.values.shape[0]))
        else:
            print('FileSpecificationError: The label file contains more than 1 column.')
            exit()

        # train and test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(raw.values.astype(dtype),
                                                                                label_flatten.astype('int'), test_size=0.2,
                                                                                random_state=self.seed,
                                                                                stratify=label_flatten)
        self.print_data_shapes()


    #Principal Component Analysis
    def pca(self, ratio=0.99):
        # manipulating an experiment identifier in the output file
        self.prefix = self.prefix + 'PCA_'

        # PCA
        pca = PCA()
        pca.fit(self.X_train)
        n_comp = 0
        ratio_sum = 0.0

        for comp in pca.explained_variance_ratio_:
            ratio_sum += comp
            n_comp += 1
            if ratio_sum >= ratio:  # Selecting components explaining 99% of variance
                break

        pca = PCA(n_components=n_comp)
        pca.fit(self.X_train)

        X_train = pca.transform(self.X_train)
        X_test = pca.transform(self.X_test)

        # applying the eigenvectors to the whole training and the test set.
        self.X_train = X_train
        self.X_test = X_test
        self.print_data_shapes()

    #Gausian Random Projection
    def rp(self):
        # manipulating an experiment identifier in the output file
        self.prefix = self.prefix + 'RandP_'
        # GRP
        rf = GaussianRandomProjection(eps=0.5)
        rf.fit(self.X_train)

        # applying GRP to the whole training and the test set.
        self.X_train = rf.transform(self.X_train)
        self.X_test = rf.transform(self.X_test)
        self.print_data_shapes()

    #Shallow Autoencoder & Deep Autoencoder
    def ae(self,
           dims = [50],
           epochs= 2000,
           batch_size=100,
           verbose=2,
           loss='mean_squared_error',
           latent_act=False,
           output_act=False,
           act='relu',
           patience=20,
           val_rate=0.2,
           no_trn=False
           ):

        # manipulating an experiment identifier in the output file
        if patience != 20:
            self.prefix += 'p' + str(patience) + '_'

        # filename for temporary model checkpoint
        model_name = self.prefix + self.data + '.h5'

        # clean up model checkpoint before use
        if os.path.isfile(model_name):
            os.remove(model_name)

        # callbacks for each epoch
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1),
                     ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)]

        # spliting the training set into the inner-train and the inner-test set (validation set)
        X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(self.X_train, self.y_train, test_size=val_rate, random_state=self.seed, stratify=self.y_train)

        # insert input shape into dimension list
        dims.insert(0, X_inner_train.shape[1])

        # create autoencoder model
        self.autoencoder, self.encoder = DNN_models.autoencoder(dims, act=act, latent_act=latent_act, output_act=output_act)
        self.autoencoder.summary()

        if no_trn:
            return

        # compile model
        self.autoencoder.compile(optimizer='adam', loss=loss)

        # fit model
        self.history = self.autoencoder.fit(X_inner_train, X_inner_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                             verbose=verbose, validation_data=(X_inner_test, X_inner_test))
        # save loss progress
        self.save_loss_progress()

        # load best model
        self.autoencoder = load_model(model_name)
        layer_idx = int((len(self.autoencoder.layers) - 1) / 2)
        self.encoder = Model(self.autoencoder.layers[0].input, self.autoencoder.layers[layer_idx].output)

        # applying the learned encoder into the whole training and the test set.
        self.X_train = self.encoder.predict(self.X_train)
        self.X_test = self.encoder.predict(self.X_test)

    # Variational Autoencoder
    def vae(self, dims = [10], epochs=2000, batch_size=100, verbose=2, loss='mse', output_act=False, act='relu', patience=25, beta=1.0, warmup=True, warmup_rate=0.01, val_rate=0.2, no_trn=False):

        # manipulating an experiment identifier in the output file
        if patience != 25:
            self.prefix += 'p' + str(patience) + '_'
        if warmup:
            self.prefix += 'w' + str(warmup_rate) + '_'
        self.prefix += 'VAE'
        if loss == 'binary_crossentropy':
            self.prefix += 'b'
        if output_act:
            self.prefix += 'T'
        if beta != 1:
            self.prefix += 'B' + str(beta)
        self.prefix += str(dims).replace(", ", "-") + '_'
        if act == 'sigmoid':
            self.prefix += 'sig_'

        # filename for temporary model checkpoint
        model_name = self.prefix + self.data + '.h5'

        # clean up model checkpoint before use
        if os.path.isfile(model_name):
            os.remove(model_name)

        # callbacks for each epoch
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1),
                     ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True,save_weights_only=True)]

        # warm-up callback
        warm_up_cb = LambdaCallback(on_epoch_end=lambda epoch, logs: [warm_up(epoch)])  # , print(epoch), print(K.get_value(beta))])

        # warm-up implementation
        def warm_up(epoch):
            val = epoch * warmup_rate
            if val <= 1.0:
                K.set_value(beta, val)
        # add warm-up callback if requested
        if warmup:
            beta = K.variable(value=0.0)
            callbacks.append(warm_up_cb)

        # spliting the training set into the inner-train and the inner-test set (validation set)
        X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(self.X_train, self.y_train,
                                                                                    test_size=val_rate,
                                                                                    random_state=self.seed,
                                                                                    stratify=self.y_train)

        # insert input shape into dimension list
        dims.insert(0, X_inner_train.shape[1])

        # create vae model
        self.vae, self.encoder, self.decoder = DNN_models.variational_ae(dims, act=act, recon_loss=loss, output_act=output_act, beta=beta)
        self.vae.summary()

        if no_trn:
            return

        # fit
        self.history = self.vae.fit(X_inner_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose, validation_data=(X_inner_test, None))

        # save loss progress
        self.save_loss_progress()

        # load best model
        self.vae.load_weights(model_name)
        self.encoder = self.vae.layers[1]

        # applying the learned encoder into the whole training and the test set.
        _, _, self.X_train = self.encoder.predict(self.X_train)
        _, _, self.X_test = self.encoder.predict(self.X_test)

    # Convolutional Autoencoder
    def cae(self, dims = [32], epochs=2000, batch_size=100, verbose=2, loss='mse', output_act=False, act='relu', patience=25, val_rate=0.2, rf_rate = 0.1, st_rate = 0.25, no_trn=False):

        # manipulating an experiment identifier in the output file
        self.prefix += 'CAE'
        if loss == 'binary_crossentropy':
            self.prefix += 'b'
        if output_act:
            self.prefix += 'T'
        self.prefix += str(dims).replace(", ", "-") + '_'
        if act == 'sigmoid':
            self.prefix += 'sig_'

        # filename for temporary model checkpoint
        model_name = self.prefix + self.data + '.h5'

        # clean up model checkpoint before use
        if os.path.isfile(model_name):
            os.remove(model_name)

        # callbacks for each epoch
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1),
                     ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True,save_weights_only=True)]


        # fill out blank
        onside_dim = int(math.sqrt(self.X_train.shape[1])) + 1
        enlarged_dim = onside_dim ** 2
        self.X_train = np.column_stack((self.X_train, np.zeros((self.X_train.shape[0], enlarged_dim - self.X_train.shape[1]))))
        self.X_test = np.column_stack((self.X_test, np.zeros((self.X_test.shape[0], enlarged_dim - self.X_test.shape[1]))))

        # reshape
        self.X_train = np.reshape(self.X_train, (len(self.X_train), onside_dim, onside_dim, 1))
        self.X_test = np.reshape(self.X_test, (len(self.X_test), onside_dim, onside_dim, 1))
        self.print_data_shapes()

        # spliting the training set into the inner-train and the inner-test set (validation set)
        X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(self.X_train, self.y_train,
                                                                                    test_size=val_rate,
                                                                                    random_state=self.seed,
                                                                                    stratify=self.y_train)

        # insert input shape into dimension list
        dims.insert(0, (onside_dim, onside_dim, 1))

        # create cae model
        self.cae, self.encoder = DNN_models.conv_autoencoder(dims, act=act, output_act=output_act, rf_rate = rf_rate, st_rate = st_rate)
        self.cae.summary()
        if no_trn:
            return

        # compile
        self.cae.compile(optimizer='adam', loss=loss)

        # fit
        self.history = self.cae.fit(X_inner_train, X_inner_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose, validation_data=(X_inner_test, X_inner_test, None))

        # save loss progress
        self.save_loss_progress()

        # load best model
        self.cae.load_weights(model_name)
        if len(self.cae.layers) % 2 == 0:
            layer_idx = int((len(self.cae.layers) - 2) / 2)
        else:
            layer_idx = int((len(self.cae.layers) - 1) / 2)
        self.encoder = Model(self.cae.layers[0].input, self.cae.layers[layer_idx].output)

        # applying the learned encoder into the whole training and the test set.
        self.X_train = self.encoder.predict(self.X_train)
        self.X_test = self.encoder.predict(self.X_test)
        self.print_data_shapes()

    # Classification
    def ml_classification(self,
                          hyper_parameters,
                          method='svm',
                          cv=5,
                          scoring='roc_auc',
                          n_jobs=1,
                          cache_size=10000
                          ):
        clf_start_time = time.time()

        print("# Tuning hyper-parameters")
        print(self.X_train.shape, self.y_train.shape)

        # Support Vector Machine
        if method == 'svm':
            clf = GridSearchCV(SVC(probability=True, cache_size=cache_size), hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring, n_jobs=n_jobs, verbose=100, )
            clf.fit(self.X_train, self.y_train)

        # Random Forest
        if method == 'rf':
            clf = GridSearchCV(RandomForestClassifier(n_jobs=-1, random_state=0), hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring, n_jobs=n_jobs, verbose=100)
            clf.fit(self.X_train, self.y_train)


        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)

        # Evaluate performance of the best model on test set
        y_true, y_pred = self.y_test, clf.predict(self.X_test)
        y_prob = clf.predict_proba(self.X_test)

        # Performance Metrics: AUC, ACC, Recall, Precision, F1_score
        metrics = [round(roc_auc_score(y_true, y_prob[:, 1]), 4),
                   round(accuracy_score(y_true, y_pred), 4),
                   round(recall_score(y_true, y_pred), 4),
                   round(precision_score(y_true, y_pred), 4),
                   round(f1_score(y_true, y_pred), 4), ]

        # time stamp
        metrics.append(str(datetime.datetime.now()))

        # running time
        metrics.append(round( (time.time() - self.t_start), 2))

        # classification time
        metrics.append(round( (time.time() - clf_start_time), 2))

        # best hyper-parameter append
        metrics.append(str(clf.best_params_))

        # Write performance metrics as a file
        res = pd.DataFrame([metrics], index=[self.prefix + method])
        with open(self.data_dir + "results/" + self.data + "_result.txt", 'a') as f:
            res.to_csv(f, header=None)

        print('Accuracy metrics')
        print('AUC, ACC, Recall, Precision, F1_score, time-end, runtime(sec), classfication time(sec), best hyper-parameter')
        print(metrics)

    def print_data_shapes(self, train_only=False):
        print("X_train.shape: ", self.X_train.shape)
        if not train_only:
            print("y_train.shape: ", self.y_train.shape)
            print("X_test.shape: ", self.X_test.shape)
            print("y_test.shape: ", self.y_test.shape)

    # ploting loss progress over epochs
    def save_loss_progress(self):
        #print(self.history.history.keys())
        #print(type(self.history.history['loss']))
        #print(min(self.history.history['loss']))

        loss_collector, loss_max_at_the_end = self.save_loss_progress_ylim()

        # save loss progress - train and val loss only
        figure_name = self.prefix + self.data + '_' + str(self.seed)
        plot_loss(figure_title=figure_name,
                  loss_collector=loss_collector,
                  loss_max_at_the_end=loss_max_at_the_end,
                  history=self.history,
                  save_path_dir=self.data_dir,
                  save=True,
                  show=False
                  )

    # supporting loss plot
    def save_loss_progress_ylim(self):
        loss_collector = []
        loss_max_at_the_end = 0.0
        for hist in self.history.history:
            current = self.history.history[hist]
            loss_collector += current
            if current[-1] >= loss_max_at_the_end:
                loss_max_at_the_end = current[-1]
        return loss_collector, loss_max_at_the_end

if __name__ == '__main__':

    args = parse_args()
    # set labels for diseases and controls
    label_dict = {
        # Controls
        'n': 0,
        # Chirrhosis
        'cirrhosis': 1,
        # Colorectal Cancer
        'cancer': 1, 'small_adenoma': 0,
        # IBD
        'ibd_ulcerative_colitis': 1, 'ibd_crohn_disease': 1,
        # T2D and WT2D
        't2d': 1,
        # Obesity
        'leaness': 0, 'obesity': 1,
    }

    # hyper-parameter grids for classifiers
    rf_hyper_parameters = [{'n_estimators': [s for s in range(100, 1001, 200)],
                            'max_features': ['sqrt', 'log2'],
                            'min_samples_leaf': [1, 2, 3, 4, 5],
                            'criterion': ['gini', 'entropy']
                            }, ]
    #svm_hyper_parameters_pasolli = [{'C': [2 ** s for s in range(-5, 16, 2)], 'kernel': ['linear']},
    #                        {'C': [2 ** s for s in range(-5, 16, 2)], 'gamma': [2 ** s for s in range(3, -15, -2)],
    #                         'kernel': ['rbf']}]
    svm_hyper_parameters = [{'C': [2 ** s for s in range(-5, 6, 2)], 'kernel': ['linear']},
                            {'C': [2 ** s for s in range(-5, 6, 2)], 'gamma': [2 ** s for s in range(3, -15, -2)],'kernel': ['rbf']}]
    mlp_hyper_parameters = [{'numHiddenLayers': [1, 2, 3],
                             'epochs': [30, 50, 100, 200, 300],
                             'numUnits': [10, 30, 50, 100],
                             'dropout_rate': [0.1, 0.3],
                             },]


    # run exp function
    def run_exp(seed):

        # create an object and load data_test
        ## no argument founded
        if args.data == None and args.custom_data == None:
            print("[Error] Please specify an input file. (use -h option for help)")
            exit()
        ## provided data_test
        elif args.data != None:
            dm = DeepMicrobiome(data=args.data + '.txt', seed=seed, data_dir=args.data_dir)

            ## specify feature string
            feature_string = ''
            data_string = str(args.data)
            if data_string.split('_')[0] == 'abundance':
                feature_string = "k__"
            if data_string.split('_')[0] == 'marker':
                feature_string = "gi|"

            ## load data_test into the object
            dm.load_data(feature_string=feature_string, label_string='disease', label_dict=label_dict,
                         dtype=dtypeDict[args.dataType])

        ## user data_test
        elif args.custom_data != None:

            ### without labels - only conducting representation learning
            if args.custom_data_labels == None:
                dm = DeepMicrobiome(data=args.custom_data, seed=seed, data_dir=args.data_dir)
                dm.load_custom_data(dtype=dtypeDict[args.dataType])

            ### with labels - conducting representation learning + classification
            else:
                dm = DeepMicrobiome(data=args.custom_data, seed=seed, data_dir=args.data_dir)
                dm.load_custom_data_with_labels(label_data=args.custom_data_labels, dtype=dtypeDict[args.dataType])

        else:
            exit()

        num_rl_required = args.pca + args.ae + args.rp + args.vae + args.cae

        if num_rl_required > 1:
            raise ValueError('No multiple dimensionality Reduction')

        # time check after data_test has been loaded
        dm.t_start = time.time()

        # Representation learning (Dimensionality reduction)
        if args.pca:
            dm.pca()
        if args.ae:
            dm.ae(dims=[int(i) for i in args.dims.split(',')], act=args.act, epochs=args.max_epochs, loss=args.aeloss,
                  latent_act=args.ae_lact, output_act=args.ae_oact, patience=args.patience, no_trn=args.no_trn)
        if args.vae:
            dm.vae(dims=[int(i) for i in args.dims.split(',')], act=args.act, epochs=args.max_epochs, loss=args.aeloss, output_act=args.ae_oact,
                   patience= 25 if args.patience==20 else args.patience, beta=args.vae_beta, warmup=args.vae_warmup, warmup_rate=args.vae_warmup_rate, no_trn=args.no_trn)
        if args.cae:
            dm.cae(dims=[int(i) for i in args.dims.split(',')], act=args.act, epochs=args.max_epochs, loss=args.aeloss, output_act=args.ae_oact,
                   patience=args.patience, rf_rate = args.rf_rate, st_rate = args.st_rate, no_trn=args.no_trn)
        if args.rp:
            dm.rp()

        # write the learned representation of the training set as a file
        if args.save_rep:
            if num_rl_required == 1:
                rep_file = dm.data_dir + "results/" + dm.prefix + dm.data + "_rep.csv"
                pd.DataFrame(dm.X_train).to_csv(rep_file, header=None, index=None)
                print("The learned representation of the training set has been saved in '{}'".format(rep_file))
            else:
                print("Warning: Command option '--save_rep' is not applied as no representation learning or dimensionality reduction has been conducted.")

        # Classification
        if args.no_clf or (args.data == None and args.custom_data_labels == None):
            print("Classification task has been skipped.")
        else:
            # turn off GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            importlib.reload(tensorflow.keras)# WTF is this??

            # training classification models
            if args.method == "svm":
                dm.classification(hyper_parameters=svm_hyper_parameters, method='svm', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring, cache_size=args.svm_cache)
            elif args.method == "rf":
                dm.classification(hyper_parameters=rf_hyper_parameters, method='rf', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring)
            elif args.method == "mlp":
                dm.classification(hyper_parameters=mlp_hyper_parameters, method='mlp', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring)
            elif args.method == "svm_rf":
                dm.classification(hyper_parameters=svm_hyper_parameters, method='svm', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring, cache_size=args.svm_cache)
                dm.classification(hyper_parameters=rf_hyper_parameters, method='rf', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring)
            else:
                dm.classification(hyper_parameters=svm_hyper_parameters, method='svm', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring, cache_size=args.svm_cache)
                dm.classification(hyper_parameters=rf_hyper_parameters, method='rf', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring)
                dm.classification(hyper_parameters=mlp_hyper_parameters, method='mlp', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring)



    # run experiments
    try:
        if args.repeat > 1:
            for i in range(args.repeat):
                run_exp(i)
        else:
            run_exp(args.seed)

    except OSError as error:
        exception_handle.log_exception(error)
