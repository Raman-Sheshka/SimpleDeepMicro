import tensorflow as tf
import optuna
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras import layers, models
import logging

import tensorflow as tf
import optuna
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy, mse
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
import logging


class MLPClassifier:
    def __init__(self,
                 input_dim,
                 num_hidden_layers=3,
                 num_units=64,
                 dropout_rate=0.5
                 ):
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_units = num_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        if self.num_hidden_layers >= 1:
            model.add(Dense(self.num_units, input_dim=self.input_dim, activation='relu'))
            model.add(Dropout(self.dropout_rate))

            for i in range(self.num_hidden_layers - 1):
                self.num_units = self.num_units // 2
                model.add(Dense(self.num_units, activation='relu'))
                model.add(Dropout(self.dropout_rate))

            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(1, input_dim=self.input_dim, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def get_model(self):
        return self.model


class HyperparameterOptimizer:
    def __init__(self, X_train, y_train, n_trials=100):
        self.X_train = X_train
        self.y_train = y_train
        self.n_trials = n_trials

    def create_model(self, trial):
        model = models.Sequential()
        n_layers = trial.suggest_int('n_layers', 1, 5)

        model.add(layers.Dense(
            trial.suggest_int('n_units_0', 16, 256),
            activation=trial.suggest_categorical('activation_0', ['relu', 'elu']),
            input_shape=(self.X_train.shape[1],)
        ))

        for i in range(n_layers - 1):
            model.add(layers.Dense(
                trial.suggest_int(f'n_units_{i+1}', 16, 256),
                activation=trial.suggest_categorical(f'activation_{i+1}', ['relu', 'elu'])
            ))
            model.add(layers.Dropout(trial.suggest_float(f'dropout_{i}', 0.1, 0.5)))

        model.add(layers.Dense(1))
        return model

    def objective(self, trial):
        batch_size = trial.suggest_int('batch_size', 16, 128)
        epochs = trial.suggest_int('epochs', 10, 100)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in kf.split(self.X_train):
            X_train_fold, X_val_fold = self.X_train[train_idx], self.X_train[val_idx]
            y_train_fold, y_val_fold = self.y_train[train_idx], self.y_train[val_idx]

            model = self.create_model(trial)
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mse')

            history = model.fit(
                X_train_fold, y_train_fold,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val_fold, y_val_fold),
                verbose=0
            )

            val_score = history.history['val_loss'][-1]
            scores.append(val_score)

        return np.mean(scores)

    def optimize_hyperparameters(self):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1)
        return study.best_params, study.best_value


# Usage example:
"""
X_train = ... # Your training data_test
y_train = ... # Your target values

mlp_classifier = MLPClassifier(input_dim=X_train.shape[1])
model = KerasClassifier(build_fn=mlp_classifier.get_model, verbose=0)
clf = GridSearchCV(estimator=model, param_grid=hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring, n_jobs=n_jobs, verbose=100)
clf.fit(X_train, y_train, batch_size=32)

optimizer = HyperparameterOptimizer(X_train, y_train, n_trials=100)
best_params, best_score = optimizer.optimize_hyperparameters()
print(f"Best parameters: {best_params}")
print(f"Best validation score: {best_score}")
"""

# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# create MLP model
def mlp_model(input_dim, num_hidden_layers=3, num_units=64, dropout_rate=0.5):

    model = Sequential()

    #Check number of hidden layers
    if num_hidden_layers >= 1:
        # First Hidden layer
        model.add(Dense(num_units, input_dim=input_dim, activation='relu'))
        model.add(Dropout(dropout_rate))

        # Second to the last hidden layers
        for i in range(num_hidden_layers - 1):
            num_units = num_units // 2
            model.add(Dense(num_units, activation='relu'))
            model.add(Dropout(dropout_rate))

        # output layer
        model.add(Dense(1, activation='sigmoid'))

    else:
        # output layer
        model.add(Dense(1, input_dim=input_dim, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', )#metrics=['accuracy'])

    return model

model = KerasClassifier(build_fn=DNN_models.mlp_model, input_dim=self.X_train.shape[1], verbose=0, )
clf = GridSearchCV(estimator=model, param_grid=hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring, n_jobs=n_jobs, verbose=100)
            clf.fit(self.X_train, self.y_train, batch_size=32)


# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def create_model(trial):
    """Create TensorFlow model with Optuna-suggested hyperparameters"""
    model = models.Sequential()

    # Number of layers (between 1-5)
    n_layers = trial.suggest_int('n_layers', 1, 5)

    # Input layer
    model.add(layers.Dense(
        trial.suggest_int(f'n_units_0', 16, 256),
        activation=trial.suggest_categorical('activation_0', ['relu', 'elu']),
        input_shape=(X_train.shape[1],)
    ))

    # Hidden layers
    for i in range(n_layers - 1):
        model.add(layers.Dense(
            trial.suggest_int(f'n_units_{i+1}', 16, 256),
            activation=trial.suggest_categorical(f'activation_{i+1}', ['relu', 'elu'])
        ))

        # Add dropout
        model.add(layers.Dropout(trial.suggest_float(f'dropout_{i}', 0.1, 0.5)))

    # Output layer
    model.add(layers.Dense(1))

    return model

def objective(trial, X, y, n_folds=5):
    """Optuna objective function with cross-validation"""

    # Suggest hyperparameters
    batch_size = trial.suggest_int('batch_size', 16, 128)
    epochs = trial.suggest_int('epochs', 10, 100)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    # Initialize cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Create and compile model
        model = create_model(trial)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')

        # Train model
        history = model.fit(
            X_train_fold, y_train_fold,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val_fold, y_val_fold),
            verbose=0
        )

        # Get validation score
        val_score = history.history['val_loss'][-1]
        scores.append(val_score)

    # Return mean validation score
    return np.mean(scores)

def optimize_hyperparameters(X_train, y_train, n_trials=100):
    """Main function to run Optuna optimization"""

    def run_optimization():
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: objective(trial, X_train, y_train),
            n_trials=n_trials,
            n_jobs=1  # Use 1 for GPU to avoid memory issues
        )
        return study.best_params, study.best_value

    # Run optimization in separate process
    best_params, best_score = run_in_separate_process(run_optimization, ())

    return best_params, best_score

# Usage example:
"""
X_train = ... # Your training data_test
y_train = ... # Your target values

best_params, best_score = optimize_hyperparameters(X_train, y_train, n_trials=100)
print(f"Best parameters: {best_params}")
print(f"Best validation score: {best_score}")
"""
