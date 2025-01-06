# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

""" Neural Networks external optimize modules."""

import math
import random
from numbers import Integral, Real 
import numpy as np
import matplotlib.pyplot as plt

from ..api.property import NNLearner
from ..compat.sklearn import validate_params, Interval, Hidden 
from ..utils.deps_utils import ensure_pkg 
from ..utils.validator import validate_keras_model, check_consistent_length 
from ..utils.validator import check_is_fitted
from ..metrics import get_scorer 

from . import KERAS_DEPS, KERAS_BACKEND, dependency_message

if KERAS_BACKEND: 
    callbacks=KERAS_DEPS.callbacks 
    LSTM = KERAS_DEPS.LSTM
    Conv1D=KERAS_DEPS.Conv1D 
    Adam=KERAS_DEPS.Adam
    SGD=KERAS_DEPS.SGD 
    MaxPooling1D=KERAS_DEPS.MaxPooling1D
    GlobalAveragePooling1D=KERAS_DEPS.GlobalAveragePooling1D
    LayerNormalization = KERAS_DEPS.LayerNormalization 
    EarlyStopping = KERAS_DEPS.EarlyStopping
    MultiHeadAttention = KERAS_DEPS.MultiHeadAttention
    Model = KERAS_DEPS.Model 
    BatchNormalization = KERAS_DEPS.BatchNormalization
    Input = KERAS_DEPS.Input
    Softmax = KERAS_DEPS.Softmax
    Flatten = KERAS_DEPS.Flatten
    Dropout = KERAS_DEPS.Dropout 
    Dense = KERAS_DEPS.Dense
    ReduceLROnPlateau =KERAS_DEPS.ReduceLROnPlateau 
    Layer = KERAS_DEPS.Layer 
    register_keras_serializable=KERAS_DEPS.register_keras_serializable
    concatenate=KERAS_DEPS.concatenate
    tf_set_seed=KERAS_DEPS.set_seed
    
    from .utils import extract_callbacks_from 
    
DEP_MSG = dependency_message('optimize') 

__all__=['QPSOOptimizer']


class QPSOOptimizer(NNLearner):
    @validate_params({
        "n_particles": [Interval(Integral, 1, None, closed='left')],
        "n_features": [Interval(Integral, 1, None, closed='left')], 
        "alpha": [Interval(Real, 0, 1, closed="right")],
        "max_iter": [Interval(Integral, 1, None, closed='left')], 
        "max_bound" : ['array-like', None], 
        "min_bound" : ['array-like', None],
        "random_seed": ['random-state'], 
        "stopping_tol": [Hidden(Real, 0, 1, closed="neither")],
        "head_size": [Interval(Integral, 1, None, closed='left')],
        "num_heads": [Interval(Integral, 1, None, closed='left')], 
        "ff_dim": [Interval(Integral, 1, None, closed='left')],
        "ff_dim1": [Interval(Integral, 1, None, closed='left')], 
        "ff_dim2": [Interval(Integral, 1, None, closed='left')],
        "dropout": [Interval(Real, 0, 1, closed="both")],
        "learning_rate": [Interval(Real, 0, 1, closed='neither')],
        "batch_size": [Interval(Integral, 1, None, closed='left')], 
        "epochs": [Interval(Integral, 1, None, closed='left')], 
        "factor": [Interval(Real, 0, 1, closed='neither')], 
        "use_time_distributed": [bool],
        "verbose": [bool, Real],
    })
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        model, 
        n_particles,
        n_features,
        alpha=0.6,
        max_iter=50,
        max_bound=None,
        min_bound=None,
        scoring ='mape', 
        stopping_tol=1e-4,
        head_size=16,
        num_heads=4,
        ff_dim=64,
        ff_dim1=32,
        ff_dim2=32,
        num_trans_blocks=1,
        mlp_units=(64,),
        dropout=0.1,
        optimizer="adam",
        learning_rate=1e-3,
        batch_size=32,
        epochs=10,
        factor=0.2,
        user_callbacks=None,
        random_seed=None,
        verbose=1,
    ):
        # Main QPSO attributes
        self.model              = model 
        self.n_particles        = n_particles
        self.n_features         = n_features
        self.alpha              = alpha
        self.max_iter           = max_iter
        self.max_bound          = max_bound
        self.min_bound          = min_bound
        self.random_seed        = random_seed
        self.stopping_tol       = stopping_tol
        self.verbose            = verbose

        # Transformer-building relevant hyperparams
        self.head_size         = head_size
        self.num_heads         = num_heads
        self.ff_dim            = ff_dim
        self.ff_dim1           = ff_dim1
        self.ff_dim2           = ff_dim2
        self.num_trans_blocks  = num_trans_blocks
        self.mlp_units         = mlp_units
        self.dropout           = dropout
        

        # Keras compile settings
        self.optimizer         = optimizer
        self.learning_rate     = learning_rate

        # Training loop hyperparams
        self.batch_size        = batch_size
        self.epochs            = epochs

        # Optional user-provided callbacks
        self.user_callbacks    = user_callbacks

        # Seed fix for reproducibility if set
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            tf_set_seed(self.random_seed)


    def _initialize_swarm(self):
        # Initialize swarm positions uniformly within [min_bound, max_bound].
        loc = []
        for _ in range(self.n_particles):
            row_ = []
            for j in range(self.n_features):
                r = random.random()
                val = (
                    r * (self.max_bound[j] - self.min_bound[j])
                    + self.min_bound[j]
                )
                row_.append(val)
            loc.append(row_)
        return loc

    def _build_and_train_model(
        self,
        param_vector,
        X,
        y, 
        fit_params 
    ):
        # Extract hyperparams from param_vector (example).
        # param_vector is expected to have at least 9 entries:
        # e.g. [head_size, num_heads, ff_dim, ff_dim1, ff_dim2,
        #       num_trans_blocks, mlp_units, batch_size, epochs]
        # for demonstration.
        h_size       = int(param_vector[0])
        n_heads      = int(param_vector[1])
        ff_          = int(param_vector[2])
        ff_1         = int(param_vector[3])
        ff_2         = int(param_vector[4])
        n_blocks     = int(param_vector[5])
        mlp_u        = int(param_vector[6])
        b_sz         = int(param_vector[7])
        n_ep         = int(param_vector[8])

        # Build the model with the extracted hyperparams
        # if model='default' or model=None, then use the default transformer. 
        if self.model is None or self.model=='default': 
            # use default transformer 
            self.model = default_transformer(
                head_size      = h_size,
                num_heads      = n_heads,
                ff_dim         = ff_,
                num_trans_blocks = n_blocks,
                mlp_units      = [mlp_u],
                mlp_dropout    = self.dropout,
                ff_dim1        = ff_1,
                ff_dim2        = ff_2
            )

        validate_keras_model(self.model)
        # Compile it
        opt = None
        if self.optimizer == "adam":
            opt = Adam(learning_rate=self.learning_rate)
        elif self.optimizer == "sgd":
            opt = SGD(learning_rate=self.learning_rate)
        else:
            # fallback
            opt =Adam(learning_rate=self.learning_rate)

        self.model.compile(loss="mse", optimizer=opt, metrics=["mae", "mape"])

        # Reshape the X, y if needed. For demonstration we assume X is
        # already (batch_size, 40, 1) or similar.
        # If the user needs further pre-processing, they'd do it outside
        # or do it here.
        if fit_params: 
            callbacks, fit_params = extract_callbacks_from(
                fit_params, return_fit_params=True
            )
        # update the epochs and batch_size if they are 
        # explicitly provided in fit_params. This avoid 
        # repeating the same param twice in keras fit method.
        self.epochs = fit_params.pop('epochs', self.epochs) 
        self.batch_size = fit_params.pop('batch_size', self.batch_size) 
        self.verbose= fit_params.pop('verbose', self.verbose) 
        
        # default callbacks
        default_cbs = [
           EarlyStopping(monitor='val_loss', patience=5,
                         mode='min', verbose=0),
           ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                            patience=2, min_lr=1e-5, verbose=0),
         ]
        if fit_params and callbacks: 
            # Then use the callbacks to replace the default callbacks 
            default_cbs= callbacks 
            
        # Merge user callbacks if provided
        if self.user_callbacks:
            # then take priority 
            default_cbs.extend(self.user_callbacks)

        # Fit the model
        self.model.fit(
            X, y,
            batch_size=self.batch_size or b_sz,
            epochs=n_ep,
            validation_split=0.3,
            callbacks= default_cbs,
            verbose=self.verbose
        )

        # Evaluate performance with MAPE or anything else
        # Let's do a quick prediction on X
        pred = self.model.predict(X, verbose=0)
        
        if isinstance (self.scoring, str):  
            scorer = get_scorer(self.scoring)
            score=scorer(y, pred) 
        else:
            score = self.scoring(y, pred) 
             
        return score

    def _compute_fitness(self, swarm_positions, X, y, fit_params=None):
        # Evaluate each particle's hyperparam set by building
        # a model, training, computing MAPE, etc.

        best_fit   = -999999
        best_param = None
        fitness_vals = []

        for idx in range(self.n_particles):
            param_vector = swarm_positions[idx]
            # If param_vector is less than 9 dims, adjust logic
            # accordingly. We assume it matches what build_and_train
            # expects.
            score = self._build_and_train_model(
                param_vector, X, y, fit_params=fit_params )
            fitness_vals.append(score)
            # track best
            if score > best_fit:
                best_fit    = score
                best_param  = param_vector

        return fitness_vals, best_fit, best_param

    def _update_positions(self, swarm_positions, gbest, pbest_positions):
        # Standard QPSO update. We assume alpha is constant.
        # Compute mbest
        n_feats = len(swarm_positions[0])
        mbest = [0.0]*n_feats
        total = np.zeros(n_feats)

        for i in range(self.n_particles):
            total += np.array(pbest_positions[i])
        for j in range(n_feats):
            mbest[j] = total[j] / self.n_particles

        # partial update for personal best
        for i in range(self.n_particles):
            a_ = random.uniform(0, 1)
            pbest_positions[i] = list(
                np.array(pbest_positions[i])*a_
                + np.array(gbest)*(1 - a_)
            )

        # QPSO positional update
        for i in range(self.n_particles):
            dist_to_mbest = []
            for j in range(n_feats):
                dist = abs(mbest[j] - swarm_positions[i][j])
                dist_to_mbest.append(dist)
            u_ = random.uniform(0, 1)
            if random.random() > 0.5:
                swarm_positions[i] = list(
                    np.array(pbest_positions[i])
                    + np.array([self.alpha * math.log(1/u_)*d
                                for d in dist_to_mbest])
                )
            else:
                swarm_positions[i] = list(
                    np.array(pbest_positions[i])
                    - np.array([self.alpha * math.log(1/u_)*d
                                for d in dist_to_mbest])
                )

        # Bound the updated positions in [min_bound, max_bound]
        # Re-scale each dim
        for j in range(n_feats):
            # gather the jth dimension of all particles
            dims = [p[j] for p in swarm_positions]
            dmax = max(dims)
            dmin = min(dims)
            for i in range(self.n_particles):
                if dmax != dmin:  # avoid zero-div
                    swarm_positions[i][j] = (
                        (swarm_positions[i][j] - dmin)/(dmax - dmin)*
                        (self.max_bound[j] - self.min_bound[j])
                        + self.min_bound[j]
                    )
                else:
                    swarm_positions[i][j] = self.min_bound[j]

        return swarm_positions

    def plot_results(self):
        check_is_fitted(self, attributes=['best_params_'])
        xs = np.arange(len(self.results_))+1
        plt.plot(xs, self.results_, '-o')
        plt.xlabel("Iterations")
        plt.ylabel("Fitness (MAPE or similar)")
        plt.title("QPSO Optimization")
        plt.show()

    def fit(self, X, y, **fit_params):
        # Main QPSO loop: init swarm, track pbest, gbest, do updates,
        # store best, etc.
        if self.max_bound is None: 
            self.max_bound  = [128, 6, 64, 64, 64, 4,100,200, 200]
        if self.min_bound is None: 
            self.min_value = [4, 2, 8, 8, 8, 2, 8, 25, 50]
            
        if (self.max_bound is None) or (self.min_bound is None):
            raise ValueError("max_bound and min_bound must be provided.")
        
        check_consistent_length(self.min_bound, self.max_bound ) 
        
        self.results_      = []
        self.best_fitness_ = -999999
        # init swarm
        swarm_positions    = self._initialize_swarm()
        # prepare pbest
        pbest_positions    = [row[:] for row in swarm_positions]
        pbest_fitness      = [-999999]*self.n_particles

        gbest             = [0]*self.n_features

        for iteration in range(self.max_iter):
            fit_vals, best_fit_iter, best_param_iter = (
                self._compute_fitness(
                    swarm_positions, X, y, fit_params=fit_params
                    )
            )

            # update pbest
            for i in range(self.n_particles):
                if fit_vals[i] > pbest_fitness[i]:
                    pbest_fitness[i]    = fit_vals[i]
                    pbest_positions[i]  = swarm_positions[i]

            # update gbest
            if best_fit_iter > self.best_fitness_:
                self.best_fitness_ = best_fit_iter
                gbest              = best_param_iter

            self.results_.append(self.best_fitness_)

            if self.verbose>0:
                print(f"Iter={iteration+1}, best fitness={self.best_fitness_}, "
                      f"params={gbest}")

            # check tolerance
            if iteration>0:
                improvement = abs(self.results_[-1]-self.results_[-2])
                if improvement < self.stopping_tol:
                    if self.verbose>0:
                        print("Converged early.")
                    break

            # update positions
            swarm_positions = self._update_positions(
                swarm_positions, gbest, pbest_positions
            )

        if self.verbose>1:
            self._plot_results()

        self.best_params_ = gbest
        return self

# For demonstration, we define a simple MAPE and a build_transformer
# function. In a real scenario, these would be imported from the user's
# code or a library.
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-4], kernel_size=1)(x)
    return x + res

def MAPE(true, pred):
    # Mean Absolute Percentage Error, returning
    # (1 - average error ratio).
    diff = np.abs(np.array(true) - np.array(pred))
    return (1 - np.mean(diff / true))

def default_transformer(
    head_size,
    num_heads,
    ff_dim,
    num_trans_blocks,
    mlp_units,
    mlp_dropout,
    ff_dim1,
    ff_dim2
):
    # Default model architecture
    # In real usage, adapt to
    # your data shapes and logic.
    inputs = Input(shape=(40, 1))  # dummy shape
    x = inputs

    # A simple 1D Convolution
    x_cnn = Conv1D(
        filters=ff_dim1, kernel_size=3, activation='relu'
    )(x)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
    x_cnn = LSTM(ff_dim2, return_sequences=False)(x_cnn)

    # A pseudo "transformer encoder" loop
    # for demonstration
    x_transform = x
    for _ in range(num_trans_blocks):
        # Insert  transformer logic, multi-head
        # attention, feed-forward, etc.
        x_transform=transformer_encoder(
            inputs, head_size, num_heads, ff_dim, mlp_dropout)
        x_transform = GlobalAveragePooling1D()(x_transform)

    merged = concatenate([x_cnn, x_transform])
    for dim in mlp_units:
        merged = Dense(dim, activation="relu")(merged)
        merged = Dropout(mlp_dropout)(merged)

    outputs = Dense(1, activation='relu')(merged)
    model = Model(inputs, outputs)
    return model