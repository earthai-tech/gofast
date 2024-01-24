# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import random
import numpy as np 
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator 
from sklearn.model_selection import cross_val_score
from sklearn.base import  clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from scipy.stats import norm

class SMBOSearchCV(BaseEstimator):
    r"""
    Sequential Model-Based Optimization (SMBO) for hyperparameter tuning 
    of estimators.
    
    SMBOSearchCV implements a Bayesian optimization strategy for tuning 
    hyperparameters. It builds a probabilistic model (Gaussian Process by default) 
    of the objective function and uses this model to select hyperparameters 
    to evaluate in the true objective function, balancing exploration and 
    exploitation.
    
    Mathematical Formulation
    ------------------------
    Sequential Model-Based Optimization (SMBO), particularly Bayesian optimization, 
    involves the iterative update of a surrogate model, typically a Gaussian Process,
    which approximates the true objective function. The surrogate model is updated 
    based on observations, and an acquisition function is used to guide the 
    search for new hyperparameters.
    
    Given a set of observations \( \{(X_i, y_i)\} \), where \( X_i \) represents
    hyperparameters and \( y_i \) their corresponding performance scores, 
    the surrogate model \( P(y|X) \) is updated to reflect this data. The 
    acquisition function \( A(X; P) \) is then optimized to select the next 
    set of hyperparameters to evaluate:
    
    \[ X_{\text{new}} = \underset{X}{\text{argmax}} \, A(X; P) \]
    
    Popular choices for the acquisition function \( A \) include Expected 
    Improvement (EI), Probability of Improvement (PI), and Upper Confidence 
    Bound (UCB). These functions aim to balance exploration
    (evaluating untested hyperparameters) and exploitation 
    (focusing on areas of the hyperparameter space known to yield 
     high performance):
    
    - Expected Improvement (EI): Maximizes the expected improvement over 
      the current best observation.
    - Probability of Improvement (PI): Maximizes the probability of improving 
      over the current best observation.
    - Upper Confidence Bound (UCB): Selects hyperparameters based on a 
    trade-off between their predicted mean performance and uncertainty.
    
    These strategies enable efficient navigation of the hyperparameter space, 
    leveraging the surrogate model to make informed decisions about which
    hyperparameters to evaluate next.
    
    Parameters
    ----------
    estimator : estimator object
        The machine learning estimator to be optimized.
    
    param_space : dict
        Dictionary with parameters names as keys and distributions or lists 
        as values, defining the search space for each hyperparameter.
    
    scoring : str, callable, list/tuple, dict or None, default=None
        Strategy to evaluate the performance of the cross-validated model on 
        the test set.
    
    cv : int, cross-validation generator or an iterable, default=3
        Determines the cross-validation splitting strategy.
    
    n_iter : int, default=50
        Number of iterations to run the optimization process.
    
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator and the optimization algorithm.
    
    Attributes
    ----------
    best_params_ : dict
        The best parameter setting found during the optimization process. 
        This is the set of  hyperparameters that yielded the highest 
        cross-validation score.
    
    best_score_ : float
        The highest cross-validation score achieved by any set of hyperparameters 
        during the  optimization process.
    
    best_estimator_ : estimator object
        The estimator fitted with the best_params_. This estimator is a 
        clone of the original estimator passed to SMBOSearchCV, trained on 
        the entire dataset.
    
    surrogate_model_ : GaussianProcessRegressor or similar
        The surrogate model used to approximate the objective function. 
        This model is updated iteratively as new observations 
        (hyperparameter sets and their scores) are acquired.
    
    cv_results_ : list of dicts
        A comprehensive record of the optimization process. Each dictionary 
        in the list contains information about the hyperparameters evaluated, 
        their cross-validation score, and other relevant details.
    
    Examples
    --------
    >>> from gofast.models.selection import SMBOSearchCV
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> X, y = load_iris(return_X_y=True)
    >>> param_space = {'C': [1, 10, 100], 'gamma': [0.001, 0.0001]}
    >>> search = SMBOSearchCV(estimator=SVC(), param_space=param_space)
    >>> search.fit(X, y)
    >>> print(search.best_params_)
    
    Notes
    -----
    SMBO is particularly useful for optimization problems with expensive evaluations,
    as it efficiently narrows down the search space using the surrogate model.
    
    References
    ----------
    - Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization 
       of Machine Learning Algorithms. Advances in Neural Information Processing Systems.
    """

    def __init__(
        self, 
        estimator, 
        param_space, 
        scoring=None, 
        cv=3, 
        n_iter=50, 
        random_state=None
        ):
        """
        Initialize the SMBOSearchCV with the given parameters.
    
        Parameters
        ----------
        estimator : estimator object
            The machine learning estimator to be optimized.
    
        param_space : dict
            Dictionary with parameters names as keys and distributions or 
            lists as values, defining the search space for each 
            hyperparameter.
    
        scoring : str, callable, list/tuple, dict or None, default=None
            Strategy to evaluate the performance of the cross-validated model 
            on the test set.
    
        cv : int, cross-validation generator or an iterable, default=3
            Determines the cross-validation splitting strategy.
    
        n_iter : int, default=50
            Number of iterations to run the optimization process.
    
        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the estimator and the optimization algorithm.
    
        Attributes
        ----------
        best_params_ : dict
            The best parameter setting found during the optimization process.
    
        best_score_ : float
            The score of the best_params_.
    
        best_estimator_ : estimator object
            Estimator fitted with the best_params_.
    
        surrogate_model_ : GaussianProcessRegressor or similar
            The surrogate model used for optimization.
    
        cv_results_ : list of dicts
            A record of each iteration's parameter set and score.
        """
        self.estimator = estimator
        self.param_space = param_space
        self.scoring = scoring
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None
        self.surrogate_model_ = None
        self.cv_results_ = []

    def fit(self, X, y):
        """
        Fit the SMBO to the dataset.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
    
        y : array-like of shape (n_samples,)
            Target values.
    
        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self.surrogate_model_ = GaussianProcessRegressor(
            kernel=Matern(), random_state=self.random_state)
        observed_X, observed_y = [], []
    
        for _ in range(self.n_iter):
            next_hyperparameters = self._optimize(observed_X, observed_y)
            observed_X.append(next_hyperparameters)
    
            score = self._evaluate_hyperparameters(next_hyperparameters, X, y)
            observed_y.append(score)
    
            self.cv_results_.append({'params': next_hyperparameters, 'score': score})
    
            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = next_hyperparameters
                self.best_estimator_ = clone(self.estimator).set_params(
                    **next_hyperparameters)
    
        return self
    
    def _optimize(self, observed_X, observed_y):
        """
        Update the surrogate model and select the next hyperparameters 
        based on the acquisition function.
    
        Parameters
        ----------
        observed_X : list of dicts
            Previously evaluated hyperparameters.
    
        observed_y : list
            Scores corresponding to the evaluated hyperparameters.
    
        Returns
        -------
        next_hyperparameters : dict
            The next set of hyperparameters to evaluate.
        """
        if observed_X:
            # Fit the surrogate model to the observed data
            X_train = self._transform_hyperparameters_to_array(observed_X)
            self.surrogate_model_.fit(X_train, observed_y)
    
        def acquisition_function(X):
            # Acquisition function (such as Expected Improvement)
            mean, std = self.surrogate_model_.predict(X, return_std=True)
            best_y = max(observed_y) if observed_y else 0
            z = (mean - best_y) / std
            return (mean - best_y) * norm.cdf(z) + std * norm.pdf(z)
    
        # Optimize the acquisition function to find the next set of hyperparameters
        result = minimize(lambda X: -acquisition_function(X.reshape(1, -1)),
                          x0=self._random_hyperparameters_array(),
                          bounds=self._hyperparameters_bounds())
    
        next_hyperparameters = self._transform_array_to_hyperparameters(result.x)
        return next_hyperparameters

    def _transform_hyperparameters_to_array(self, hyperparameters_list):
        """
        Convert a list of hyperparameter dicts to a numpy array for model fitting.
    
        Parameters
        ----------
        hyperparameters_list : list of dicts
            List of hyperparameter dictionaries.
    
        Returns
        -------
        np.array
            Numpy array representing the hyperparameters.
        """
        # Assuming param_space is defined with continuous parameters
        return np.array([[hyperparams[param] for param in self.param_space.keys()]
                         for hyperparams in hyperparameters_list])

    def _random_hyperparameters_array(self):
        """
        Generate a random set of hyperparameters as a numpy array.
    
        Returns
        -------
        np.array
            A numpy array representing a random set of hyperparameters.
        """
        random_hyperparams = []
        for param, values in self.param_space.items():
            if isinstance(values, list):
                random_hyperparams.append(np.random.choice(values))
            elif isinstance(values, tuple) and len(values) == 2:
                random_hyperparams.append(np.random.uniform(values[0], values[1]))
        return np.array(random_hyperparams)

    def _hyperparameters_bounds(self):
        """
        Return the bounds of the hyperparameters for optimization.
    
        Returns
        -------
        list of tuples
            Bounds for each hyperparameter.
        """
        bounds = []
        for param, values in self.param_space.items():
            if isinstance(values, tuple):
                bounds.append(values)  # For continuous parameters
            else:
                # For categorical parameters, bounds are not well-defined;
                # use a workaround or handle separately
                bounds.append((min(values), max(values)))
        return bounds

    def _transform_array_to_hyperparameters(self, array):
        """
        Convert a numpy array back to a hyperparameters dict.
    
        Parameters
        ----------
        array : np.array
            Numpy array representing hyperparameters.
    
        Returns
        -------
        dict
            Dictionary of hyperparameters.
        """
        hyperparameters = {}
        for i, param in enumerate(self.param_space.keys()):
            hyperparameters[param] = array[i]
        return hyperparameters

    def _evaluate_hyperparameters(self, hyperparameters, X, y):
        """
        Evaluate a set of hyperparameters and return the score.
    
        Parameters
        ----------
        hyperparameters : dict
            Hyperparameters of the estimator.
    
        X : array-like of shape (n_samples, n_features)
            Training vectors.
    
        y : array-like of shape (n_samples,)
            Target values.
    
        Returns
        -------
        score : float
            The mean cross-validation score of the estimator with the given hyperparameters.
        """
        estimator = clone(self.estimator).set_params(**hyperparameters)
        scores = cross_val_score(estimator, X, y, scoring=self.scoring, cv=self.cv)
        return np.mean(scores)


class PSOSearchCV(BaseEstimator):
    """
    Particle Swarm Optimization for hyperparameter tuning of estimators.

    PSOSearchCV implements a Particle Swarm Optimization algorithm to find the
    optimal hyperparameters for a given estimator. PSO is a computational method
    that iteratively improves a set of candidate solutions concerning a measure
    of quality, inspired by social behaviors such as bird flocking or fish schooling.

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn-compatible estimator. The object used to fit the data.

    param_space : dict
        Dictionary with parameters names (`str`) as keys and lists or tuples as values,
        specifying the hyperparameter search space.

    scoring : str, callable, list/tuple, dict or None, default=None
        Strategy to evaluate the performance of the cross-validated model on the test set.

    cv : int, cross-validation generator or an iterable, default=3
        Determines the cross-validation splitting strategy.

    n_particles : int, default=30
        Number of particles in the swarm.

    max_iter : int, default=100
        Maximum number of iterations for the optimization process.

    inertia_weight : float, default=0.9
        Inertia weight to balance the global and local exploration.

    cognitive_coeff : float, default=2.0
        Cognitive coefficient to guide particles towards their best known position.

    social_coeff : float, default=2.0
        Social coefficient to guide particles towards the swarm's best known position.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the algorithm.

    Attributes
    ----------
    best_params_ : dict
        The best parameter setting found during the optimization process.

    best_score_ : float
        The score of the best_params_.

    best_estimator_ : estimator object
        Estimator fitted with the best_params_.

    cv_results_ : list of dicts
        A record of each particle's position and score at each iteration.

    Mathematical Formulation
    ------------------------
    Each particle in the swarm represents a candidate solution (a set of hyperparameters).
    The position and velocity of each particle are updated as follows:

    .. math::
        v_{id}^{(t+1)} = w \cdot v_{id}^{(t)} + c_1 \cdot r_1 \cdot (p_{id} - x_{id}^{(t)}) 
                        + c_2 \cdot r_2 \cdot (g_d - x_{id}^{(t)})

        x_{id}^{(t+1)} = x_{id}^{(t)} + v_{id}^{(t+1)}

    where:
    - \(v_{id}^{(t)}\) is the velocity of particle i in dimension d at iteration t.
    - \(x_{id}^{(t)}\) is the position of particle i in dimension d at iteration t.
    - \(w\) is the inertia weight.
    - \(c_1, c_2\) are cognitive and social coefficients, respectively.
    - \(r_1, r_2\) are random numbers uniformly distributed in [0, 1].
    - \(p_{id}\) is the best known position of particle i in dimension d.
    - \(g_d\) is the best known position among all particles in dimension d.

    The algorithm iterates this process, guiding the swarm towards the optimal solution.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> X, y = load_iris(return_X_y=True)
    >>> param_space = {'C': [1, 10, 100], 'gamma': [0.001, 0.0001]}
    >>> search = PSOSearchCV(estimator=SVC(), param_space=param_space)
    >>> search.fit(X, y)
    >>> print(search.best_params_)

    Notes
    -----
    PSO simulates the behaviors of bird flocking, where each particle (candidate solution)
    moves in the search space with a velocity that is dynamically adjusted according to 
    its own and its neighbors' experiences.

    References
    ----------
    - Kennedy, J., & Eberhart, R. (1995). Particle Swarm Optimization. 
      Proceedings of ICNN'95 - International Conference on Neural Networks.
    """
    def __init__(self, estimator, param_space, scoring=None, cv=3, 
                 n_particles=30, max_iter=100, inertia_weight=0.9, 
                 cognitive_coeff=2.0, social_coeff=2.0, random_state=None):
        """
        Initialize the PSOSearchCV with the given parameters.

        Parameters
        ----------
        estimator : estimator object
            The machine learning estimator to be optimized.

        param_space : dict
            Dictionary with parameters names as keys and lists or tuples as values.
            Each entry specifies the hyperparameter search space.

        scoring : str, callable, list, tuple, dict, or None, default=None
            A string or a scorer callable object / function with signature scorer(estimator, X, y).
            If None, the estimator's default scorer is used.

        cv : int, cross-validation generator or an iterable, default=3
            Determines the cross-validation splitting strategy.

        n_particles : int, default=30
            Number of particles in the swarm.

        max_iter : int, default=100
            Maximum number of iterations for the optimization process.

        inertia_weight : float, default=0.9
            Inertia weight that influences the particle's velocity.

        cognitive_coeff : float, default=2.0
            Cognitive coefficient to guide particles towards their own best known position.

        social_coeff : float, default=2.0
            Social coefficient to guide particles towards the swarm's best known position.

        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the estimator and the algorithm.
        """
        self.estimator = estimator
        self.param_space = param_space
        self.scoring = scoring
        self.cv = cv
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.random_state = random_state

    def fit(self, X, y):
        """
        Run the Particle Swarm Optimization algorithm for hyperparameter optimization.

        Iteratively moves a swarm of particles in the hyperparameter space to find the
        optimal settings for the given estimator. Each particle represents a set of 
        hyperparameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.

        Attributes
        ----------
        best_params_ : dict
            The best parameter setting found during the optimization process.

        best_score_ : float
            The score of the best_params_.

        best_estimator_ : estimator object
            Estimator fitted with the best_params_.

        cv_results_ : list of dicts
            A record of each particle's position and score at each iteration.
        """
        random.seed(self.random_state)
        self.cv_results_ = []
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None
        # Initialize particles
        particles = self._initialize_particles()
        global_best = None
        global_best_score = -np.inf

        for iteration in range(self.max_iter):
            for particle in particles:
                # Evaluate each particle
                score = self._evaluate_particle(particle, X, y)

                # Update particle's best known position
                if score > particle['best_score']:
                    particle['best_score'] = score
                    particle['best_position'] = particle['position'].copy()

                # Update global best
                if score > global_best_score:
                    global_best_score = score
                    global_best = particle['position'].copy()

                # Record results
                self.cv_results_.append({'params': particle['position'],
                                         'score': score, 'iteration': iteration})
            # Move particles
            self._move_particles(particles, global_best)

            # Update best estimator
            if global_best_score > self.best_score_:
                self.best_score_ = global_best_score
                self.best_params_ = global_best
                self.best_estimator_ = clone(self.estimator).set_params(**global_best)

        return self

    def _initialize_particles(self):
        """
        Initialize particles with random positions and velocities.
    
        Returns
        -------
        particles : list of dicts
            A list of particles, where each particle is represented as a dict containing 
            'position', 'velocity', 'best_position', and 'best_score'.
        """
        particles = []
        for _ in range(self.n_particles):
            particle = {'position': self._random_hyperparameters(),
                        'velocity': self._random_velocity(),
                        'best_position': None,
                        'best_score': -np.inf}
            particles.append(particle)
        return particles

    def _random_hyperparameters(self):
        """
        Randomly generate hyperparameters from the defined search space.
    
        Returns
        -------
        hyperparameters : dict
            A dictionary of hyperparameters where each key-value pair is a parameter 
            and its randomly selected value.
        """
        hyperparameters = {}
        for param, values in self.param_space.items():
            if isinstance(values, list):
                # Categorical hyperparameters
                hyperparameters[param] = random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                # Continuous hyperparameters (defined by a range)
                hyperparameters[param] = random.uniform(values[0], values[1])
            else:
                raise ValueError(f"Invalid parameter space for {param}.")
        return hyperparameters

    def _random_velocity(self):
        """
        Randomly generate initial velocities for the particles.
    
        The velocity of a particle represents the rate of change of its position 
        (hyperparameters). These are typically small values or zeros.
    
        Returns
        -------
        velocity : dict
            A dictionary representing the velocity of a particle. Each key-value pair 
            corresponds to a parameter and its velocity.
        """
        velocity = {}
        for param in self.param_space.keys():
            # Assuming a small range for velocity initialization, e.g., [-0.1, 0.1]
            velocity[param] = random.uniform(-0.1, 0.1)
        return velocity

    def _evaluate_particle(self, particle, X, y):
        """
        Evaluate the fitness of a particle's position.
    
        Parameters
        ----------
        particle : dict
            A particle representing a set of hyperparameters.
    
        X : array-like of shape (n_samples, n_features)
            Training vectors.
    
        y : array-like of shape (n_samples,)
            Target values.
    
        Returns
        -------
        score : float
            The fitness score of the particle's position.
        """
        estimator = clone(self.estimator).set_params(**particle['position'])
        score = np.mean(cross_val_score(estimator, X, y, cv=self.cv, scoring=self.scoring))
        return score

    def _move_particles(self, particles, global_best):
        """
        Update particles' positions and velocities based on PSO rules.
    
        Parameters
        ----------
        particles : list of dicts
            The current swarm of particles.
    
        global_best : dict
            The global best position found by the swarm.
        """
        for particle in particles:
            for param in particle['position']:
                # Update velocity
                r1, r2 = random.random(), random.random()
                cognitive_velocity = self.cognitive_coeff * r1 * (particle['best_position'][param] - particle['position'][param])
                social_velocity = self.social_coeff * r2 * (global_best[param] - particle['position'][param])
                particle['velocity'][param] = (self.inertia_weight * particle['velocity'][param] +
                                               cognitive_velocity + social_velocity)
    
                # Update position
                particle['position'][param] += particle['velocity'][param]
    
                # Ensure the new position is within bounds
                if isinstance(self.param_space[param], list):
                    # For categorical parameters, use nearest valid value
                    nearest_value = min(self.param_space[param], key=lambda x: abs(x - particle['position'][param]))
                    particle['position'][param] = nearest_value
                elif isinstance(self.param_space[param], tuple):
                    # For continuous parameters, clip to the range
                    particle['position'][param] = np.clip(particle['position'][param], self.param_space[param][0], self.param_space[param][1])

class AnnealingSearchCV(BaseEstimator):
    """
    Simulated annealing for hyperparameter optimization of estimators.

    AnnealingSearchCV implements a simulated annealing algorithm for tuning 
    hyperparameters of a given estimator. It is a probabilistic technique to 
    approximate the global optimum of the hyperparameter optimization problem. 
    The method is effective at avoiding local minima by allowing uphill moves 
    (worse solutions) probabilistically, with this ability decreasing as the 
    'temperature' lowers over iterations.

    Mathematical Formulation
    ------------------------
    The simulated annealing algorithm is inspired by the physical process of 
    annealing in metallurgy. The algorithm starts at a high 'temperature' and 
    gradually cools down. At each step, a new solution is randomly generated. 
    If the new solution is better, it is accepted; if it is worse, it may be 
    accepted with a probability that decreases with the temperature and the 
    difference in solution quality.

    .. math::
        P(\text{{accept}}) = \exp\left(-\frac{{\Delta E}}{{kT}}\right)

    where:
    - \Delta E is the change in energy (score difference between current and new solution).
    - T is the current temperature.
    - k is a constant that controls the probability of accepting worse solutions.

    As the temperature decreases, the probability of accepting worse solutions 
    reduces, allowing the algorithm to focus on exploitation over exploration.
    
    Parameters
    ----------
    estimator : estimator object
        A scikit-learn-compatible estimator. The object used to fit the data.

    param_space : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter 
        settings to try as values, specifying the hyperparameter search space.

    scoring : str, callable, list/tuple, dict or None, default=None
        Strategy to evaluate the performance of the cross-validated model 
        on the test set. 
        Can be a string, a callable, or None.

    cv : int, cross-validation generator or an iterable, default=3
        Determines the cross-validation splitting strategy. Could be an integer,
        a CV splitter, or an iterable.

    initial_temp : float, default=1.0
        The initial temperature for annealing process.

    alpha : float, default=0.9
        The rate at which the temperature decreases in each iteration.

    max_iter : int, default=100
        Maximum number of iterations for the optimization process.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.

    Attributes
    ----------
    best_params_ : dict
        The best parameter setting found during the annealing process. It is the set of 
        hyperparameters that achieved the highest score.

    best_score_ : float
        The highest score achieved by any set of hyperparameters during the annealing process. 
        It is the mean cross-validation score of the best_params_.

    best_estimator_ : estimator object
        The estimator fitted with the best_params_. This estimator is a clone of the 
        original estimator passed to AnnealingSearchCV and is fitted to the data 
        passed to the fit method.

    cv_results_ : list of dicts
        A detailed record of the annealing process. Each dictionary in the list contains 
        the following keys:
            - 'params': The hyperparameters set evaluated in that iteration.
            - 'score': The mean cross-validation score of those hyperparameters.
            - 'iteration': The iteration number at which these hyperparameters were evaluated.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> X, y = load_iris(return_X_y=True)
    >>> param_space = {'C': [1, 10, 100], 'gamma': [0.001, 0.0001]}
    >>> search = AnnealingSearchCV(estimator=SVC(), param_space=param_space)
    >>> search.fit(X, y)
    >>> print(search.best_params_)

    Notes
    -----
    Simulated annealing is analogous to the physical process of heating a material 
    and then slowly lowering the temperature to decrease defects. In the context 
    of hyperparameter tuning, this allows exploration of the search space with 
    decreasing probability of accepting worse solutions over time.
    """
    def __init__(
        self, 
        estimator, 
        param_space, 
        scoring=None, 
        cv=3, 
        initial_temp=1.0, 
        alpha=0.9, 
        max_iter=100, 
        random_state=None
        ):
        """
        Initialize the AnnealingSearchCV with the given parameters.

        Parameters
        ----------
        estimator : estimator object
            A scikit-learn-compatible estimator.

        param_space : dict
            Dictionary with parameter names as keys and lists of parameter 
            settings to try as values.

        scoring : str, callable, list, tuple, dict or None, default=None
            Strategy to evaluate the performance of the cross-validated model 
            on the test set.

        cv : int, cross-validation generator or iterable, default=3
            Determines the cross-validation splitting strategy.

        initial_temp : float, default=1.0
            Initial temperature for annealing.

        alpha : float, default=0.9
            Temperature reduction factor.

        max_iter : int, default=100
            Maximum number of iterations for the annealing process.

        random_state : int, RandomState instance or None, default=None
            Random state for reproducibility.
        """
        self.estimator = estimator
        self.param_space = param_space
        self.scoring = scoring
        self.cv = cv
        self.initial_temp = initial_temp
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        Run the simulated annealing algorithm for hyperparameter optimization.

        This method iteratively explores the hyperparameter space by adjusting
        hyperparameters and evaluating their performance, while gradually reducing
        the temperature to minimize the objective function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.

        Attributes
        ----------
        cv_results_ : list of dicts
            A record of the hyperparameter set and corresponding 
            score at each iteration.
        """
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None
        random.seed(self.random_state)
        temperature = self.initial_temp
        self.cv_results_ = []

        current_params = self._random_hyperparameters()
        current_score = self._evaluate_fitness(current_params, X, y)

        for iteration in range(self.max_iter):
            next_params = self._random_hyperparameters()
            next_score = self._evaluate_fitness(next_params, X, y)

            # Record the current state before deciding whether to move to the next state
            self.cv_results_.append({'params': current_params, 
                                     'score': current_score, 'iteration': iteration})

            if self._acceptance_criterion(current_score, next_score, temperature):
                current_params, current_score = next_params, next_score

            if next_score > self.best_score_:
                self.best_score_ = next_score
                self.best_params_ = next_params
                self.best_estimator_ = clone(self.estimator).set_params(**next_params)

            temperature *= self.alpha

        return self

    def _random_hyperparameters(self):
        """
        Randomly select hyperparameters from the param_space.
    
        Returns
        -------
        hyperparameters : dict
            A dictionary containing a set of randomly selected hyperparameters.
        """
        hyperparameters = {}
        for param, values in self.param_space.items():
            if isinstance(values, list):
                # For categorical hyperparameters
                hyperparameters[param] = random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                # For continuous hyperparameters (defined by a range)
                hyperparameters[param] = random.uniform(values[0], values[1])
            else:
                raise ValueError(f"Invalid parameter space for {param}.")
        return hyperparameters

    def _evaluate_fitness(self, hyperparameters, X, y):
        """
        Evaluate the fitness of the given hyperparameters.
    
        Parameters
        ----------
        hyperparameters : dict
            Hyperparameters of the estimator.
    
        X : array-like of shape (n_samples, n_features)
            Training vectors.
    
        y : array-like of shape (n_samples,)
            Target values.
    
        Returns
        -------
        score : float
            The mean cross-validation score of the estimator with the given hyperparameters.
        """
        estimator = clone(self.estimator).set_params(**hyperparameters)
        score = np.mean(cross_val_score(estimator, X, y, cv=self.cv, scoring=self.scoring))
        return score

    def _acceptance_criterion(self, current_score, next_score, temperature):
        """
        Determine whether to accept the new solution based on simulated annealing.
    
        Parameters
        ----------
        current_score : float
            The score of the current solution.
    
        next_score : float
            The score of the new solution.
    
        temperature : float
            The current temperature in simulated annealing.
    
        Returns
        -------
        accept : bool
            True if the new solution is accepted, False otherwise.
        """
        if next_score > current_score:
            return True
        else:
            # Probability of accepting worse solution depends on the temperature
            # and difference between scores
            accept_prob = np.exp((next_score - current_score) / temperature)
            return accept_prob > random.random()

class EvolutionarySearchCV(BaseEstimator):
    """
    Evolutionary algorithm-based hyperparameter optimization for estimators.

    EvolutionarySearchCV implements an evolutionary algorithm for tuning hyperparameters 
    of a given estimator. Evolutionary algorithms mimic natural evolutionary processes 
    such as mutation, recombination, and selection to iteratively improve solutions 
    (parameter sets) to an optimization problem, particularly useful for complex and 
    non-linear objective functions.

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn-compatible estimator. The object used to fit the data.

    param_space : dict
        Dictionary with parameter names (`str`) as keys and lists of possible 
        values as values,specifying the hyperparameter search space.

    scoring : str, callable, list/tuple, dict or None, default=None
        Strategy to evaluate the performance of the cross-validated model 
        on the test set. Can be a string, a callable, or None.

    cv : int, cross-validation generator or iterable, default=3
        Determines the cross-validation splitting strategy. Could be an integer,
        a CV splitter, 
        or an iterable.

    n_population : int, default=10
        Number of individuals (parameter sets) in each generation.

    n_generations : int, default=10
        Number of generations for the evolutionary algorithm.

    mutation_rate : float, default=0.1
        The probability of mutation for each parameter in an individual.

    crossover_rate : float, default=0.5
        The probability of crossover between pairs of individuals.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the algorithm. Used for reproducible results.

    Attributes
    ----------
    best_params_ : dict
        The parameter setting that yielded the best results on the holdout data.

    best_score_ : float
        The score of the best_params_ on the holdout data.

    best_estimator_ : estimator object
       Estimator fitted with the best_params_.

   cv_results_ : list of dicts
       A list containing the scores and hyperparameters of each generation.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> X, y = load_iris(return_X_y=True)
    >>> param_space = {'C': [1, 10, 100], 'gamma': [0.001, 0.0001]}
    >>> search = EvolutionarySearchCV(estimator=SVC(), param_space=param_space)
    >>> search.fit(X, y)
    >>> print(search.best_params_)

    Notes
    -----
    Evolutionary algorithms are particularly suited for optimization problems with complex 
    landscapes, as they can escape local optima and handle a diverse range of parameter types. 
    These algorithms can be computationally intensive but are highly parallelizable.
    """
    def __init__(self, estimator, param_space, scoring=None, cv=3, 
                 n_population=10, n_generations=10, mutation_rate=0.1, 
                 crossover_rate=0.5, random_state=None):
        self.estimator=estimator 
        self.param_space=param_space 
        self.scoring=scoring 
        self.cv=cv
        self.n_population=n_population 
        self.n_generations=n_generations 
        self.mutation_rate=mutation_rate 
        self.crossover_rate=crossover_rate 
        self.random_state=random_state  
        
    def fit(self, X, y):
        """
        Run the evolutionary algorithm for hyperparameter optimization.

        This method applies an evolutionary algorithm to optimize the hyperparameters
        of the given estimator. It involves processes like selection, mutation,
        and crossover to evolve a population of hyperparameter sets towards 
        better performance.

        The implementation parallelizes the fitness evaluation to enhance 
        computational efficiency,especially beneficial for large populations 
        or complex models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of self.

        Attributes
        ----------
        best_params_ : dict
            The best parameter setting found.

        best_score_ : float
            The score of the best_params_.

        best_estimator_ : estimator object
            Estimator fitted with the best_params_.

        cv_results_ : list of dicts
            A list containing the scores and hyperparameters of each generation.
        """
        self.cv_results_ = []
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None

        # Initialize population
        population = self._initialize_population()

        for generation in range(self.n_generations):
            # # Evaluate fitness
            # fitness_scores = self._evaluate_fitness(population, X, y)
            # Parallelize fitness evaluation for computational efficiency
            fitness_scores = Parallel(n_jobs=-1)(
                delayed(self._evaluate_individual_fitness)(ind, X, y) for ind in population)
            # Record results
            self._record_results(population, fitness_scores)

            # Selection, Crossover, and Mutation to create next generation
            population = self._evolve(population, fitness_scores)

        return self

    def _evaluate_individual_fitness(self, individual, X, y):
        """
        Evaluate the fitness of an individual hyperparameter set.

        Parameters
        ----------
        individual : dict
            An individual's hyperparameters.

        X : array-like of shape (n_samples, n_features)
            Training vectors.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        fitness_score : float
            The fitness score of the individual.
        """
        estimator = clone(self.estimator).set_params(**individual)
        score = np.mean(cross_val_score(estimator, X, y, cv=self.cv, scoring=self.scoring))
        return score

    def _initialize_population(self):
        """
        Initialize the population for the evolutionary algorithm.

        Creates an initial population of individuals, where each individual 
        is a set of hyperparameters randomly chosen from the specified 
        parameter space.

        Returns
        -------
        population : list of dicts
            A list where each dict represents an individual's hyperparameters.
        """
        population = []
        for _ in range(self.n_population):
            individual = {}
            for param, values in self.param_space.items():
                if isinstance(values, list):
                    individual[param] = np.random.choice(values)
                elif isinstance(values, tuple) and len(values) == 2:  # Assuming continuous range
                    individual[param] = np.random.uniform(values[0], values[1])
                else:
                    raise ValueError(f"Invalid parameter space for {param}.")
            population.append(individual)
        return population

    def _evaluate_fitness(self, population, X, y):
        """
        Evaluate the fitness of each individual in the population.

        The fitness of an individual is determined by the cross-validation score
        of the model trained with its hyperparameters.

        Parameters
        ----------
        population : list of dicts
            The current population of individuals, where each individual is a 
            dict of hyperparameters.

        X : array-like of shape (n_samples, n_features)
            Training vectors.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        fitness_scores : list
            A list containing the fitness score of each individual in the population.
        """
        fitness_scores = []
        for individual in population:
            estimator = clone(self.estimator).set_params(**individual)
            score = np.mean(cross_val_score(estimator, X, y, cv=self.cv, scoring=self.scoring))
            fitness_scores.append(score)
        return fitness_scores

    def _evolve(self, population, fitness_scores):
        """
        Evolve the population using genetic operations.

        This method applies selection, crossover, and mutation to create a new
        generation of individuals.

        Parameters
        ----------
        population : list of dicts
            The current population of individuals.

        fitness_scores : list
            The fitness scores for each individual in the population.

        Returns
        -------
        new_population : list of dicts
            The new generation of individuals.
        """
        new_population = []

        # Selection: Implement your selection logic here (e.g., tournament selection)
        selected_individuals = self._select(population, fitness_scores)

        # Crossover and Mutation: Pair selected individuals and apply crossover and mutation
        while len(new_population) < len(population):
            parent1, parent2 = self._select_parents(selected_individuals)
            child1, child2 = self._crossover(parent1, parent2)

            # Apply mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)

            new_population.extend([child1, child2])

        return new_population[:len(population)]
    
    def _select_parents(self, selected_individuals):
        """
        Randomly select two parents from the selected individuals.
    
        Parameters
        ----------
        selected_individuals : list of dicts
            The individuals selected for breeding.
    
        Returns
        -------
        parent1, parent2 : tuple of dicts
            Two randomly chosen parents from the selected individuals.
        """
        parent1 = random.choice(selected_individuals)
        parent2 = random.choice(selected_individuals)
        while parent1 == parent2:
            parent2 = random.choice(selected_individuals)
        return parent1, parent2


    def _crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to produce offspring.
    
        Parameters
        ----------
        parent1 : dict
            The first parent individual's hyperparameters.
        parent2 : dict
            The second parent individual's hyperparameters.
    
        Returns
        -------
        offspring1, offspring2 : tuple of dicts
            Two offspring produced by crossover.
        """
        offspring1, offspring2 = parent1.copy(), parent2.copy()
        for param in parent1:
            if random.random() < self.crossover_rate:
                offspring1[param], offspring2[param] = offspring2[param], offspring1[param]
        return offspring1, offspring2

    def _mutate(self, individual):
        """
        Mutate an individual's hyperparameters.
    
        Parameters
        ----------
        individual : dict
            The individual's hyperparameters.
    
        Returns
        -------
        mutated_individual : dict
            The mutated individual.
        """
        mutated_individual = individual.copy()
        for param, values in self.param_space.items():
            if random.random() < self.mutation_rate:
                if isinstance(values, list):
                    mutated_individual[param] = random.choice(values)
                elif isinstance(values, tuple) and len(values) == 2:
                    mutated_individual[param] = random.uniform(values[0], values[1])
                else:
                    raise ValueError(f"Invalid parameter space for {param}.")
        return mutated_individual

    def _record_results(self, population, fitness_scores):
        """
        Record the results of the current generation.

        This method updates the `cv_results_` attribute with the hyperparameters
        and fitness scores of each individual in the current population.

        Parameters
        ----------
        population : list of dicts
            The current population of individuals, each represented as a dict of hyperparameters.

        fitness_scores : list
            The fitness scores corresponding to each individual in the population.
        """
        for individual, score in zip(population, fitness_scores):
            self.cv_results_.append({'params': individual, 'score': score})

            # Update the best parameters and score if the current individual is better
            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = individual
                self.best_estimator_ = clone(self.estimator).set_params(**individual)


class GradientBasedSearchCV(BaseEstimator):
    """
    Gradient-based hyperparameter optimization for estimators.

    This class implements a gradient-based optimization method for tuning 
    hyperparameters of a given estimator. It is particularly suited for 
    continuous and differentiable hyperparameters, employing gradient information 
    to iteratively adjust these parameters to minimize a defined loss function. 
    The optimization process is akin to gradient descent, but adapted for 
    hyperparameter optimization.

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn-compatible estimator. The object used to fit the data.

    param_space : dict
        Dictionary with parameter names (`str`) as keys and tuples representing
        continuous ranges as values, specifying the hyperparameter search space.

    scoring : str, callable, list/tuple, dict or None, default=None
        Strategy to evaluate the performance of the cross-validated model on 
        the test set. Can be a string, a callable, or None.

    cv : int, cross-validation generator or iterable, default=3
        Determines the cross-validation splitting strategy. Could be an integer,
        a CV splitter, or an iterable.

    max_iter : int, default=100
        Maximum number of iterations for the optimization process.

    learning_rate : float, default=0.01
        The step size (learning rate) for updating hyperparameters.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. Used for reproducible results.

    Attributes
    ----------
    best_params_ : dict
        The parameter setting that yielded the best results on the holdout data.

    best_score_ : float
        The score of the best_params_ on the holdout data.

    convergence_history_ : list of tuples
        A history of the optimization process. Each tuple contains a parameter set 
        and its corresponding score at each iteration.

    best_estimator_ : estimator object
        The estimator fitted with the best found parameters after the optimization
        process.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVR
    >>> X, y = load_iris(return_X_y=True)
    >>> param_space = {'C': (0.1, 10), 'gamma': (0.001, 0.1)}
    >>> search = GradientBasedSearchCV(estimator=SVR(), param_space=param_space)
    >>> search.fit(X, y)
    >>> print(search.best_params_)

    Notes
    -----
    The gradient-based optimization in hyperparameter space can be formulated as:

    .. math::
        \Theta_{\text{new}} = \Theta_{\text{old}} - \eta \cdot \nabla_\Theta J(\Theta)

    Where:
    - \Theta represents the hyperparameters.
    - \eta is the learning rate.
    - J(\Theta) is the loss function, typically the negative cross-validation score.
    - \nabla_\Theta J(\Theta) is the gradient of the loss function with respect to 
      the hyperparameters.

    This method requires an approximation of the gradient in the hyperparameter space, 
    as direct computation may often be infeasible. It is most effective for problems 
    where hyperparameters are continuous and the gradient can be sensibly approximated.

    References
    ----------
    - Bengio, Y., 2000. Gradient-based optimization of hyperparameters.
    - Rasmussen, C.E., Williams, C.K.I., 2006. Gaussian Processes for Machine Learning.
    """
    def __init__(self, estimator, param_space, scoring=None, cv=3, 
                 max_iter=100, learning_rate=0.01, random_state=None, 
                 n_jobs=1):
        self.estimator = estimator
        self.param_space = param_space
        self.scoring = scoring
        self.cv = cv
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_jobs=n_jobs 

    def fit(self, X, y):
        """
        Run gradient-based optimization to tune hyperparameters.

        This method employs gradient approximation techniques to iteratively 
        adjust and optimize hyperparameters of the given estimator based on 
        cross-validation scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of self.

        Attributes
        ----------
        cv_results_ : dict
            A dictionary containing cross-validation results.

        convergence_history_ : list
            List of tuples containing hyperparameters and corresponding scores 
            at each iteration.

        best_params_ : dict
            The parameters setting that gave the best results on the hold out data.

        best_score_ : float
            The best cross-validation score achieved.

        best_estimator_ : estimator object
            Estimator with the best found parameters.
        """
        self.cv_results_ = {'params': [], 'mean_test_score': [], 'std_test_score': []}
        self.convergence_history_ = []
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None

        # Initialize hyperparameters
        current_params = {param: np.mean(bounds) for param, bounds in self.param_space.items()}

        for iteration in range(self.max_iter):
            gradients = {}
            # Approximate gradient for each hyperparameter
            for param in self.param_space:
                # Compute gradients
                original_value = current_params[param]
                delta = original_value * 0.01  # Small change in the hyperparameter

                # Compute score for increased and decreased hyperparameter values
                increased_params = current_params.copy()
                decreased_params = current_params.copy()
                increased_params[param] = original_value + delta
                decreased_params[param] = original_value - delta

                increased_score = self._compute_cv_score(X, y, increased_params)
                decreased_score = self._compute_cv_score(X, y, decreased_params)

                # Finite difference approximation of gradient
                gradients[param] = (increased_score['mean'] - decreased_score['mean']) / (2 * delta)

                # Record results
                self._record_cv_results(increased_params, increased_score)
                self._record_cv_results(decreased_params, decreased_score)
                
            # Update hyperparameters
            # Compute scores 
            for param in self.param_space:
                current_params[param] += self.learning_rate * gradients[param]

            # Evaluate score with updated hyperparameters
            current_score = self._compute_cv_score(X, y, current_params)
            # Record the results
            self.cv_results_['params'].append(current_params.copy())
            self.cv_results_['mean_test_score'].append(current_score['mean'])
            self.cv_results_['std_test_score'].append(current_score['std'])
            self.convergence_history_.append((current_params.copy(), current_score['mean']))

            # Update best parameters and estimator
            if current_score['mean'] > self.best_score_:
                self.best_score_ = current_score['mean']
                self.best_params_ = current_params.copy()
                self.best_estimator_ = clone(self.estimator).set_params(**current_params)

        return self

    def _compute_cv_score(self, X, y, params):
        """
        Compute the cross-validation score for a given set of hyperparameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.

        y : array-like of shape (n_samples,)
            Target values.

        params : dict
            Hyperparameters of the estimator.

        Returns
        -------
        scores : dict
            Dictionary containing the mean and standard deviation of the 
            cross-validation scores.
        """
        estimator = clone(self.estimator).set_params(**params)
        scores = cross_val_score(estimator, X, y, cv=self.cv, scoring=self.scoring)
        return {'mean': np.mean(scores), 'std': np.std(scores)}

    def _record_cv_results(self, params, scores):
        """
        Record the cross-validation results for a set of hyperparameters.

        Parameters
        ----------
        params : dict
            Hyperparameters of the estimator.

        scores : dict
            Dictionary containing the mean and standard deviation of the 
            cross-validation scores.
        """
        self.cv_results_['params'].append(params)
        self.cv_results_['mean_test_score'].append(scores['mean'])
        self.cv_results_['std_test_score'].append(scores['std'])
        
class GeneticSearchCV(BaseEstimator):
    r"""
    A genetic algorithm-based hyperparameter optimization for estimators.

    GeneticSearchCV implements a genetic algorithm for hyperparameter tuning of 
    any estimator compliant with the scikit-learn API. It iteratively evolves a 
    population of hyperparameter sets towards better performance, measured via 
    cross-validation on the provided data.

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn estimator. The object to use to fit the data.

    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter 
        settings to try as values.

    scoring : str, callable, list/tuple, dict or None, default=None
        A single string (see The scoring parameter: defining model evaluation rules) 
        or a callable (see Defining your scoring strategy from metric functions) to 
        evaluate the predictions on the test set.

    cv : int, cross-validation generator or an iterable, default=3
        Determines the cross-validation splitting strategy.

    n_population : int, default=10
        Number of individuals in each generation.

    n_generations : int, default=10
        Number of generations for the genetic algorithm.

    crossover_prob : float, default=0.5
        Probability of crossover operation.

    mutation_prob : float, default=0.2
        Probability of mutation operation.

    selection_method : {'tournament', 'roulette'}, default='tournament'
        Method used to select individuals for breeding. Supported methods are 
        'tournament' and 'roulette'.
        
        - 'tournament': A selection strategy where a number of individuals are 
          randomly chosen from the population, and the one with the best fitness 
          is selected for crossover. This process is repeated until enough 
          individuals are selected. This method introduces selection pressure, 
          favoring stronger individuals, but still provides chances for weaker 
          individuals to be chosen. The size of the tournament controls the 
          selection pressure: larger tournaments tend to favor stronger individuals.

        - 'roulette': Also known as 'fitness proportionate selection', this method 
          involves selecting individuals based on their fitness proportion relative 
          to the population. Each individual gets a portion of the roulette wheel, 
          with fitter individuals receiving larger portions. A random selection is 
          then made similar to spinning a roulette wheel. This method can lead to 
          faster convergence as it strongly favors individuals with high fitness, 
          but might reduce genetic diversity in the population and can suffer from 
          the issue of 'fitness scaling'.

    tournament_size : int, default=3
        Number of individuals to be selected for each tournament. This parameter 
        is used only when selection_method is 'tournament'.

    random_state : int or RandomState instance, default=None
        Controls the randomness of the estimator.

    n_jobs : int, default=1
        The number of jobs to run in parallel for both fitting and scoring. -1 means 
        using all processors.
        
    Attributes
    ----------
    cv_results_ : list of dicts
        A list containing the scores and hyperparameters of each generation. 
        Each dict has the following keys:
            - 'params': list of parameter sets tested in the generation
            - 'scores': list of mean cross-validation scores for each parameter set
            - 'generation': integer representing the generation number

    best_params_ : dict
        The parameter setting that gave the best results on the hold out data.
        This attribute is updated continuously as better parameters are found.

    best_score_ : float
        The highest mean cross-validation score achieved by `best_params_`.

    best_estimator_ : estimator object
        The estimator fitted with the `best_params_`. This estimator is cloned 
        from the original estimator passed to `GeneticSearchCV` and fitted to 
        the data passed to the `fit` method.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> X, y = load_iris(return_X_y=True)
    >>> param_grid = {'C': [1, 10, 100], 'gamma': [0.001, 0.0001]}
    >>> search = GeneticSearchCV(estimator=SVC(), param_grid=param_grid)
    >>> search.fit(X, y)
    >>> search.best_params_
    
    Notes
    -----
    The genetic algorithm is based on processes observed in natural evolution,
    such as inheritance, mutation, selection, and crossover. Mathematically,
    it aims to optimize a set of parameters by treating each parameter set as 
    an individual in a population. The fitness of each individual is determined 
    by its performance, measured using cross-validation.

    .. math::
        \text{Fitness(individual)} = \text{CrossValScore}(\text{estimator}, 
                                                          \text{parameters}, 
                                                          \text{data})

    References
    ----------
    - Goldberg, D. E. (1989). Genetic algorithms in search, optimization, and 
      machine learning.
    - Holland, J.H., 1992. Adaptation in natural and artificial systems: an 
      introductory analysis with applications to biology, control, and 
      artificial intelligence.
    """
    def __init__(
        self, 
        estimator, 
        param_grid, 
        scoring=None, 
        cv=3, 
        n_population=10, 
        n_generations=10, 
        crossover_prob=0.5, 
        mutation_prob=0.2,
        selection_method='tournament', 
        tournament_size=3, 
        random_state=None, 
        n_jobs=1
        ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.n_population = n_population
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.random_state = random_state
        self.n_jobs = n_jobs


    def fit(self, X, y):
        """
        Run the genetic algorithm for hyperparameter optimization.
    
        This method applies a genetic algorithm to optimize the hyperparameters
        of the given estimator. The process includes generating an initial population,
        evaluating their fitness using cross-validation, and iteratively producing
        new generations through selection, crossover, and mutation.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        
        y : array-like of shape (n_samples,)
            Target values (class labels for classification, real numbers for 
                           regression).
    
        Returns
        -------
        self : object
            Returns an instance of self.
    
        Notes
        -----
        The `cv_results_` attribute contains a record of the fitness/score of each 
        individual in every generation.
    
        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.svm import SVC
        >>> X, y = load_iris(return_X_y=True)
        >>> param_grid = {'C': [1, 10, 100], 'gamma': [0.001, 0.0001]}
        >>> search = GeneticSearchCV(estimator=SVC(), param_grid=param_grid)
        >>> search.fit(X, y)
        """
        self.cv_results_ = []
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None
    
        # Generate initial population
        population = self._generate_population()
    
        for generation in range(self.n_generations):
            # Parallel evaluation of fitness for each individual in the population
            scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self._evaluate_individual)(individual, X, y, self.scoring)
                for individual in population
            )
            # Update best parameters, score, and estimator
            for individual, score in zip(population, scores):
                if score > self.best_score_:
                    self.best_params_ = individual
                    self.best_score_ = score
                    self.best_estimator_ = clone(self.estimator).set_params(**individual)
    
            # Store results for the generation
            self.cv_results_.append({
                'params': population,
                'scores': scores,
                'generation': generation
            })
    
            # Create next generation
            population = self._create_next_generation(population, scores)
    
        return self
    
    def _evaluate_individual(self, individual, X, y, scoring):
        """
        Evaluate a single individual (set of hyperparameters) using cross-validation.
    
        Parameters
        ----------
        individual : dict
            Dictionary representing an individual's hyperparameters.
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
        scoring : str or callable
            Scoring method to use on the test set.
    
        Returns
        -------
        float
            Mean cross-validation score for the individual.
        """
        estimator = clone(self.estimator).set_params(**individual)
        return np.mean(cross_val_score(estimator, X, y, cv=self.cv, scoring=scoring))
    
    def _generate_population(self):
        """
        Generate the initial population of hyperparameter sets.

        This method creates a population of individuals, each representing a set of 
        hyperparameters. The individuals are randomly generated based on the specified
        parameter grid.

        Returns
        -------
        population : list
            A list of individuals, where each individual is a dictionary of 
            hyperparameters.
        """
        population = []
        for _ in range(self.n_population):
            individual = {}
            for param, values in self.param_grid.items():
                individual[param] = np.random.choice(values)
            population.append(individual)
        return population

    def _select(self, population, scores):
        """
        Select individuals for crossover based on the specified selection method.

        Parameters
        ----------
        population : list
            The current population of individuals.
        scores : list
            The scores for each individual in the population.

        Returns
        -------
        selected : list
            A list of selected individuals for crossover.
        """
        if self.selection_method == 'tournament':
            return self._tournament_selection(population, scores)
        elif self.selection_method == 'roulette':
            return self._roulette_wheel_selection(population, scores)
        else:
            raise ValueError("Unknown selection method: {}".format(self.selection_method))

    def _tournament_selection(self, population, scores):
        """
        Tournament selection implementation.

        Repeatedly selects `tournament_size` individuals randomly and chooses 
        the best among them until the desired number of parents is selected.

        """
        selected = []
        for _ in range(len(population)):
            contenders = np.random.choice(len(population), 
                                          self.tournament_size, replace=False)
            best = np.argmax(np.array(scores)[contenders])
            selected.append(population[contenders[best]])
        return selected

    def _roulette_wheel_selection(self, population, scores):
        """
        Roulette wheel selection implementation.

        Assigns a roulette wheel slice to each individual proportional to its
        score and selects individuals by spinning the wheel.

        [Add more detailed documentation if necessary]
        """
        selected = []
        max_score = max(scores)
        score_sum = sum(max_score - score for score in scores)  # Inverse scoring for minimization
        wheel = [(max_score - score) / score_sum for score in scores]
        cumulative_wheel = np.cumsum(wheel)

        for _ in range(len(population)):
            r = np.random.rand()
            selected_idx = np.searchsorted(cumulative_wheel, r)
            selected.append(population[selected_idx])
        return selected


    def _crossover(self, parent1, parent2):
        """
        Apply crossover operation between two parents.

        This method combines two individuals (parents) to produce a new individual
        (offspring) by randomly swapping some of their hyperparameters.

        Parameters
        ----------
        parent1 : dict
            The first parent individual.
        parent2 : dict
            The second parent individual.

        Returns
        -------
        offspring : dict
            The resulting offspring individual.
        """
        offspring = parent1.copy()
        for param in parent1:
            if np.random.rand() < self.crossover_prob:
                offspring[param] = parent2[param]
        return offspring

    def _create_next_generation(self, current_generation, scores):
        """
        Create the next generation from the current population.
    
        This method applies selection to choose individuals from the current 
        generation, uses crossover to create offspring, and then applies mutation 
        to these offspring. The resulting generation replaces the current one.
    
        Parameters
        ----------
        current_generation : list
            The current population of individuals.
        scores : list
            The scores for each individual in the current population.
    
        Returns
        -------
        next_generation : list
            The new generation of individuals.
        """
        next_generation = []
        selected = self._select(current_generation, scores)
    
        # Pairing for crossover
        for i in range(0, len(selected), 2):
            parent1, parent2 = selected[i], selected[np.random.randint(len(selected))]
            offspring1 = self._crossover(parent1, parent2)
            offspring2 = self._crossover(parent2, parent1)
            next_generation.append(self._mutate(offspring1))
            if len(next_generation) < len(current_generation):
                next_generation.append(self._mutate(offspring2))
    
        # In case of odd number, randomly add one more to complete the population
        if len(next_generation) < len(current_generation):
            next_generation.append(selected[np.random.randint(len(selected))])
    
        return next_generation

    def _mutate(self, individual):
        """
        Apply mutation to an individual's hyperparameters.
    
        Each hyperparameter of the individual has a chance to be mutated based on 
        the mutation probability. The method randomly alters the hyperparameter, 
        selecting a new value from the parameter grid.
    
        Parameters
        ----------
        individual : dict
            An individual's hyperparameters.
    
        Returns
        -------
        mutated_individual : dict
            The mutated individual.
        """
        mutated_individual = individual.copy()
        for param, values in self.param_grid.items():
            if np.random.rand() < self.mutation_prob:
                mutated_individual[param] = np.random.choice(values)
        return mutated_individual
    
    def __getstate__(self):
       """
       Return the state of the object as a dictionary for pickling.
       """
       state = self.__dict__.copy()

       # Exclude non-picklable attributes if necessary
       # For example, if 'estimator' is a complex object, handle it accordingly
       if 'estimator' in state and hasattr(state['estimator'], '__getstate__'):
           state['estimator'] = state['estimator'].__getstate__()

       # Exclude other non-picklable attributes if there are any
       # e.g., state.pop('non_picklable_attribute', None)

       return state

    def __setstate__(self, state):
        """
        Restore the state of the object from the pickle.
        """
        # If 'estimator' was handled in __getstate__, restore it accordingly
        if 'estimator' in state and hasattr(state['estimator'], '__setstate__'):
            restored_estimator = state['estimator']
            self.estimator = restored_estimator.__class__()
            self.estimator.__setstate__(restored_estimator)
    
        # Restore other attributes
        self.__dict__.update(state)
