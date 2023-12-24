"""
validation sub-package focuses on  training and validation phases. It also composed 
of a set of grid-search tricks from model hyperparameters fine-tuning and 
the pretrained models fetching from :mod:`~gofast.models` and 
"""
from .search import ( 
    BaseEvaluation, 
    GridSearch, 
    GridSearchMultiple,
    get_best_kPCA_params, 
    naive_evaluation, 

    )
from .utils import ( 
    get_scorers,
    get_cv_mean_std_scores, 
    get_split_best_scores, 
    display_cv_tables, 
    display_fine_tuned_results, 
    display_model_max_details, 
    shrink_covariance_cv_score
    ) 
from .optimize import ( 
    parallelize_estimators, 
    optimize_hyperparameters 
    ) 
from .deep_search import ( 
    base_tuning, 
    robust_tuning, 
    neural_tuning,
    deep_tuning,
    )
__all__=[
    "BaseEvaluation", 
    "GridSearch", 
    "GridSearchMultiple", 
    "get_best_kPCA_params", 
    "get_scorers",
    "get_cv_mean_std_scores", 
    "get_split_best_scores", 
    "display_cv_tables", 
    "display_fine_tuned_results", 
    "display_model_max_details", 
    "naive_evaluation",
    "parallelize_estimators", 
    "optimize_hyperparameters", 
    "shrink_covariance_cv_score", 
    "base_tuning", 
    "robust_tuning", 
    "neural_tuning",
    "deep_tuning",
    ]