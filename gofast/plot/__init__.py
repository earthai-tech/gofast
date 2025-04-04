# -*- coding: utf-8 -*-

from ._config import PlotConfig  as config
from .evaluate import ( 
    EvalPlotter, 
    MetricPlotter, 
    plot_model_scores, 
    plot_reg_scoring, 
    plot_model, 
    )
from .explore import EasyPlotter, QuestPlotter
from .ts import TimeSeriesPlotter 
from .charts import ( 
    pie_charts, 
    radar_chart_in, 
    radar_chart, 
    donut_chart, 
    plot_donut_charts, 
    donut_chart_in, 
    chord_diagram, 
    multi_level_donut,
    two_ring_donuts, 
    plot_contrib, 
    plot_radial_groups, 
    )

from .cluster import ( 
    plot_silhouette, 
    plot_silhouette_in, 
    plot_dendrogram, 
    plot_dendrogram_in, 
    plot_clusters, 
    plot_elbow, 
    plot_cluster_comparison, 
    plot_voronoi, 
)
from .comparison import ( 
    plot_feature_trend, 
    plot_density, 
    plot_prediction_comparison, 
    plot_error_analysis, 
    plot_trends, 
    plot_variability, 
    plot_factor_contribution, 
    plot_comparative_bars, 
    plot_line_graph, 
    
    )
from .dimensionality import ( 
    plot_unified_pca, 
    plot_pca_components, 
    plot_cumulative_variance, 
 )
from .feature_analysis import ( 
    plot_importances_ranking, 
    plot_rf_feature_importances, 
    plot_feature_interactions, 
    plot_variables, 
    plot_correlation_with, 
    plot_dependence, 
    plot_sbs_feature_selection, 
    plot_permutation_importance, 
    plot_regularization_path, 
    plot_feature_importances, 
   )
from .grid import plot_feature_dist_grid
from .inspection  import ( 
    plot_learning_inspection, 
    plot_learning_inspections, 
    plot_loc_projection, 
    plot_matshow, 
    plot_heatmapx, 
    plot_matrix, 
    plot_sunburst, 
    plot_sankey, 
    plot_euler_diagram, 
    create_upset_plot, 
    plot_venn_diagram, 
    plot_woodland, 
    plot_l_curve, 
  )  
from .ml_viz import ( 
    plot_confusion_matrices, 
    plot_confusion_matrix_in, 
    plot_confusion_matrix, 
    plot_roc_curves, 
    plot_taylor_diagram,
    plot_taylor_diagram_in, 
    plot_cv, 
    plot_confidence, 
    plot_confidence_ellipse, 
    plot_shap_summary, 
    plot_abc_curve, 
    plot_learning_curves, 
    plot_cost_vs_epochs, 
    plot_regression_diagnostics, 
    plot_residuals_vs_fitted, 
    plot_residuals_vs_leverage, 
    plot_r2, 
    plot_r2_in, 
    plot_cm, 
    taylor_diagram, 

 )
from .q import ( 
    plot_qbased_preds,
    plot_quantile_distributions, 
    plot_qdist, 
)
from .utils import ( 
    boxplot, 
    plot_r_squared, 
    plot_text, 
) 
from .spatial import ( 
    plot_categorical_feature,
    plot_dist,
    plot_categories_dist,
    plot_spatial_features, 
    plot_hotspot_map, 
    plot_sampling_map
)
from .suite import (
    plot_sensitivity, 
    plot_distributions, 
    plot_uncertainty, 
    plot_prediction_intervals, 
    plot_temporal_trends, 
    plot_relationship, 
    plot_fit, 
    plot_factory_ops, 
    plot_ranking, 
    plot_coverage, 
    plot_with_uncertainty, 
)

from .testing import ( 
    plot_ab_test , 
    plot_errors, 
)

__all__= [
    "config", 
    
    "MetricPlotter", 
    "EvalPlotter",
    "EasyPlotter" , 
    "QuestPlotter",
    "TimeSeriesPlotter", 
    
    "pie_charts", 
    "radar_chart",
    "radar_chart_in", 
    "donut_chart", 
    "plot_donut_charts", 
    "donut_chart_in", 
    "chord_diagram", 
    "multi_level_donut", 
    "two_ring_donuts",
    "plot_contrib", 
    'plot_radial_groups', 
    
    'plot_feature_dist_grid', 
    
    'plot_feature_trend', 
    'plot_density', 
    'plot_prediction_comparison', 
    'plot_error_analysis', 
    'plot_trends', 
    'plot_variability', 
    'plot_factor_contribution', 
    'plot_comparative_bars', 
    'plot_line_graph', 
    
    'plot_silhouette',
    'plot_silhouette_in',
    'plot_dendrogram',
    'plot_dendroheat',
    'plot_dendrogram_in',
    'plot_clusters',
    'plot_elbow',
    'plot_cluster_comparison',
    'plot_voronoi',
    'plot_unified_pca',
    'plot_pca_components',
    'plot_cumulative_variance', 
    
    "plot_model_scores",
    "plot_reg_scoring",
    "plot_model", 
    
    'plot_importances_ranking',
    'plot_rf_feature_importances',
    'plot_feature_interactions',
    'plot_variables',
    'plot_correlation_with',
    'plot_dependence',
    'plot_sbs_feature_selection',
    'plot_permutation_importance',
    'plot_regularization_path',  
    'plot_feature_importances', 
    
    'plot_learning_inspection',
    'plot_learning_inspections',
    'plot_loc_projection',
    'plot_matshow',
    'plot_heatmapx',
    'plot_matrix',
    'plot_sunburst',
    'plot_sankey',
    'plot_euler_diagram',
    'create_upset_plot',
    'plot_venn_diagram',
    'plot_set_matrix',
    'plot_woodland', 
    'plot_l_curve', 
    
    'plot_confusion_matrices',
    'plot_confusion_matrix_in', 
    'plot_confusion_matrix', 
    'plot_roc_curves',
    'plot_taylor_diagram',
    'plot_taylor_diagram_in', 
    'plot_cv',
    'plot_confidence',
    'plot_confidence_ellipse',
    'plot_shap_summary',
    'plot_abc_curve',
    'plot_learning_curves',
    'plot_cost_vs_epochs', 
    'plot_regression_diagnostics', 
    'plot_residuals_vs_leverage', 
    'plot_residuals_vs_fitted', 
    'plot_r2', 
    'plot_r2_in', 
    'plot_cm', 
    'taylor_diagram', 
    
    "boxplot", 
    "plot_r_squared",
    "plot_text", 
    "plot_spatial_features", 
    "plot_hotspot_map",
    "plot_sampling_map", 
    "plot_categorical_feature", 
    "plot_sensitivity", 
    "plot_categories_dist", 
    "plot_distributions", 
    "plot_dist", 
    "plot_quantile_distributions" , 
    'plot_uncertainty', 
    'plot_prediction_intervals',
    "plot_temporal_trends", 
    'plot_relationship', 
    'plot_fit', 
    'plot_factory_ops', 
    'plot_ranking', 
    'plot_coverage', 
    'plot_qdist', 
    'plot_with_uncertainty', 
    'plot_qbased_preds',
    
    'plot_ab_test', 
    'plot_errors'
               
    ]

