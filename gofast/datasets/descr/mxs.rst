.. _hongliu_coal_mine_mixture_dataset:

Hongliu Coal Mine Mixture Dataset
----------------------------------

**Data Set Characteristics:**

    :Number of Instances: 11 boreholes (1,038 samples in the dataset)
    :Maximum Depth Observed: 721.13 meters
    :Number of Features: 13
    :Target Variable: :math:`K` (permeability coefficient)
    :Attribute Information:
        - `hole_id` (Borehole ID): Unique identifier for each borehole.
        - `strata_name` (Geological Structure): Geological formation.
        - `rock_name` (Strata Type): Specific rock or strata encountered.
        - `aquifer_group` (Aquifer Group): Group designation based on permeability and geological formation.
        - `depth_top` (Depth Top, meters): Depth at the top of the layer.
        - `depth_bottom` (Depth Bottom, meters): Depth at the bottom of the layer.
        - `layer_thickness` (Layer Thickness, meters): Thickness of the geological layer.
        - `resistivity` (Resistivity, ohm-m): Electrical resistivity values of the layer.
        - `gamma_gamma` (Gamma-Gamma, g/cm³): Measurement of density from gamma-gamma logging.
        - `natural_gamma` (Natural Gamma, API units): Natural gamma radiation levels.
        - `sp` (Spontaneous Polarization, mV): Measurement of spontaneous polarization.
        - `short_distance_gamma` (Short-Distance Gamma, g/cm³): Another gamma logging feature.
        - `well_diameter` (Well Diameter, meters): Diameter of the borehole.

    :Summary Statistics:

    =============== ======================== ====================
    Attribute        Count                   Missing Values
    =============== ======================== ====================
    hole_id          11                      0           
    strata_name      11                      0           
    rock_name        11                      0           
    aquifer_group    11                      0           
    depth_top        11                      0           
    depth_bottom     11                      0           
    layer_thickness  11                      0           
    resistivity      1038                    0           
    gamma_gamma      1038                    0           
    natural_gamma    1038                    0           
    sp               1038                    0           
    short_distance_gamma   1038              0           
    well_diameter    1038                    0           
    :math:`K` (target) 1038                  656 (63.2%)
    =============== ======================== ====================

    :Missing Attribute Values:
        - :math:`K` (permeability coefficient): Significant missing values (63.2% of samples).
        - No missing data for predictor variables.

    :Class Distribution:
        - :math:`K_1`: Low permeability (:math:`0 < K \leq 0.01` m/d) - 382 samples (36.8%)
        - :math:`K_2`: Medium permeability (:math:`0.01 < K \leq 0.07` m/d) - Derived
        - :math:`K_3`: High permeability (:math:`K > 0.07` m/d) - Derived

    :Additional Insights:
        - Target :math:`K` values were imputed or predicted using machine learning strategies for missing data.
        - The dataset combines geophysical logging features with supervised and unsupervised machine learning.

    :Creator: K.L. Kouadio (lkouadio@csu.edu.cn)
    :Source: Machine Learning Strategy for Predicting Aquifer Permeability Coefficient :math:`K` in Hongliu Coal Mine
    :Year: 2024

**Enhanced Descriptive Statistics:**

    - :math:`K` Data:
        - Non-Null Count: 382
        - Missing Count: 656
        - Data Type: float64
        - Memory Usage: ~8.2 KB
    - Overall Dataset Shape: (1,038 rows × 13 columns)

    - Feature Highlights:
        - `depth_top` and `depth_bottom` span 0 to 721.13 meters.
        - `layer_thickness` ranges from 0.01 to 10.3 meters.
        - `resistivity` shows a wide range, indicating variability in geological composition.

**Advanced Correlation Analysis:**

    - Feature Correlations:
        - Strong Positive Correlation:
            - `depth_bottom` and `depth_top` (:math:`r = 0.98`): Indicates deeper layers follow a predictable geological sequence.
            - `gamma_gamma` and `resistivity` (:math:`r = 0.75`): Suggests higher resistivity layers have increased density.
        - Moderate Positive Correlation:
            - `natural_gamma` and `sp` (:math:`r = 0.55`): Possible shared dependency on aquifer composition.
        - Weak or No Correlation:
            - `well_diameter` and :math:`K` (:math:`r = 0.05`): Suggests borehole diameter has minimal 
              direct influence on permeability.

    - :math:`K` vs Features:
        - Significant Negative Correlation:
            - `layer_thickness` and :math:`K` (:math:`r = -0.48`): Thicker layers may reduce permeability due to higher compaction.
        - Significant Positive Correlation:
            - `resistivity` and :math:`K` (:math:`r = 0.61`): More resistive layers tend to be more 
              permeable, possibly indicating sandy aquifers.

    - Multivariate Insights:
        - Principal Component Analysis (PCA) indicates the first two components explain 82% of the variance, with `depth_top`, `resistivity`, 
          and `gamma_gamma` contributing most to the variance.

This dataset stems from a study on aquifer permeability prediction, using a mixture learning strategy (MXS). 
It includes extensive geophysical logging and pumping test data to determine the permeability coefficient :math:`K`. Given the 
challenges associated with missing :math:`K` values, a clustering-based proxy labeling system was 
implemented, supported by supervised machine learning models such as SVM and XGBoost. The study aims to 
enhance prediction accuracy and inform groundwater resource management in mining contexts.

.. topic:: References

   - Kouadio, K.L., Liu, J., Liu, R., et al. (2024). "A mixture learning strategy for predicting aquifer 
     permeability coefficient :math:`K`." Computers & Geosciences, Elsevier. DOI: https://doi.org/10.1016/j.cageo.2024.105819

