.. _hlogs_dataset:

Hydro-Logging Dataset for Hydrogeophysical Analysis
---------------------------------------------------

**Data Set Characteristics:**

    :Number of Instances: 1148
    :Number of Attributes: 27 (including both numerical and categorical attributes)
    :Attribute Information:
        - hole_id: Identifier for the borehole
        - depth_top: Top depth of the layer measured
        - depth_bottom: Bottom depth of the layer measured
        - strata_name: Name of the strata
        - rock_name: Name of the rock
        - layer_thickness: Thickness of the layer
        - resistivity: Electrical resistivity of the layer
        - gamma_gamma: Gamma-Gamma logging data
        - natural_gamma: Natural gamma radiation data
        - sp: Spontaneous potential data
        - short_distance_gamma: Short distance gamma-ray data
        - well_diameter: Diameter of the well
        - aquifer_group: Group of the aquifer
        - pumping_level: Level of pumping
        - aquifer_thickness: Thickness of the aquifer
        - hole_depth_before_pumping: Depth of the hole before pumping
        - hole_depth_after_pumping: Depth of the hole after pumping
        - hole_depth_loss: Loss in depth due to pumping
        - depth_starting_pumping: Depth at the start of pumping
        - pumping_depth_at_the_end: Pumping depth at the end of the operation
        - pumping_depth: Depth of pumping
        - section_aperture: Aperture of the section
        - k: Permeability coefficient
        - kp: Alternative permeability coefficient
        - r: Resistance measure
        - rp: Alternative resistance measure
        - remark: Additional remarks or notes
        
    :Summary Statistics:

    | Attribute                    | Mean     | SD        | Min      | Max      | Correlation with Resistivity |
    |------------------------------|----------|-----------|----------|----------|------------------------------|
    | depth_top                    | 432.58   | 188.79    | 0.00     | 700.44   | 0.286                        |
    | depth_bottom                 | 438.53   | 185.38    | 1.40     | 721.13   | 0.282                        |
    | layer_thickness              | 5.29     | 7.95      | 0.08     | 70.67    | -0.191                       |
    | resistivity                  | 24.86    | 11.07     | 7.82     | 112.43   | 1.000                        |
    | gamma_gamma                  | 1882.29  | 1720.05   | 440.37   | 8907.03  | 0.279                        |
    | natural_gamma                | 12.80    | 4.23      | 1.44     | 31.58    | -0.289                       |
    | sp                           | -8.25    | 6.69      | -26.18   | -0.06    | 0.263                        |
    | short_distance_gamma         | 4199.60  | 3601.58   | 130.66   | 16062.44 | 0.340                        |
    | well_diameter                | 176.81   | 46.95     | 62.35    | 273.71   | -0.303                       |
    | aquifer_thickness            | 76.20    | 34.55     | 43.82    | 139.61   | 0.300                        |
    | hole_depth_before_pumping    | 514.74   | 162.32    | 242.00   | 699.14   | -0.068                       |
    | hole_depth_after_pumping     | 513.19   | 162.08    | 240.06   | 697.34   | -0.066                       |
    | hole_depth_loss              | -0.26    | 1.69      | -2.77    | 2.48     | -0.314                       |
    | depth_starting_pumping       | 339.08   | 175.77    | 80.00    | 580.06   | -0.109                       |
    | pumping_depth_at_the_end     | 517.82   | 162.09    | 242.06   | 700.64   | -0.065                       |
    | pumping_depth                | 178.74   | 54.72     | 104.00   | 280.49   | 0.159                        |
    | section_aperture             | 198.77   | 11.95     | 190.00   | 215.00   | 0.005                        |
    | k                            | 0.045    | 0.052     | 0.0002   | 0.173    | -0.087                       |
    | kp                           | 0.044    | 0.049     | 0.0002   | 0.130    | -0.077                       |
    | r                            | 49.37    | 35.01     | 3.41     | 94.90    | 0.265                        |
    | rp                           | 72.47    | 44.28     | 3.41     | 124.67   | -0.026                       |

    :Missing Attribute Values:
    - rock_name: 82.93%
    - layer_thickness: 2.26%
    - resistivity: 0.26%
    - gamma_gamma: 8.36%
    - sp: 0.09%
    - short_distance_gamma: 0.61%
    - aquifer_group: 0.61%
    - pumping_level: 45.64%
    - aquifer_thickness: 66.72%
    - hole_depth_before_pumping: 66.72%
    - hole_depth_after_pumping: 66.72%
    - hole_depth_loss: 66.72%
    - depth_starting_pumping: 66.72%
    - pumping_depth_at_the_end: 66.72%
    - pumping_depth: 66.72%
    - section_aperture: 66.72%
    - k: 66.72%
    - kp: 66.72%
    - r: 74.91%
    - rp: 66.72%
    - remark: 92.51%
    [Other attributes have no missing values]

    :Class Distribution: The dataset is used for classification tasks, focusing on K and kp.
    :Creator: K.L. Laurent (lkouao@csu.edu.cn) and Liu Rong (liurongkaoyan@csu.edu.cn)
    :Donor: Central South University - School of Geosciences and Info-physics (https://en.csu.edu.cn/)
    :Date: June 2023

This dataset, known as the Hengshan dataset, originates from the Hengshan coal mine, one of the super-large mines in China. 
A considerable amount of drilling construction and geophysical logging exploration has been conducted in the Hengshan coal 
mine. The dataset contains 11 over 106 boreholes with complete pumping test samples. The dataset is critical for understanding 
subsurface structures and properties, particularly in the context of hydraulic conductivity in coal mine development. The repeated 
failures in pumping tests at the Hengshan mine underline the importance of accurate analysis and prediction using this dataset.


.. topic:: References

   - Liu, J., Liu, W., Blanchard Allechy, F., Zheng, Z., Liu, R., & Kouadio, K.L. (2024). Machine learning-based techniques for land subsidence simulation in an urban area. Journal of Environmental Management, 352, 120078. https://doi.org/10.1016/j.jenvman.2024.120078
   - Kouadio, K.L., Liu, J., Liu, R., Wang, Y., & Liu, W. (2024). Earth Science Informatics. https://doi.org/10.1007/s12145-024-01236-3

