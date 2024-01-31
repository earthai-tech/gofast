.. _nanshang_plus_plus_dataset:

Nanshang Land Subsidence Dataset
--------------------------------

**Localization**
    :Country: China
    :Province: Guangdong
    :City: Guangzhou
    :UTM Zone: 49Q
    :EPSG: 21419
    :Projection: Xian 80-94/Beijing 1954-19° Belt

**Data Set Characteristics:**

    :Number of Instances: 274,967
    :Number of Attributes: 13
    :Attribute Information:
        - Longitude in decimal degrees
        - Latitude in decimal degrees
        - Easting coordinates in meters
        - Northing coordinates in meters
        - Year of land subsidence with respective periods:
          - 2015: 2015-06-15 to 2015-12-30
          - 2016: 2015-06-15 to 2016-12-30
          - 2017: 2015-06-15 to 2017-12-25
          - 2018: 2015-06-15 to 2018-12-20
          - 2019: 2015-06-15 to 2019-12-27
          - 2020: 2015-06-15 to 2020-12-21
          - 2021: 2015-06-15 to 2021-12-28
          - 2022: 2015-06-15 to 2022-12-23
        - Display rate (disp_rate): Ratio used to display land subsidence image at 20m x 20m resolution

    :Summary Statistics:

      ======================== ================== ============== =============== ===================
                                Min                Max               Mean          Std
      ======================== ================== ============== =============== ===================
      Easting:                  2.493606e+06       2.536786e+06   2.517969e+06    11000.145496
      Northing:                 1.973515e+07       1.978140e+07   1.976086e+07    11080.489651
      Longitude:                113.2913           113.7352       113.5394        0.106325
      Latitude:                 22.51763           22.91122       22.73937        0.100613
      2015:                     -2.173150e+01      2.198071e+01   2.454098e+00    4.564091
      2016:                     -6.418880e+01      2.242734e+01   -1.681817e+00   7.910394
      2017:                     -1.286565e+02      3.088651e+01   -4.526142e+00   13.303205
      2018:                     -1.767192e+02      4.598118e+01   -8.579401e+00   18.219551
      2019:                     -2.250049e+02      5.309525e+01   -1.256542e+01   24.311286
      2020:                     -2.630617e+02      5.890371e+01   -1.751829e+01   30.102894
      2021:                     -3.208993e+02      5.802831e+01   -2.481840e+01   37.282043
      2022:                     -3.682477e+02      6.414452e+01   -2.837542e+01   43.147739
      Display Rate:             -54.57732          7.876161       -4.416209       6.062086
      ======================== ================== ============== =============== =====================

    :Missing Attribute Values: None
    :Creator: K.L. Laurent (lkouao@csu.edu.cn) and Liu Rong (liurongkaoyan@csu.edu.cn)
    :Donor: Central South University - School of Geosciences and Info-physics (https://en.csu.edu.cn/)
    :Date: June, 2023

The Nanshang land subsidence (LS) data was collected during the Nanshang project from June 2015 to December 2022. 
Data was gathered via Interferometric Synthetic Aperture Radar (InSAR) and preprocessed to fit the LS. 
Yearly reports detail the extent of deformation and its potential causes, providing valuable insights for urban planning and 
environmental management.

.. topic:: References

   - Liu, Jianxin, Liu, Wenxiang, Allechy, Fabrice Blanchard, Zheng, Zhiwen, Liu, Rong, and Kouadio, Kouao Laurent. 
     "Machine learning-based techniques for land subsidence simulation in an urban area." Journal of Environmental Management, 
     vol. 352, 2024, pp. 120078. Elsevier. DOI: https://doi.org/10.1016/j.jenvman.2024.120078.
