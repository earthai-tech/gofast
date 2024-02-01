.. _nansha_plus_dataset:

Nansha Drilling Dataset
-----------------------

**Localization**
    :Country: China
    :Province: Guangdong
    :City: Guangzhou
    :UTM Zone: 49Q
    :EPSG: 21419
    :Projection: Xian 80-94/Beijing 1954-19° Belt

**Data Set Characteristics:**

    :Number of Instances: 58
    :Number of Attributes: 9 (3 dates, 4 numerics, 2 categoricals)
    :Attribute Information:
        - Year of drilling
        - Hole ID (hole_id) as the code of drilling
        - Type of drilling (53 engineering or 5 hydrogeological)
        - Longitude in decimal degrees
        - Latitude in decimal degrees
        - Easting coordinates in meters
        - Northing coordinates in meters
        - Ground height distance in meters
        - Depth of boreholes in meters
        - Opening date of the drilling
        - End date of the drilling

    :Summary Statistics:

      - Dates

      =================== ============== ================ ============== ================== ================ ============
      Year                Opening Date   End Date         Drilling       Type              Easting          Northing 
      =================== ============== ================ ============== ================== ================ ============
      2018                01/07/2018     -                NSGXX-NSSXX    Engineering        -                -  
      -                   01/07/2018     -                NSGC25         Engineering        2522589          19759356
      -                   -              13/07/2019       19NSSW01       Hydrogeological    2509081          19774075
      =================== ============== ================ ============== ================== ================ ============

      - Numerics

      ======================== =============== ============== ===============
                                Min             Max           Mean        
      ======================== =============== ============== ===============
      Year:                     2018            2019          2018.579
      Longitude:                -87.44506       -86.68665      -86.88448
      Latitude:                 1.864942        2.187123       2.025237
      Easting:                  2499111         2587894        2522117
      Northing:                 19741740        19778700       19760640
      Ground Height Distance:   0.1             12.0           4.115789
      Depth:                    20.8            203.15         89.40228
      ======================== =============== ============== ===============

    :Missing Attribute Values: None
    :Creator: K.L. Laurent (lkouao@csu.edu.cn) and Liu Rong (liurongkaoyan@csu.edu.cn)
    :Donor: Central South University - School of Geosciences and Info-physics (https://en.csu.edu.cn/)
    :Date: June, 2023

The Nansha data was collected during the Nashang project from 2018 to 2019. The primary focus of the 
drilling was engineering drillings for controlling soil settlement and quality. Additionally, several 
hydrogeological drillings were also performed. The overarching goal of the Nanshang project is to forecast 
land subsidence from 2024 to 2035 using various influential factors, including InSAR data, highways map, 
proximity to roads, rivers (such as the Pearl River), and others.

.. topic:: References

   - @article{liu2024machine,
       title={Machine learning-based techniques for land subsidence simulation in an urban area},
       author={Liu, Jianxin and Liu, Wenxiang and Allechy, Fabrice Blanchard and Zheng, Zhiwen and Liu, Rong and Kouadio, Kouao Laurent},
       journal={Journal of Environmental Management},
       volume={352},
       pages={120078},
       year={2024},
       doi={https://doi.org/10.1016/j.jenvman.2024.120078},
       publisher={Elsevier}
   }
