.. _bagoue_dataset:

Bagoue DC Parameters Dataset
-----------------------------

**Data Set Characteristics:**

    :Number of Instances: 431
    :Number of Attributes: 12 numeric/categorical predictive attributes. 'Flow' (Attribute 13) is usually the target.
    :Attribute Information:
        - NUM: Number of boreholes collected
        - NAME: Borehole ID
        - EAST: Easting coordinates in UTM (zone 27N, WGS84) in meters (m)
        - NORTH: Northing coordinates in UTM (zone 29N, WGS84) in meters (m)
        - POWER: Power computed at the selected anomaly (conductive zone) in meters (m)
        - MAGNITUDE: Magnitude computed at the selected anomaly (conductive zone) in ohm.m
        - SHAPE: Shape detected at the selected anomaly (conductive zone)
        - TYPE: Type detected at the selected anomaly (conductive zone)
        - SFI: Pseudo-fracturing index computed at the selected anomaly
        - OHMS: Ohmic-area (presumed to reveal the existence of the fracture) in ohm.m^2
        - LWI: Water inrush collected after drilling operations in meters (m)
        - GEOL: Geology of the exploration area
        - FLOW: Flow rate obtained after drilling operations in cubic meters per hour (m^3/h)

    :Missing Attribute Values: 5.34% on OHMS

    :Creator: Kouadio, L.
    :Date: October, 2022

The Bagoue dataset was collected in the Bagoue region (northern part of Côte d'Ivoire, West Africa) during the 
campaign for Drinking Water Supply (CDWS) projects. These projects include the Presidential Emergency Program 
(2012-2013) and the National Drinking Water Supply Program (PNAEP, 2014). The dataset comprises 431 DC-Electrical 
Resistivity Profiles (ERP), 407 DC-Electrical Soundings (VES), and 431 boreholes data. The ERP and VES methods 
utilized the Schlumberger array configuration with a 200m distance for current electrodes and 20m for potential 
electrodes during both programs.

.. topic:: References

   - Kouadio, K.L., Kouame, L.N., Drissa, C., Mi, B., Kouamelan, K.S., Gnoleba, S.P.D., Zhang, H., et al. (2022). "Groundwater Flow Rate Prediction from Geo-Electrical Features using Support Vector Machines." Water Resour. Res. https://doi.org/10.1029/2021WR031623
   - Kouadio, K.L., Liu, Jianxin, Kouamelan, S.K., Liu, Rong. (2023). "Ensemble Learning Paradigms for Flow Rate Prediction Boosting." Water Resources Management, vol. 37, no. 11, pp. 4413-4431. doi:https://doi.org/10.1007/s11269-023-03562-5. Springer.
   - Kra, K.J., Koffi, Y.S.K., Alla, K.A. & Kouadio, A.F. (2016). "Projets d'emergence post-crise et disparite territoriale en Côte d'Ivoire." Les Cah. du CELHTO, 2, 608-624.
   - Mel, E.A.C.T., Adou, D.L. & Ouattara, S. (2017). "Le programme presidentiel d'urgence (PPU) et son impact dans le departement de Daloa (Côte d'Ivoire)." Rev. Geographie Trop. d'Environnement, 2, 10. Retrieved from http://revue-geotrope.com/update/root_revue/20181202/13-Article-MEL-AL.pdf.
   
