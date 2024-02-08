.. _statlog_heart_dataset:

Statlog Heart Disease Dataset
-----------------------------

**Data Set Characteristics:**

    :Number of Instances: 270
    :Number of Attributes: 13 numeric, predictive attributes and 1 class attribute
    :Attribute Information:
        - age: Age in years
        - sex: Sex (1 = male; 0 = female)
        - cp: Chest pain type (Value 1: typical angina; Value 2: atypical angina; Value 3: non-anginal pain; Value 4: asymptomatic)
        - trestbps: Resting blood pressure (in mm Hg on admission to the hospital)
        - chol: Serum cholesterol in mg/dl
        - fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
        - restecg: Resting electrocardiographic results (Values 0, 1, 2)
        - thalach: Maximum heart rate achieved
        - exang: Exercise-induced angina (1 = yes; 0 = no)
        - oldpeak: ST depression induced by exercise relative to rest
        - slope: The slope of the peak exercise ST segment (Values 1, 2, 3)
        - ca: Number of major vessels (0-3) colored by fluoroscopy
        - thal: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
        - presence: Presence of heart disease (Value 0: absence; Value 1: presence)

    :Summary Statistics:

    ==================== ====== ====== ======= ======= =================
                         Min    Max    Mean    SD     Class Correlation
    ==================== ====== ====== ======= ======= =================
    age:                 29.0   77.0   54.43   9.11    0.2123
    sex:                 0.0    1.0    0.68    0.47    0.2977
    cp:                  1.0    4.0    3.17    0.95    0.4174
    trestbps:            94.0   200.0  131.34  17.86   0.1554
    chol:                126.0  564.0  249.66  51.69   0.1180
    fbs:                 0.0    1.0    0.15    0.36   -0.0163
    restecg:             0.0    2.0    1.02    1.00    0.1821
    thalach:             71.0   202.0  149.68  23.17  -0.4185
    exang:               0.0    1.0    0.33    0.47    0.4193
    oldpeak:             0.0    6.2    1.05    1.15    0.4180
    slope:               1.0    3.0    1.59    0.61    0.3376
    ca:                  0.0    3.0    0.67    0.94    0.4553
    thal:                3.0    7.0    4.70    1.94    0.5250
    ==================== ====== ====== ======= ======= =================

    :Missing Attribute Values: None
    :Class Distribution: The dataset contains two classes with the following distribution: 44.44% (Value 1: absence of heart disease), 
	55.56% (Value 2: presence of heart disease).

**Dataset Description:**

This dataset is a heart disease database similar to a database already present in the repository but in a slightly different form. 
It consists of 270 instances, each described by 13 numeric, predictive attributes, and one target class attribute that indicates 
the presence of heart disease. The dataset provides a diverse set of variables that are commonly used for heart disease prediction, 
including demographic, physiological, and blood test data. 
Further information might be sought directly through the UCI Machine Learning Repository or relevant clinical study 
references in the field of cardiology.

.. topic:: References

   Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, 
   School of Information and Computer Science. 
