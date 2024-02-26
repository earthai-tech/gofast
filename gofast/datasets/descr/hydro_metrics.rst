
.. _load_hydro_metrics 


Environmental Conditions Dataset
================================

Data Set Characteristics
------------------------

- **Number of Instances:** 276
- **Number of Attributes:** 9 (all numeric)
- **Attribute Information:**

    - date: The date of the observation.
    - min_temp: The minimum temperature in degrees Celsius on that day.
    - max_temp: The maximum temperature in degrees Celsius.
    - humidity: The percentage of humidity.
    - wind: Wind speed (unit not specified).
    - insulation: Likely refers to solar radiation (unit not specified).
    - ray: Sunlight intensity (MJ/m²/day).
    - eto: Potential evapotranspiration (mm/month).
    - rain: Rainfall (unit likely millimeters).
    - flow: Flow measurement (cubic meters per second).

- **Associated Tasks:** Regression, Time Series Analysis
- **Missing Values:** None
:Class Distribution: 
:Creator: K.K.Laurent < lkouao@csu.edu.cn>
:Donor: Daniel <etanoyau@gmail.com> 
:Date: July, 1988
	
Summary Statistics
------------------

+------------------+-------+-------+-------+-------+
| Measurement      | Min   | Max   | Mean  | Std   |
+==================+=======+=======+=======+=======+
| Min Temperature  | 22    | 27    | 24.39 | 0.93  |
+------------------+-------+-------+-------+-------+
| Max Temperature  | 25    | 31    | 27.67 | 1.24  |
+------------------+-------+-------+-------+-------+
| Humidity         | 40    | 92    | 74.13 | 9.62  |
+------------------+-------+-------+-------+-------+
| Wind             | 0.8   | 2.8   | 1.78  | 0.42  |
+------------------+-------+-------+-------+-------+
| Insulation       | 0     | 7     | 4.04  | 1.49  |
+------------------+-------+-------+-------+-------+
| Ray              | 8.2   | 20.2  | 14.96 | 2.20  |
+------------------+-------+-------+-------+-------+
| ETO              | 73.63 | 171.87| 109.9 | 18.98 |
+------------------+-------+-------+-------+-------+
| Rain             | 0     | 322   | 92.36 | 65.09 |
+------------------+-------+-------+-------+-------+
| Flow             | 0     | 326.33| 4.99  | 25.38 |
+------------------+-------+-------+-------+-------+


Dataset Description
-------------------

This dataset comprises 276 instances of daily environmental and meteorological measurements, including temperature,
humidity, wind speed, solar radiation, evapotranspiration, rainfall, and flow measurements. It is aimed at supporting studies 
in environmental science, agriculture, meteorology, and hydrology, particularly for understanding weather patterns, climate impact 
on agriculture, or water resource management. The dataset covers various aspects of environmental conditions that can be used for 
regression, time series analysis, and other associated tasks in data analysis and predictive modeling.
