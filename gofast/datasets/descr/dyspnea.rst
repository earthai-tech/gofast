
.. _dyspnea_dataset:

Dyspnea Dataset
-------------------

**Data Set Characteristics:**

    :Number of Instances: 279
    :Number of Attributes: 44
    :Attribute Information:
        - start_date
		- starttime
		- endtime
		- date_entered
		- submission_time
		- version
		- submitted_by
		- meta_instanceid
		- uuid
		- id
		- sex
		- index
		- file_number
		- age
		- not
		- pad
		- fc
		- fr
		- spo
		- temperature
		- glasgow_score
		- diagnosis_pneumonitis
		- diagnosis_asthma_attack
		- diagnosis_pulmonary_tuberculosis
		- diagnosis_covid_19
		- diagnosis_heart_failure
		- diagnosis_copd
		- diagnosis_bronchial_cancer
		- diagnosis_pulmonary_fibrosis
		- diagnostic_other
		- parent_index
		- duration
		- xform_id
		- dyspnea
		- nyha_intensity
		- frequency
		- cough
		- fever
		- asthenia
		- admission_method
		- establishment_of_origin
		- toxic_tobacco
		- toxic_alcohol
		- condition
		- state_of_the_pupils
		- conjunctivas
		- imo
		- condition_of_the_mucous_membranes
		- dehydration_skin_fold
		- respiratory_distress
		- heart_sound
		- breath
		- heart_failure
		- lymphadenopathy
		- diagnosis_retained
		- outcome_of_hospitalization
		
    :Summary Statistics:

    | Attribute                                     | Mean       | SD         | Min   | 25%    | 50%    | 75%    | Max   |
    |-----------------------------------------------|------------|------------|-------|--------|--------|--------|-------|
    | age                                           | 49.4444    | 16.0549    | 16.0  | 39.0   | 49.0   | 61.0   | 88.0  |
    | not                                           | 138.7778   | 39.5220    | 57.0  | 110.0  | 130.0  | 166.0  | 286.0 |
    | pad                                           | 85.5878    | 24.1461    | 10.0  | 70.0   | 80.0   | 100.0  | 186.0 |
    | fc                                            | 103.0860   | 17.9139    | 48.0  | 90.0   | 102.0  | 111.5  | 172.0 |
    | fr                                            | 30.9283    | 7.2416     | 12.0  | 28.0   | 32.0   | 34.0   | 102.0 |
    | spo                                           | 87.0394    | 9.9083     | 50.0  | 85.0   | 90.0   | 93.0   | 99.0  |
    | temperature                                   | 37.6308    | 2.0116     | 7.0   | 37.0   | 38.0   | 38.0   | 40.0  |
    | glasgow_score                                 | 14.4265    | 1.2978     | 8.0   | 15.0   | 15.0   | 15.0   | 16.0  |
    | diagnosis_pneumonitis                         | 0.3513     | 0.4782     | 0.0   | 0.0    | 0.0    | 1.0    | 1.0   |
    | diagnosis_asthma_attack                       | 0.0538     | 0.2260     | 0.0   | 0.0    | 0.0    | 0.0    | 1.0   |
    | diagnosis_pulmonary_tuberculosis              | 0.1254     | 0.3318     | 0.0   | 0.0    | 0.0    | 0.0    | 1.0   |
    | diagnosis_covid_19                            | 0.2294     | 0.4212     | 0.0   | 0.0    | 0.0    | 0.0    | 1.0   |
    | diagnosis_heart_failure                       | 0.0824     | 0.2755     | 0.0   | 0.0    | 0.0    | 0.0    | 1.0   |
    | diagnosis_copd                                | 0.0215     | 0.1453     | 0.0   | 0.0    | 0.0    | 0.0    | 1.0   |
    | diagnosis_bronchial_cancer                    | 0.0143     | 0.1191     | 0.0   | 0.0    | 0.0    | 0.0    | 1.0   |
    | diagnosis_pulmonary_fibrosis                  | 0.0143     | 0.1191     | 0.0   | 0.0    | 0.0    | 0.0    | 1.0   |
    | diagnostic_other                              | 0.4373     | 0.4969     | 0.0   | 0.0    | 0.0    | 1.0    | 1.0   |
    | parent_index                                  | -1.0000    | 0.0000     | -1.0  | -1.0   | -1.0   | -1.0   | -1.0  |
    | duration                                      | 1996.3871  | 9048.6305  | 189.0 | 267.0  | 338.0  | 501.0  | 102454.0 |
    | xform_id                                      | 745540.0000 | 0.0000     | 745540.0 | 745540.0 | 745540.0 | 745540.0 | 745540.0 |
    | gender                                        | 0.4875     | 0.5007     | 0.0   | 0.0    | 0.0    | 1.0    | 1.0   |
    | dyspnea                                       | 0.1004     | 0.3010     | 0.0   | 0.0    | 0.0    | 0.0    | 1.0   |
    | nyha_intensity                                | 1.2258     | 0.7510     | 0.0   | 1.0    | 1.0    | 2.0    | 2.0   |
    | frequency                                     | 0.9391     | 0.2396     | 0.0   | 1.0    | 1.0    | 1.0    | 1.0   |
    | cough                                         | 0.7921     | 0.4065     | 0.0   | 1.0    | 1.0    | 1.0    | 1.0   |
    | fever                                         | 0.8100     | 0.3930     | 0.0   | 1.0    | 1.0    | 1.0    | 1.0   |
    | asthenia                                      | 0.7634     | 0.4257     | 0.0   | 1.0    | 1.0    | 1.0    | 1.0   |
    | admission_method                              | 2.5197     | 1.8424     | 0.0   | 0.0    | 4.0    | 4.0    | 4.0   |
    | establishment_of_origin                       | 9.4659     | 4.1167     | 0.0   | 6.0    | 13.0   | 13.0   | 15.0  |
    | toxic_tobacco                                 | 0.2832     | 0.4513     | 0.0   | 0.0    | 0.0    | 1.0    | 1.0   |
    | toxic_alcohol                                 | 0.2079     | 0.4065     | 0.0   | 0.0    | 0.0    | 0.0    | 1.0   |
    | condition                                     | 0.5591     | 0.6753     | 0.0   | 0.0    | 0.0    | 1.0    | 2.0   |
    | state_of_the_pupils                           | 0.0072     | 0.0845     | 0.0   | 0.0    | 0.0    | 0.0    | 1.0   |
    | conjunctivas                                  | 1.0645     | 0.6964     | 0.0   | 1.0    | 1.0    | 2.0    | 2.0   |
    | imo                                           | 0.3369     | 0.4735     | 0.0   | 0.0    | 0.0    | 1.0    | 1.0   |
    | condition_of_the_mucous_membranes             | 0.6953     | 0.4611     | 0.0   | 0.0    | 1.0    | 1.0    | 1.0   |
    | dehydration_skin_fold                         | 0.3118     | 0.4641     | 0.0   | 0.0    | 0.0    | 1.0    | 1.0   |
    | respiratory_distress                          | 0.8351     | 0.3717     | 0.0   | 1.0    | 1.0    | 1.0    | 1.0   |
    | heart_sound                                   | 0.0824     | 0.2755     | 0.0   | 0.0    | 0.0    | 0.0    | 1.0   |
    | breath                                        | 0.0681     | 0.2524     | 0.0   | 0.0    | 0.0    | 0.0    | 1.0   |
    | heart_failure                                 | 0.0645     | 0.2461     | 0.0   | 0.0    | 0.0    | 0.0    | 1.0   |
    | lymphadenopathy                               | 0.0466     | 0.2111     | 0.0   | 0.0    | 0.0    | 0.0    | 1.0   |
    | diagnosis_retained                            | 12.6416    | 5.4615     | 0.0   | 9.0    | 13.0   | 17.0   | 26.0  |
    | outcome_of_hospitalization                    | 1.7168     | 1.1701     | 0.0   | 0.0    | 2.0    | 3.0    | 3.0   |

    :Missing Attribute Values: None

    :Creator: Kouadio K. Laurent (etanoyau@gmail.com)
    :Donor: Martial Konan
    :Year: 2024

The Dyspnea Dataset, collected from Côte d'Ivoire, represents a significant effort to compile clinical data on patients experiencing 
dyspnea. It stands as a valuable resource for medical research, offering insights into the prevalence, causes, and outcomes of 
respiratory distress across a diverse patient population. By analyzing this dataset, researchers can contribute to the development 
of more effective diagnostic and treatment strategies for dyspnea and its underlying conditions.

    - Demographic data: age, sex
	- Clinical measurements: heart rate (fc), respiratory rate (fr), blood oxygen saturation (spo), temperature, Glasgow Coma Scale score (glasgow_score)
	- Diagnoses: including but not limited to pneumonitis, asthma attack, pulmonary tuberculosis, COVID-19, heart failure, COPD (Chronic Obstructive Pulmonary Disease), bronchial cancer, and pulmonary fibrosis
	- Symptom and history indicators: dyspnea, NYHA (New York Heart Association) intensity, frequency of symptoms, cough, fever, asthenia, exposure to tobacco and alcohol
	- Outcome measures: diagnosis retained, outcome of hospitalization

.. topic:: References
