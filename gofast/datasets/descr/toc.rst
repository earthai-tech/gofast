.. _toc_dataset:

TOC Dataset (Well 906)
-----------------------

    :Number of Instances: 3,002 samples from Well 906
    :Number of Features: 11 well-logging measurements
    :Target Variable: Total Organic Carbon (TOC)
    :Attribute Information:
        - `GR` (Natural Gamma Ray, API units): Measures natural gamma radiation levels. 
          Typically reflects shale composition, with radioactive material often linked to kerogen presence.
        - `RES` (Resistivity, ohm-m): Captures variations in electrical conductivity. 
          High resistivity zones indicate hydrocarbons or kerogen due to their insulating properties.
        - `CNL` (Neutron Logging, Hydrogen Index): Sensitive to bound water and hydrocarbons. 
          Helps detect organic material or gas.
        - `DEN` (Density, g/cm³): Differentiates kerogen from rock matrix. Similarities 
          between kerogen and water density can obscure relationships.
        - `AC` (Acoustic, µs/ft): Measures sonic wave velocity. Reduced velocities 
          indicate gas or kerogen presence.
        - `PE` (Photoelectric Absorption): Influenced by lithology and hydrocarbons. 
          Low PE values often appear in organic-rich shale formations.
        - `RT10` (Resistivity, ohm-m): Resistivity measured at 10 cm; detects
          insulating properties of hydrocarbons or kerogen.
        - `RT20` (Resistivity, ohm-m): Resistivity measured at 20 cm.
        - `RT30` (Resistivity, ohm-m): Resistivity measured at 30 cm.
        - `RT60` (Resistivity, ohm-m): Resistivity measured at 60 cm.
        - `RT90` (Resistivity, ohm-m): Resistivity measured at 90 cm.

        - TOC (Total Organic Carbon, %): Target variable representing the organic 
          carbon content of the formation.

    :Logs Analyzed:
        - GR: Provides insights into shale and radioactive material levels.
        - RES: Indicates hydrocarbon or kerogen zones.
        - CNL: Detects water or hydrocarbons using the hydrogen index.
        - DEN: Differentiates between kerogen and rock.
        - AC: Highlights formations with hydrocarbons or gas.
        - PE: Reflects lithology and hydrocarbon content.
        - GR (Natural Gamma Ray): Strong correlation with shale-rich, kerogen-enriched zones.
        - RT10–RT90 (Resistivity Logs): Provided critical insights into kerogen
          zones due to their insulating properties.

    :Sensitivity Analysis:
        - GR and RES were identified as key predictors of TOC in Well 906.
        - Combined feature extraction from the knowledge graph and well-logging 
          data significantly enhanced the predictive accuracy of the TOCGraph model.
          
        
    :Creator: K.L. Kouadio (lkouadio@csu.edu.cn)
    :Source: 
    The dataset was provided by the China National Petroleum Corporation and
    analyzed Total Organic Carbon Prediction
    :Year: 2025
    
   :Model Training:
      - Well 906 data was split into training and validation sets.
      - TOCGraph was trained using combined features from well-logging measurements 
        and the knowledge graph.

   :Dataset Summary:

    =============== ======================== ====================================
    Attribute        Count                   Description        
    =============== ======================== ====================================
    AC               3,002                   Sonic wave velocity (µs/ft)
    CAL              3,002                   Borehole diameter (inches)
    CNL              3,002                   Neutron hydrogen index
    DEN              3,002                   Volume density (g/cm³)
    GR               3,002                   Natural gamma radiation (API units)
    PE               3,002                   Photoelectric absorption index
    RT10             3,002                   Resistivity at 10 cm (ohm-m)
    RT20             3,002                   Resistivity at 20 cm (ohm-m)
    RT30             3,002                   Resistivity at 30 cm (ohm-m)
    RT60             3,002                   Resistivity at 60 cm (ohm-m)
    RT90             3,002                   Resistivity at 90 cm (ohm-m)
    TOC (target)     3,002                   Total Organic Carbon (%)
    =============== ======================== ====================================

   :Advanced Insights:

       - Gamma ray spectrometry (GR) showed strong correlations with shale-rich zones 
         enriched in organic material.
       - Resistivity logs (RES) provided key insights into kerogen zones due to their 
         insulating properties.
       - Photoelectric absorption (PE) values were consistently lower in organic-rich shale.

.. topic:: References

