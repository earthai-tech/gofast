.. _sjr_bet_plus_dataset:

Enhanced Ghanaian Lottery Dataset for Machine Learning ("SJR bet+")
-------------------------------------------------------------------

**Data Set Characteristics:**

    :Number of Instances: [Number of instances in the ML dataset]
    :Number of Attributes: 252 (including both numerical and categorical attributes)
    :Attribute Information:
        - number: The specific lottery number being analyzed
        - [1-264]: Binary indicators for each draw, showing if the number won (1) or not (0)
        - frequency: Total frequency of the number's appearance in past draws
        - most_recent_draw: The most recent draw in which the number appeared
        - target: Binary target indicating if the number wins in the next draw

    :Summary Statistics:
	
	
    :Class Distribution: The dataset is used for binary classification tasks, predicting whether a specific number will win in the next draw based on historical data.
    :Creator: K K.L. Laurent (etanoyau@gmail.com/ lkouao@csu.edu.cn) 
    :Donor: Sekongo N. Jean-Rene, CEO of Bouake Department of Mining operations, Bouake, Cote d'Ivoire (jeanrenesekongo@yahoo.fr)
    :Date: January 2024
	
	
**Dataset Description:**

The "SJR bet+" dataset is an enhanced version of the original "SJR bet" dataset, specifically prepared for classical machine 
learning predictions. This dataset transforms the original lottery draw data into a format suitable for binary classification 
tasks, where the goal is to predict the likelihood of a specific number winning in the next draw.

By leveraging features such as the historical frequency of each number and its most recent appearance, along with a binary 
representation of past wins and losses, this dataset offers a comprehensive foundation for applying various machine learning 
models, including logistic regression, decision trees, and ensemble methods.

The transformation into this format opens new avenues for predictive analytics in the field of lottery number prediction, 
although it's important to remember the inherently random nature of lottery draws.


**References:**

[Include any relevant references or publications related to the dataset or methods used in its preparation]
