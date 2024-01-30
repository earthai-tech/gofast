.. _sjr_bet_dataset:

Ghanaian Lottery Dataset ("SJR bet")
-------------------------------------

**Data Set Characteristics:**

    :Number of Instances: 265
    :Number of Attributes: 4 (including both numerical and categorical attributes)
    :Attribute Information:
        - date: The date of the lottery draw
        - location: The location where the lottery draw occurred
        - win number: The winning numbers for the draw
        - lost number: The losing numbers for the draw
        
    :Summary Statistics:
    
    | Attribute         | Count | Unique | Top            | Frequency | Missing Values |
    |-------------------|-------|--------|----------------|-----------|----------------|
    | date              | 265   | 263    | 2018-12-21     | 2         | 0              |
    | location          | 265   | 6      | GH-BONANZA     | 179       | 0              |
    | win number        | 249   | 249    | 46-48-79-43-20 | 1         | 16             |
    | lost number       | 249   | 248    | 35-61-45-31-42 | 2         | 16             |

    :Missing Attribute Values:
    - win number: 6.04%
    - lost number: 6.04%
    [Other attributes have no missing values]

    :Class Distribution: The dataset is primarily used for understanding and analyzing lottery outcomes, focusing on patterns in winning and losing numbers.
    :Creator: K K.L. Laurent (etanoyau@gmail.com/ lkouao@csu.edu.cn) 
    :Donor: Sekongo N. Jean-Rene, CEO of Bouake Department of Mining operations, Bouake, Cote d'Ivoire (jeanrenesekongo@yahoo.fr)
    :Date: January 2024
	
**Dataset Description:**

The "SJR bet" dataset encapsulates the intriguing world of the Ghanaian lottery, a popular game of chance in Ghana, West Africa. 
This dataset uniquely captures the mechanics of the lottery, where participants propose ten numbers, grouped in five pairs 
(e.g., 00-11-23-33-44). These numbers are specifically even-numbered pairs, meaning they are taken two by two from the range 
00-99. 
Each weekend, the lottery draw reveals five winning and five losing numbers. To emerge victorious in this game, a bettor's selection 
of ten numbers (grouped as five pairs) must include at least one pair that matches any of the pairs in the winning set. The order 
of the numbers in the pair is irrelevant. Additionally, none of the bettor's pairs should correspond with any pair in the losing 
set. For instance, on 2018-05-04, the winning numbers were 46-48-79-43-20, and the losing numbers were 63-86-82-37-19. A player 
who chose pairs like 46-48 or 79-20 would win, while selections like 00-10 or 56-45, which do not match any winning pairs, would 
lose. Moreover, if any number in a player's pair appears in the losing numbers, the player is automatically disqualified.

The dataset illustrates the exciting dynamics of the lottery – the more pairs a player matches with the winning numbers, the 
greater their rewards. This system creates a thrilling experience where players aspire to match as many pairs as possible with
the winning numbers to maximize their winnings.

The "SJR bet" offers a comprehensive view of the lottery outcomes, providing insights into patterns and trends that could be 
valuable for statistical analysis, predictive modeling, and understanding the probabilities involved in this popular Ghanaian 
game.

** Important Notes:** 

The "SJR bet" dataset is modified and loaded for educational purpose that imply time series and trend.  And it's important to consider 
the ethical implications and legal boundaries around using predictive models for gambling or betting purposes.

.. topic:: References

   - Liu, J., Liu, W., Blanchard Allechy, F., Zheng, Z., Liu, R., & Kouadio, K.L. (2024). Machine learning-based techniques for land subsidence simulation in an urban area. Journal of Environmental Management, 352, 120078. https://doi.org/10.1016/j.jenvman.2024.120078
   - Kouadio, K.L., Liu, J., Liu, R., Wang, Y., & Liu, W. (2024). Earth Science Informatics. https://doi.org/10.1007/s12145-024-01236-3



[Relevant references or publications related to the dataset here]

