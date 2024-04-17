## USwB - Data Analysis & Prediction with Apache Spark, Databricks, MLLib {#uswb---data-analysis--prediction-with-apache-spark-databricks-mllib}

#### Author: Martyna Pitera

## CHECK THIS OUT ON DATABRICKS --> <https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/173542347700804/2929076465318146/6176203754563543/latest.html>

The project was carried out using Apache Spark on Databricks, utilizing
Python and SQL.

The goal of this project is to analyze the Body Fat Dataset and generate
predictive insights. 
(dataset - <https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset>)

#### The dataset contains of:

1.  Density determined from underwater weighing
2.  Percent body fat from Siri\'s (1956) equation
3.  Age (years)
4.  Weight (lbs)
5.  Height (inches)
6.  Neck circumference (cm)
7.  Chest circumference (cm)
8.  Abdomen circumference (cm)
9.  Hip circumference (cm)
10. Thigh circumference (cm)
11. Knee circumference (cm)
12. Ankle circumference (cm)
13. Biceps (extended) circumference (cm)
14. Forearm circumference (cm)
15. Wrist circumference (cm)
 
#### The project involved:

1.  Loading and preprocessing of the dataset
2.  Statistical analysis of the data
3.  Exploratory Data Analysis to uncover patterns and insights
4.  Correlation Analysis to understand relationships between variables
5.  Utilizing tree models to predict Body Fat percentage
    The Root Mean Squared Error (RMSE) for each model on the test data was:

-   Linear Regression: 0.622103
-   Decision Tree Regression: 0.96897
-   Gradient-Boosted Tree Regression: 0.891016 These results highlight
    the effectiveness of the Linear Regression model in predicting Body
    Fat percentage, outperforming both Linear Decision Tree Regression
    and Gradient-Boosted Tree Regression models.
