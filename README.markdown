## USwB - Data Analysis & Prediction with Apache Spark, Databricks, MLLib {#uswb---data-analysis--prediction-with-apache-spark-databricks-mllib}

#### Author: Martyna Pitera

The project was carried out using Apache Spark on Databricks, utilizing
Python and SQL.

The goal of this project is to analyze the Body Fat Dataset and generate
predictive insights. (dataset -
<https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset>)

This dataset contains of:

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
:::

#### Loading the data into dataframe
``` python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

spark = SparkSession.builder.appName("BodyFat").getOrCreate()

bodyfatSchema = StructType([
    StructField("Density", DoubleType(), True),
    StructField("BodyFat", DoubleType(), True),
    StructField("Age", IntegerType(), True),
    StructField("Weight", DoubleType(), True),
    StructField("Height", DoubleType(), True),
    StructField("Neck", DoubleType(), True),
    StructField("Chest", DoubleType(), True),
    StructField("Abdomen", DoubleType(), True),
    StructField("Hip", DoubleType(), True),
    StructField("Thigh", DoubleType(), True),
    StructField("Knee", DoubleType(), True),
    StructField("Ankle", DoubleType(), True),
    StructField("Biceps", DoubleType(), True),
    StructField("Forearm", DoubleType(), True),
    StructField("Wrist", DoubleType(), True),
])

df = spark.read.schema(bodyfatSchema).option("header", "true").csv("/FileStore/tables/bodyfat-3.csv")

df.show()
```


    +-------+-------+---+------+------+----+-----+-------+-----+-----+----+-----+------+-------+-----+
    |Density|BodyFat|Age|Weight|Height|Neck|Chest|Abdomen|  Hip|Thigh|Knee|Ankle|Biceps|Forearm|Wrist|
    +-------+-------+---+------+------+----+-----+-------+-----+-----+----+-----+------+-------+-----+
    | 1.0708|   12.3| 23|154.25| 67.75|36.2| 93.1|   85.2| 94.5| 59.0|37.3| 21.9|  32.0|   27.4| 17.1|
    | 1.0853|    6.1| 22|173.25| 72.25|38.5| 93.6|   83.0| 98.7| 58.7|37.3| 23.4|  30.5|   28.9| 18.2|
    | 1.0414|   25.3| 22| 154.0| 66.25|34.0| 95.8|   87.9| 99.2| 59.6|38.9| 24.0|  28.8|   25.2| 16.6|
    | 1.0751|   10.4| 26|184.75| 72.25|37.4|101.8|   86.4|101.2| 60.1|37.3| 22.8|  32.4|   29.4| 18.2|
    |  1.034|   28.7| 24|184.25| 71.25|34.4| 97.3|  100.0|101.9| 63.2|42.2| 24.0|  32.2|   27.7| 17.7|
    | 1.0502|   20.9| 24|210.25| 74.75|39.0|104.5|   94.4|107.8| 66.0|42.0| 25.6|  35.7|   30.6| 18.8|
    | 1.0549|   19.2| 26| 181.0| 69.75|36.4|105.1|   90.7|100.3| 58.4|38.3| 22.9|  31.9|   27.8| 17.7|
    | 1.0704|   12.4| 25| 176.0|  72.5|37.8| 99.6|   88.5| 97.1| 60.0|39.4| 23.2|  30.5|   29.0| 18.8|
    |   1.09|    4.1| 25| 191.0|  74.0|38.1|100.9|   82.5| 99.9| 62.9|38.3| 23.8|  35.9|   31.1| 18.2|
    | 1.0722|   11.7| 23|198.25|  73.5|42.1| 99.6|   88.6|104.1| 63.1|41.7| 25.0|  35.6|   30.0| 19.2|
    |  1.083|    7.1| 26|186.25|  74.5|38.5|101.5|   83.6| 98.2| 59.7|39.7| 25.2|  32.8|   29.4| 18.5|
    | 1.0812|    7.8| 27| 216.0|  76.0|39.4|103.6|   90.9|107.7| 66.2|39.2| 25.9|  37.2|   30.2| 19.0|
    | 1.0513|   20.8| 32| 180.5|  69.5|38.4|102.0|   91.6|103.9| 63.4|38.3| 21.5|  32.5|   28.6| 17.7|
    | 1.0505|   21.2| 30|205.25| 71.25|39.4|104.1|  101.8|108.6| 66.0|41.5| 23.7|  36.9|   31.6| 18.8|
    | 1.0484|   22.1| 35|187.75|  69.5|40.5|101.3|   96.4|100.1| 69.0|39.0| 23.1|  36.1|   30.5| 18.2|
    | 1.0512|   20.9| 35|162.75|  66.0|36.4| 99.1|   92.8| 99.2| 63.1|38.7| 21.7|  31.1|   26.4| 16.9|
    | 1.0333|   29.0| 34|195.75|  71.0|38.9|101.9|   96.4|105.2| 64.8|40.8| 23.1|  36.2|   30.8| 17.3|
    | 1.0468|   22.9| 32|209.25|  71.0|42.1|107.6|   97.5|107.0| 66.9|40.0| 24.4|  38.2|   31.6| 19.3|
    | 1.0622|   16.0| 28|183.75| 67.75|38.0|106.8|   89.6|102.4| 64.2|38.7| 22.9|  37.2|   30.5| 18.5|
    |  1.061|   16.5| 33|211.75|  73.5|40.0|106.2|  100.5|109.0| 65.8|40.6| 24.0|  37.1|   30.1| 18.2|
    +-------+-------+---+------+------+----+-----+-------+-----+-----+----+-----+------+-------+-----+
    only showing top 20 rows

# Number of rows in dataframe
df.count()
```
#### Converting inches to meters ang lbs to kgs
``` python
from pyspark.sql import functions as F

# Convert height from inches to meters
df = df.withColumn("Height", F.col("Height") * 0.0254)  # 1 inch = 0.0254 meters

# Convert weight from lbs to kg
df = df.withColumn("Weight", F.col("Weight") * 0.453592)  # 1 lb = 0.453592 kg
```

``` python
display(df.describe())
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>summary</th><th>Density</th><th>BodyFat</th><th>Age</th><th>Weight</th><th>Height</th><th>Neck</th><th>Chest</th><th>Abdomen</th><th>Hip</th><th>Thigh</th><th>Knee</th><th>Ankle</th><th>Biceps</th><th>Forearm</th><th>Wrist</th></tr></thead><tbody><tr><td>count</td><td>252</td><td>252</td><td>252</td><td>252</td><td>252</td><td>252</td><td>252</td><td>252</td><td>252</td><td>252</td><td>252</td><td>252</td><td>252</td><td>252</td><td>252</td></tr><tr><td>mean</td><td>1.055573809523809</td><td>19.15079365079365</td><td>44.88492063492063</td><td>81.15867860476192</td><td>1.7817797619047617</td><td>37.99206349206346</td><td>100.82420634920639</td><td>92.55595238095235</td><td>99.90476190476186</td><td>59.405952380952364</td><td>38.59047619047622</td><td>23.10238095238097</td><td>32.273412698412706</td><td>28.663888888888888</td><td>18.229761904761904</td></tr><tr><td>stddev</td><td>0.01903143417152082</td><td>8.368740413029712</td><td>12.602039722717862</td><td>13.330687810724323</td><td>0.09303653700708002</td><td>2.4309132340195085</td><td>8.43047553192002</td><td>10.783076801381702</td><td>7.164057666842286</td><td>5.249952028401046</td><td>2.4118045870187563</td><td>1.6948933981786372</td><td>3.021273751250864</td><td>2.020691165026929</td><td>0.9335849289587025</td></tr><tr><td>min</td><td>0.995</td><td>0.0</td><td>22</td><td>53.750652</td><td>0.7493</td><td>31.1</td><td>79.3</td><td>69.4</td><td>85.0</td><td>47.2</td><td>33.0</td><td>19.1</td><td>24.8</td><td>21.0</td><td>15.8</td></tr><tr><td>max</td><td>1.1089</td><td>47.5</td><td>81</td><td>164.72193479999999</td><td>1.97485</td><td>51.2</td><td>136.2</td><td>148.1</td><td>147.7</td><td>87.3</td><td>49.1</td><td>33.9</td><td>45.0</td><td>34.9</td><td>21.4</td></tr></tbody></table></div>
```
 
``` python
df.printSchema()
```

::: {.output .stream .stdout}
    root
     |-- Density: double (nullable = true)
     |-- BodyFat: double (nullable = true)
     |-- Age: integer (nullable = true)
     |-- Weight: double (nullable = true)
     |-- Height: double (nullable = true)
     |-- Neck: double (nullable = true)
     |-- Chest: double (nullable = true)
     |-- Abdomen: double (nullable = true)
     |-- Hip: double (nullable = true)
     |-- Thigh: double (nullable = true)
     |-- Knee: double (nullable = true)
     |-- Ankle: double (nullable = true)
     |-- Biceps: double (nullable = true)
     |-- Forearm: double (nullable = true)
     |-- Wrist: double (nullable = true)

## Exploratory Data Analysis

``` python
# Creating temporary view
df.createOrReplaceTempView('BodyFat')
```

#### Histogram of Body Fat

``` python
%sql

SELECT BodyFat FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>BodyFat</th></tr></thead><tbody><tr><td>12.3</td></tr><tr><td>6.1</td></tr><tr><td>25.3</td></tr><tr><td>10.4</td></tr><tr><td>28.7</td></tr><tr><td>20.9</td></tr><tr><td>19.2</td></tr><tr><td>12.4</td></tr><tr><td>4.1</td></tr><tr><td>11.7</td></tr><tr><td>7.1</td></tr><tr><td>7.8</td></tr><tr><td>20.8</td></tr><tr><td>21.2</td></tr><tr><td>22.1</td></tr><tr><td>20.9</td></tr><tr><td>29.0</td></tr><tr><td>22.9</td></tr><tr><td>16.0</td></tr><tr><td>16.5</td></tr><tr><td>19.1</td></tr><tr><td>15.2</td></tr><tr><td>15.6</td></tr><tr><td>17.7</td></tr><tr><td>14.0</td></tr><tr><td>3.7</td></tr><tr><td>7.9</td></tr><tr><td>22.9</td></tr><tr><td>3.7</td></tr><tr><td>8.8</td></tr><tr><td>11.9</td></tr><tr><td>5.7</td></tr><tr><td>11.8</td></tr><tr><td>21.3</td></tr><tr><td>32.3</td></tr><tr><td>40.1</td></tr><tr><td>24.2</td></tr><tr><td>28.4</td></tr><tr><td>35.2</td></tr><tr><td>32.6</td></tr><tr><td>34.5</td></tr><tr><td>32.9</td></tr><tr><td>31.6</td></tr><tr><td>32.0</td></tr><tr><td>7.7</td></tr><tr><td>13.9</td></tr><tr><td>10.8</td></tr><tr><td>5.6</td></tr><tr><td>13.6</td></tr><tr><td>4.0</td></tr><tr><td>10.2</td></tr><tr><td>6.6</td></tr><tr><td>8.0</td></tr><tr><td>6.3</td></tr><tr><td>3.9</td></tr><tr><td>22.6</td></tr><tr><td>20.4</td></tr><tr><td>28.0</td></tr><tr><td>31.5</td></tr><tr><td>24.6</td></tr><tr><td>26.1</td></tr><tr><td>29.8</td></tr><tr><td>30.7</td></tr><tr><td>25.8</td></tr><tr><td>32.3</td></tr><tr><td>30.0</td></tr><tr><td>21.5</td></tr><tr><td>13.8</td></tr><tr><td>6.3</td></tr><tr><td>12.9</td></tr><tr><td>24.3</td></tr><tr><td>8.8</td></tr><tr><td>8.5</td></tr><tr><td>13.5</td></tr><tr><td>11.8</td></tr><tr><td>18.5</td></tr><tr><td>8.8</td></tr><tr><td>22.2</td></tr><tr><td>21.5</td></tr><tr><td>18.8</td></tr><tr><td>31.4</td></tr><tr><td>26.8</td></tr><tr><td>18.4</td></tr><tr><td>27.0</td></tr><tr><td>27.0</td></tr><tr><td>26.6</td></tr><tr><td>14.9</td></tr><tr><td>23.1</td></tr><tr><td>8.3</td></tr><tr><td>14.1</td></tr><tr><td>20.5</td></tr><tr><td>18.2</td></tr><tr><td>8.5</td></tr><tr><td>24.9</td></tr><tr><td>9.0</td></tr><tr><td>17.4</td></tr><tr><td>9.6</td></tr><tr><td>11.3</td></tr><tr><td>17.8</td></tr><tr><td>22.2</td></tr><tr><td>21.2</td></tr><tr><td>20.4</td></tr><tr><td>20.1</td></tr><tr><td>22.3</td></tr><tr><td>25.4</td></tr><tr><td>18.0</td></tr><tr><td>19.3</td></tr><tr><td>18.3</td></tr><tr><td>17.3</td></tr><tr><td>21.4</td></tr><tr><td>19.7</td></tr><tr><td>28.0</td></tr><tr><td>22.1</td></tr><tr><td>21.3</td></tr><tr><td>26.7</td></tr><tr><td>16.7</td></tr><tr><td>20.1</td></tr><tr><td>13.9</td></tr><tr><td>25.8</td></tr><tr><td>18.1</td></tr><tr><td>27.9</td></tr><tr><td>25.3</td></tr><tr><td>14.7</td></tr><tr><td>16.0</td></tr><tr><td>13.8</td></tr><tr><td>17.5</td></tr><tr><td>27.2</td></tr><tr><td>17.4</td></tr><tr><td>20.8</td></tr><tr><td>14.9</td></tr><tr><td>18.1</td></tr><tr><td>22.7</td></tr><tr><td>23.6</td></tr><tr><td>26.1</td></tr><tr><td>24.4</td></tr><tr><td>27.1</td></tr><tr><td>21.8</td></tr><tr><td>29.4</td></tr><tr><td>22.4</td></tr><tr><td>20.4</td></tr><tr><td>24.9</td></tr><tr><td>18.3</td></tr><tr><td>23.3</td></tr><tr><td>9.4</td></tr><tr><td>10.3</td></tr><tr><td>14.2</td></tr><tr><td>19.2</td></tr><tr><td>29.6</td></tr><tr><td>5.3</td></tr><tr><td>25.2</td></tr><tr><td>9.4</td></tr><tr><td>19.6</td></tr><tr><td>10.1</td></tr><tr><td>16.5</td></tr><tr><td>21.0</td></tr><tr><td>17.3</td></tr><tr><td>31.2</td></tr><tr><td>10.0</td></tr><tr><td>12.5</td></tr><tr><td>22.5</td></tr><tr><td>9.4</td></tr><tr><td>14.6</td></tr><tr><td>13.0</td></tr><tr><td>15.1</td></tr><tr><td>27.3</td></tr><tr><td>19.2</td></tr><tr><td>21.8</td></tr><tr><td>20.3</td></tr><tr><td>34.3</td></tr><tr><td>16.5</td></tr><tr><td>3.0</td></tr><tr><td>0.7</td></tr><tr><td>20.5</td></tr><tr><td>16.9</td></tr><tr><td>25.3</td></tr><tr><td>9.9</td></tr><tr><td>13.1</td></tr><tr><td>29.9</td></tr><tr><td>22.5</td></tr><tr><td>16.9</td></tr><tr><td>26.6</td></tr><tr><td>0.0</td></tr><tr><td>11.5</td></tr><tr><td>12.1</td></tr><tr><td>17.5</td></tr><tr><td>8.6</td></tr><tr><td>23.6</td></tr><tr><td>20.4</td></tr><tr><td>20.5</td></tr><tr><td>24.4</td></tr><tr><td>11.4</td></tr><tr><td>38.1</td></tr><tr><td>15.9</td></tr><tr><td>24.7</td></tr><tr><td>22.8</td></tr><tr><td>25.5</td></tr><tr><td>22.0</td></tr><tr><td>17.7</td></tr><tr><td>6.6</td></tr><tr><td>23.6</td></tr><tr><td>12.2</td></tr><tr><td>22.1</td></tr><tr><td>28.7</td></tr><tr><td>6.0</td></tr><tr><td>34.8</td></tr><tr><td>16.6</td></tr><tr><td>32.9</td></tr><tr><td>32.8</td></tr><tr><td>9.6</td></tr><tr><td>10.8</td></tr><tr><td>7.1</td></tr><tr><td>27.2</td></tr><tr><td>19.5</td></tr><tr><td>18.7</td></tr><tr><td>19.5</td></tr><tr><td>47.5</td></tr><tr><td>13.6</td></tr><tr><td>7.5</td></tr><tr><td>24.5</td></tr><tr><td>15.0</td></tr><tr><td>12.4</td></tr><tr><td>26.0</td></tr><tr><td>11.5</td></tr><tr><td>5.2</td></tr><tr><td>10.9</td></tr><tr><td>12.5</td></tr><tr><td>14.8</td></tr><tr><td>25.2</td></tr><tr><td>14.9</td></tr><tr><td>17.0</td></tr><tr><td>10.6</td></tr><tr><td>16.1</td></tr><tr><td>15.4</td></tr><tr><td>26.7</td></tr><tr><td>25.8</td></tr><tr><td>18.6</td></tr><tr><td>24.8</td></tr><tr><td>27.3</td></tr><tr><td>12.4</td></tr><tr><td>29.9</td></tr><tr><td>17.0</td></tr><tr><td>35.0</td></tr><tr><td>30.4</td></tr><tr><td>32.6</td></tr><tr><td>29.0</td></tr><tr><td>15.2</td></tr><tr><td>30.2</td></tr><tr><td>11.0</td></tr><tr><td>33.6</td></tr><tr><td>29.3</td></tr><tr><td>26.0</td></tr><tr><td>31.9</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
 
According to the American Journal of Clinical Nutrition, there are
healthy body fat percentages based on your age. For people aged 20 to
39, women should aim for 21% to 32% of body fat. Men should have 8% to
19%. For people 40 to 59, women should fall between 23% to 33% and men
should fall around 11% to 21%. If you're aged 60 to 79, women should
have 24% to 35% body fat and men should have 13% to 24%.

Women naturally have a higher body fat percentage than men. Their body
fat will also naturally increase as they age.

(source-
<https://www.webmd.com/fitness-exercise/what-is-body-composition>)

#### Histogram of Age

``` python
%sql

SELECT Age FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Age</th></tr></thead><tbody><tr><td>23</td></tr><tr><td>22</td></tr><tr><td>22</td></tr><tr><td>26</td></tr><tr><td>24</td></tr><tr><td>24</td></tr><tr><td>26</td></tr><tr><td>25</td></tr><tr><td>25</td></tr><tr><td>23</td></tr><tr><td>26</td></tr><tr><td>27</td></tr><tr><td>32</td></tr><tr><td>30</td></tr><tr><td>35</td></tr><tr><td>35</td></tr><tr><td>34</td></tr><tr><td>32</td></tr><tr><td>28</td></tr><tr><td>33</td></tr><tr><td>28</td></tr><tr><td>28</td></tr><tr><td>31</td></tr><tr><td>32</td></tr><tr><td>28</td></tr><tr><td>27</td></tr><tr><td>34</td></tr><tr><td>31</td></tr><tr><td>27</td></tr><tr><td>29</td></tr><tr><td>32</td></tr><tr><td>29</td></tr><tr><td>27</td></tr><tr><td>41</td></tr><tr><td>41</td></tr><tr><td>49</td></tr><tr><td>40</td></tr><tr><td>50</td></tr><tr><td>46</td></tr><tr><td>50</td></tr><tr><td>45</td></tr><tr><td>44</td></tr><tr><td>48</td></tr><tr><td>41</td></tr><tr><td>39</td></tr><tr><td>43</td></tr><tr><td>40</td></tr><tr><td>39</td></tr><tr><td>45</td></tr><tr><td>47</td></tr><tr><td>47</td></tr><tr><td>40</td></tr><tr><td>51</td></tr><tr><td>49</td></tr><tr><td>42</td></tr><tr><td>54</td></tr><tr><td>58</td></tr><tr><td>62</td></tr><tr><td>54</td></tr><tr><td>61</td></tr><tr><td>62</td></tr><tr><td>56</td></tr><tr><td>54</td></tr><tr><td>61</td></tr><tr><td>57</td></tr><tr><td>55</td></tr><tr><td>54</td></tr><tr><td>55</td></tr><tr><td>54</td></tr><tr><td>55</td></tr><tr><td>62</td></tr><tr><td>55</td></tr><tr><td>56</td></tr><tr><td>55</td></tr><tr><td>61</td></tr><tr><td>61</td></tr><tr><td>57</td></tr><tr><td>69</td></tr><tr><td>81</td></tr><tr><td>66</td></tr><tr><td>67</td></tr><tr><td>64</td></tr><tr><td>64</td></tr><tr><td>70</td></tr><tr><td>72</td></tr><tr><td>67</td></tr><tr><td>72</td></tr><tr><td>64</td></tr><tr><td>46</td></tr><tr><td>48</td></tr><tr><td>46</td></tr><tr><td>44</td></tr><tr><td>47</td></tr><tr><td>46</td></tr><tr><td>47</td></tr><tr><td>53</td></tr><tr><td>38</td></tr><tr><td>50</td></tr><tr><td>46</td></tr><tr><td>47</td></tr><tr><td>49</td></tr><tr><td>48</td></tr><tr><td>41</td></tr><tr><td>49</td></tr><tr><td>43</td></tr><tr><td>43</td></tr><tr><td>43</td></tr><tr><td>52</td></tr><tr><td>43</td></tr><tr><td>40</td></tr><tr><td>43</td></tr><tr><td>43</td></tr><tr><td>47</td></tr><tr><td>42</td></tr><tr><td>48</td></tr><tr><td>40</td></tr><tr><td>48</td></tr><tr><td>51</td></tr><tr><td>40</td></tr><tr><td>44</td></tr><tr><td>52</td></tr><tr><td>44</td></tr><tr><td>40</td></tr><tr><td>47</td></tr><tr><td>50</td></tr><tr><td>46</td></tr><tr><td>42</td></tr><tr><td>43</td></tr><tr><td>40</td></tr><tr><td>42</td></tr><tr><td>49</td></tr><tr><td>40</td></tr><tr><td>47</td></tr><tr><td>50</td></tr><tr><td>41</td></tr><tr><td>44</td></tr><tr><td>39</td></tr><tr><td>43</td></tr><tr><td>40</td></tr><tr><td>49</td></tr><tr><td>40</td></tr><tr><td>40</td></tr><tr><td>52</td></tr><tr><td>23</td></tr><tr><td>23</td></tr><tr><td>24</td></tr><tr><td>24</td></tr><tr><td>25</td></tr><tr><td>25</td></tr><tr><td>26</td></tr><tr><td>26</td></tr><tr><td>26</td></tr><tr><td>27</td></tr><tr><td>27</td></tr><tr><td>27</td></tr><tr><td>28</td></tr><tr><td>28</td></tr><tr><td>28</td></tr><tr><td>30</td></tr><tr><td>31</td></tr><tr><td>31</td></tr><tr><td>33</td></tr><tr><td>33</td></tr><tr><td>34</td></tr><tr><td>34</td></tr><tr><td>35</td></tr><tr><td>35</td></tr><tr><td>35</td></tr><tr><td>35</td></tr><tr><td>35</td></tr><tr><td>35</td></tr><tr><td>35</td></tr><tr><td>35</td></tr><tr><td>36</td></tr><tr><td>36</td></tr><tr><td>37</td></tr><tr><td>37</td></tr><tr><td>37</td></tr><tr><td>38</td></tr><tr><td>39</td></tr><tr><td>39</td></tr><tr><td>40</td></tr><tr><td>40</td></tr><tr><td>40</td></tr><tr><td>40</td></tr><tr><td>40</td></tr><tr><td>41</td></tr><tr><td>41</td></tr><tr><td>41</td></tr><tr><td>41</td></tr><tr><td>41</td></tr><tr><td>42</td></tr><tr><td>42</td></tr><tr><td>42</td></tr><tr><td>42</td></tr><tr><td>42</td></tr><tr><td>42</td></tr><tr><td>42</td></tr><tr><td>42</td></tr><tr><td>43</td></tr><tr><td>43</td></tr><tr><td>43</td></tr><tr><td>43</td></tr><tr><td>44</td></tr><tr><td>44</td></tr><tr><td>44</td></tr><tr><td>44</td></tr><tr><td>47</td></tr><tr><td>47</td></tr><tr><td>47</td></tr><tr><td>49</td></tr><tr><td>49</td></tr><tr><td>49</td></tr><tr><td>50</td></tr><tr><td>50</td></tr><tr><td>51</td></tr><tr><td>51</td></tr><tr><td>51</td></tr><tr><td>52</td></tr><tr><td>53</td></tr><tr><td>54</td></tr><tr><td>54</td></tr><tr><td>54</td></tr><tr><td>55</td></tr><tr><td>55</td></tr><tr><td>55</td></tr><tr><td>55</td></tr><tr><td>55</td></tr><tr><td>56</td></tr><tr><td>56</td></tr><tr><td>57</td></tr><tr><td>57</td></tr><tr><td>58</td></tr><tr><td>58</td></tr><tr><td>60</td></tr><tr><td>62</td></tr><tr><td>62</td></tr><tr><td>63</td></tr><tr><td>64</td></tr><tr><td>65</td></tr><tr><td>65</td></tr><tr><td>65</td></tr><tr><td>66</td></tr><tr><td>67</td></tr><tr><td>67</td></tr><tr><td>68</td></tr><tr><td>69</td></tr><tr><td>70</td></tr><tr><td>72</td></tr><tr><td>72</td></tr><tr><td>72</td></tr><tr><td>74</td></tr></tbody></table></div>
```

#### Histogram of Weight
:::

``` python
%sql

SELECT Weight FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Weight</th></tr></thead><tbody><tr><td>69.966566</td></tr><tr><td>78.584814</td></tr><tr><td>69.853168</td></tr><tr><td>83.80112199999999</td></tr><tr><td>83.574326</td></tr><tr><td>95.367718</td></tr><tr><td>82.100152</td></tr><tr><td>79.83219199999999</td></tr><tr><td>86.636072</td></tr><tr><td>89.924614</td></tr><tr><td>84.48151</td></tr><tr><td>97.975872</td></tr><tr><td>81.873356</td></tr><tr><td>93.099758</td></tr><tr><td>85.161898</td></tr><tr><td>73.822098</td></tr><tr><td>88.790634</td></tr><tr><td>94.914126</td></tr><tr><td>83.34753</td></tr><tr><td>96.048106</td></tr><tr><td>81.192968</td></tr><tr><td>90.945196</td></tr><tr><td>63.616278</td></tr><tr><td>67.47181</td></tr><tr><td>68.60579</td></tr><tr><td>72.234526</td></tr><tr><td>59.647348</td></tr><tr><td>67.131616</td></tr><tr><td>60.441134</td></tr><tr><td>72.914914</td></tr><tr><td>82.553744</td></tr><tr><td>72.688118</td></tr><tr><td>76.203456</td></tr><tr><td>99.109852</td></tr><tr><td>112.150622</td></tr><tr><td>86.976266</td></tr><tr><td>91.738982</td></tr><tr><td>89.244226</td></tr><tr><td>164.72193479999999</td></tr><tr><td>92.079176</td></tr><tr><td>119.181298</td></tr><tr><td>92.98636</td></tr><tr><td>98.429464</td></tr><tr><td>96.161504</td></tr><tr><td>56.812398</td></tr><tr><td>74.502486</td></tr><tr><td>60.554532</td></tr><tr><td>67.358412</td></tr><tr><td>61.575114</td></tr><tr><td>57.83298</td></tr><tr><td>71.780934</td></tr><tr><td>63.162686</td></tr><tr><td>62.255502</td></tr><tr><td>69.28617799999999</td></tr><tr><td>61.80191</td></tr><tr><td>89.811216</td></tr><tr><td>82.326948</td></tr><tr><td>91.28538999999999</td></tr><tr><td>91.85238</td></tr><tr><td>81.533162</td></tr><tr><td>97.975872</td></tr><tr><td>81.07957</td></tr><tr><td>87.656654</td></tr><tr><td>80.739376</td></tr><tr><td>93.213156</td></tr><tr><td>83.234132</td></tr><tr><td>68.719188</td></tr><tr><td>70.193362</td></tr><tr><td>70.420158</td></tr><tr><td>71.100546</td></tr><tr><td>75.97666</td></tr><tr><td>66.564626</td></tr><tr><td>72.914914</td></tr><tr><td>56.699</td></tr><tr><td>64.863656</td></tr><tr><td>67.245014</td></tr><tr><td>73.7087</td></tr><tr><td>80.625978</td></tr><tr><td>73.14171</td></tr><tr><td>77.67763</td></tr><tr><td>74.27569</td></tr><tr><td>68.152198</td></tr><tr><td>86.295878</td></tr><tr><td>77.450834</td></tr><tr><td>76.203456</td></tr><tr><td>75.749864</td></tr><tr><td>71.554138</td></tr><tr><td>72.57472</td></tr><tr><td>80.172386</td></tr><tr><td>79.83219199999999</td></tr><tr><td>80.28578399999999</td></tr><tr><td>81.533162</td></tr><tr><td>74.956078</td></tr><tr><td>87.31645999999999</td></tr><tr><td>83.574326</td></tr><tr><td>101.83140399999999</td></tr><tr><td>85.61549</td></tr><tr><td>73.7087</td></tr><tr><td>70.987148</td></tr><tr><td>89.357624</td></tr><tr><td>90.038012</td></tr><tr><td>78.81161</td></tr><tr><td>78.358018</td></tr><tr><td>89.244226</td></tr><tr><td>80.28578399999999</td></tr><tr><td>75.069476</td></tr><tr><td>90.83179799999999</td></tr><tr><td>92.192574</td></tr><tr><td>87.996848</td></tr><tr><td>76.430252</td></tr><tr><td>77.450834</td></tr><tr><td>83.120734</td></tr><tr><td>80.852774</td></tr><tr><td>73.935496</td></tr><tr><td>79.491998</td></tr><tr><td>71.667536</td></tr><tr><td>80.399182</td></tr><tr><td>81.192968</td></tr><tr><td>86.636072</td></tr><tr><td>85.0485</td></tr><tr><td>93.666748</td></tr><tr><td>84.027918</td></tr><tr><td>72.688118</td></tr><tr><td>68.719188</td></tr><tr><td>73.028312</td></tr><tr><td>75.749864</td></tr><tr><td>80.51258</td></tr><tr><td>69.059382</td></tr><tr><td>87.203062</td></tr><tr><td>74.956078</td></tr><tr><td>77.904426</td></tr><tr><td>77.67763</td></tr><tr><td>89.357624</td></tr><tr><td>71.213944</td></tr><tr><td>76.31685399999999</td></tr><tr><td>84.368112</td></tr><tr><td>75.636466</td></tr><tr><td>85.161898</td></tr><tr><td>76.31685399999999</td></tr><tr><td>96.501698</td></tr><tr><td>80.172386</td></tr><tr><td>78.584814</td></tr><tr><td>75.749864</td></tr><tr><td>72.461322</td></tr><tr><td>85.34333480000001</td></tr><tr><td>70.760352</td></tr><tr><td>94.573932</td></tr><tr><td>93.666748</td></tr><tr><td>65.20385</td></tr><tr><td>101.151016</td></tr><tr><td>69.059382</td></tr><tr><td>109.655866</td></tr><tr><td>66.224432</td></tr><tr><td>71.100546</td></tr><tr><td>90.83179799999999</td></tr><tr><td>77.791028</td></tr><tr><td>93.326554</td></tr><tr><td>82.78054</td></tr><tr><td>61.915307999999996</td></tr><tr><td>80.399182</td></tr><tr><td>68.60579</td></tr><tr><td>88.904032</td></tr><tr><td>83.574326</td></tr><tr><td>63.50288</td></tr><tr><td>99.22325</td></tr><tr><td>98.429464</td></tr><tr><td>75.40967</td></tr><tr><td>101.944802</td></tr><tr><td>103.532374</td></tr><tr><td>78.358018</td></tr><tr><td>69.059382</td></tr><tr><td>57.039194</td></tr><tr><td>80.399182</td></tr><tr><td>79.94559</td></tr><tr><td>102.851986</td></tr><tr><td>65.884238</td></tr><tr><td>68.492392</td></tr><tr><td>109.42907</td></tr><tr><td>84.935102</td></tr><tr><td>106.480722</td></tr><tr><td>99.450046</td></tr><tr><td>53.750652</td></tr><tr><td>66.111034</td></tr><tr><td>72.234526</td></tr><tr><td>77.337436</td></tr><tr><td>75.97666</td></tr><tr><td>105.573538</td></tr><tr><td>95.481116</td></tr><tr><td>91.738982</td></tr><tr><td>83.91452</td></tr><tr><td>69.399576</td></tr><tr><td>110.789846</td></tr><tr><td>87.77005199999999</td></tr><tr><td>101.944802</td></tr><tr><td>73.822098</td></tr><tr><td>81.64656</td></tr><tr><td>70.87375</td></tr><tr><td>76.203456</td></tr><tr><td>75.863262</td></tr><tr><td>77.450834</td></tr><tr><td>80.852774</td></tr><tr><td>68.0388</td></tr><tr><td>90.945196</td></tr><tr><td>83.460928</td></tr><tr><td>101.151016</td></tr><tr><td>94.68733</td></tr><tr><td>75.296272</td></tr><tr><td>88.45044</td></tr><tr><td>72.80151599999999</td></tr><tr><td>72.461322</td></tr><tr><td>63.729676</td></tr><tr><td>98.08927</td></tr><tr><td>76.31685399999999</td></tr><tr><td>88.337042</td></tr><tr><td>78.358018</td></tr><tr><td>99.336648</td></tr><tr><td>67.698606</td></tr><tr><td>70.079964</td></tr><tr><td>90.378206</td></tr><tr><td>70.079964</td></tr><tr><td>69.512974</td></tr><tr><td>104.32616</td></tr><tr><td>73.368506</td></tr><tr><td>64.523462</td></tr><tr><td>81.533162</td></tr><tr><td>57.379388</td></tr><tr><td>76.883844</td></tr><tr><td>90.038012</td></tr><tr><td>79.151804</td></tr><tr><td>76.090058</td></tr><tr><td>67.018218</td></tr><tr><td>82.667142</td></tr><tr><td>79.605396</td></tr><tr><td>73.368506</td></tr><tr><td>71.554138</td></tr><tr><td>76.54365</td></tr><tr><td>86.862868</td></tr><tr><td>99.40468680000001</td></tr><tr><td>70.420158</td></tr><tr><td>86.069082</td></tr><tr><td>57.83298</td></tr><tr><td>101.83140399999999</td></tr><tr><td>106.25392599999999</td></tr><tr><td>103.305578</td></tr><tr><td>90.491604</td></tr><tr><td>70.533556</td></tr><tr><td>97.749076</td></tr><tr><td>60.894726</td></tr><tr><td>91.171992</td></tr><tr><td>84.708306</td></tr><tr><td>86.522674</td></tr><tr><td>94.12034</td></tr></tbody></table></div>
```

#### Histogram of Height

``` python
%sql

SELECT Height FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Height</th></tr></thead><tbody><tr><td>1.72085</td></tr><tr><td>1.8351499999999998</td></tr><tr><td>1.68275</td></tr><tr><td>1.8351499999999998</td></tr><tr><td>1.80975</td></tr><tr><td>1.89865</td></tr><tr><td>1.77165</td></tr><tr><td>1.8415</td></tr><tr><td>1.8796</td></tr><tr><td>1.8669</td></tr><tr><td>1.8922999999999999</td></tr><tr><td>1.9304</td></tr><tr><td>1.7652999999999999</td></tr><tr><td>1.80975</td></tr><tr><td>1.7652999999999999</td></tr><tr><td>1.6764</td></tr><tr><td>1.8034</td></tr><tr><td>1.8034</td></tr><tr><td>1.72085</td></tr><tr><td>1.8669</td></tr><tr><td>1.7271999999999998</td></tr><tr><td>1.77165</td></tr><tr><td>1.73355</td></tr><tr><td>1.778</td></tr><tr><td>1.72085</td></tr><tr><td>1.8160999999999998</td></tr><tr><td>1.7145</td></tr><tr><td>1.7145</td></tr><tr><td>1.64465</td></tr><tr><td>1.7526</td></tr><tr><td>1.8732499999999999</td></tr><tr><td>1.80975</td></tr><tr><td>1.80975</td></tr><tr><td>1.8034</td></tr><tr><td>1.8669</td></tr><tr><td>1.651</td></tr><tr><td>1.778</td></tr><tr><td>1.73355</td></tr><tr><td>1.8351499999999998</td></tr><tr><td>1.7018</td></tr><tr><td>1.7462499999999999</td></tr><tr><td>0.7493</td></tr><tr><td>1.778</td></tr><tr><td>1.8160999999999998</td></tr><tr><td>1.7271999999999998</td></tr><tr><td>1.86055</td></tr><tr><td>1.7145</td></tr><tr><td>1.80975</td></tr><tr><td>1.7399</td></tr><tr><td>1.69545</td></tr><tr><td>1.8351499999999998</td></tr><tr><td>1.7526</td></tr><tr><td>1.72085</td></tr><tr><td>1.8669</td></tr><tr><td>1.7145</td></tr><tr><td>1.8288</td></tr><tr><td>1.7271999999999998</td></tr><tr><td>1.7652999999999999</td></tr><tr><td>1.79705</td></tr><tr><td>1.67005</td></tr><tr><td>1.86055</td></tr><tr><td>1.7399</td></tr><tr><td>1.7843499999999999</td></tr><tr><td>1.7018</td></tr><tr><td>1.778</td></tr><tr><td>1.7145</td></tr><tr><td>1.79705</td></tr><tr><td>1.8160999999999998</td></tr><tr><td>1.75895</td></tr><tr><td>1.8160999999999998</td></tr><tr><td>1.8160999999999998</td></tr><tr><td>1.7462499999999999</td></tr><tr><td>1.8732499999999999</td></tr><tr><td>1.6256</td></tr><tr><td>1.67005</td></tr><tr><td>1.7145</td></tr><tr><td>1.7652999999999999</td></tr><tr><td>1.7399</td></tr><tr><td>1.7843499999999999</td></tr><tr><td>1.75895</td></tr><tr><td>1.72085</td></tr><tr><td>1.7081499999999998</td></tr><tr><td>1.84785</td></tr><tr><td>1.778</td></tr><tr><td>1.75895</td></tr><tr><td>1.7145</td></tr><tr><td>1.7081499999999998</td></tr><tr><td>1.67005</td></tr><tr><td>1.8415</td></tr><tr><td>1.8541999999999998</td></tr><tr><td>1.778</td></tr><tr><td>1.7652999999999999</td></tr><tr><td>1.7907</td></tr><tr><td>1.82245</td></tr><tr><td>1.8922999999999999</td></tr><tr><td>1.97485</td></tr><tr><td>1.86055</td></tr><tr><td>1.6890999999999998</td></tr><tr><td>1.73355</td></tr><tr><td>1.8288</td></tr><tr><td>1.8669</td></tr><tr><td>1.8288</td></tr><tr><td>1.80975</td></tr><tr><td>1.8732499999999999</td></tr><tr><td>1.75895</td></tr><tr><td>1.7399</td></tr><tr><td>1.8669</td></tr><tr><td>1.88595</td></tr><tr><td>1.9177</td></tr><tr><td>1.75895</td></tr><tr><td>1.7399</td></tr><tr><td>1.778</td></tr><tr><td>1.778</td></tr><tr><td>1.7843499999999999</td></tr><tr><td>1.82245</td></tr><tr><td>1.75895</td></tr><tr><td>1.84785</td></tr><tr><td>1.8288</td></tr><tr><td>1.8796</td></tr><tr><td>1.8351499999999998</td></tr><tr><td>1.8922999999999999</td></tr><tr><td>1.8160999999999998</td></tr><tr><td>1.7462499999999999</td></tr><tr><td>1.69545</td></tr><tr><td>1.6890999999999998</td></tr><tr><td>1.7018</td></tr><tr><td>1.7462499999999999</td></tr><tr><td>1.72085</td></tr><tr><td>1.86055</td></tr><tr><td>1.77165</td></tr><tr><td>1.8160999999999998</td></tr><tr><td>1.7907</td></tr><tr><td>1.86055</td></tr><tr><td>1.69545</td></tr><tr><td>1.7652999999999999</td></tr><tr><td>1.77165</td></tr><tr><td>1.79705</td></tr><tr><td>1.8796</td></tr><tr><td>1.80975</td></tr><tr><td>1.905</td></tr><tr><td>1.8034</td></tr><tr><td>1.7652999999999999</td></tr><tr><td>1.72085</td></tr><tr><td>1.8351499999999998</td></tr><tr><td>1.9685</td></tr><tr><td>1.79705</td></tr><tr><td>1.84785</td></tr><tr><td>1.77165</td></tr><tr><td>1.8415</td></tr><tr><td>1.7843499999999999</td></tr><tr><td>1.7526</td></tr><tr><td>1.8922999999999999</td></tr><tr><td>1.8351499999999998</td></tr><tr><td>1.7081499999999998</td></tr><tr><td>1.8669</td></tr><tr><td>1.9113499999999999</td></tr><tr><td>1.7526</td></tr><tr><td>1.8351499999999998</td></tr><tr><td>1.7462499999999999</td></tr><tr><td>1.8160999999999998</td></tr><tr><td>1.8351499999999998</td></tr><tr><td>1.8541999999999998</td></tr><tr><td>1.7462499999999999</td></tr><tr><td>1.7907</td></tr><tr><td>1.8288</td></tr><tr><td>1.8732499999999999</td></tr><tr><td>1.7271999999999998</td></tr><tr><td>1.8351499999999998</td></tr><tr><td>1.7652999999999999</td></tr><tr><td>1.7652999999999999</td></tr><tr><td>1.72085</td></tr><tr><td>1.6637</td></tr><tr><td>1.8034</td></tr><tr><td>1.8160999999999998</td></tr><tr><td>1.82245</td></tr><tr><td>1.75895</td></tr><tr><td>1.7018</td></tr><tr><td>1.8160999999999998</td></tr><tr><td>1.75895</td></tr><tr><td>1.8922999999999999</td></tr><tr><td>1.88595</td></tr><tr><td>1.7271999999999998</td></tr><tr><td>1.7081499999999998</td></tr><tr><td>1.77165</td></tr><tr><td>1.88595</td></tr><tr><td>1.8160999999999998</td></tr><tr><td>1.88595</td></tr><tr><td>1.8288</td></tr><tr><td>1.8415</td></tr><tr><td>1.73355</td></tr><tr><td>1.75895</td></tr><tr><td>1.9304</td></tr><tr><td>1.7907</td></tr><tr><td>1.89865</td></tr><tr><td>1.84785</td></tr><tr><td>1.73355</td></tr><tr><td>1.7526</td></tr><tr><td>1.8160999999999998</td></tr><tr><td>1.84785</td></tr><tr><td>1.7145</td></tr><tr><td>1.7843499999999999</td></tr><tr><td>1.75895</td></tr><tr><td>1.8160999999999998</td></tr><tr><td>1.8796</td></tr><tr><td>1.77165</td></tr><tr><td>1.8541999999999998</td></tr><tr><td>1.6637</td></tr><tr><td>1.8415</td></tr><tr><td>1.7843499999999999</td></tr><tr><td>1.79705</td></tr><tr><td>1.7271999999999998</td></tr><tr><td>1.8922999999999999</td></tr><tr><td>1.82245</td></tr><tr><td>1.79705</td></tr><tr><td>1.8541999999999998</td></tr><tr><td>1.6256</td></tr><tr><td>1.77165</td></tr><tr><td>1.778</td></tr><tr><td>1.82245</td></tr><tr><td>1.75895</td></tr><tr><td>1.7907</td></tr><tr><td>1.8351499999999998</td></tr><tr><td>1.7145</td></tr><tr><td>1.7081499999999998</td></tr><tr><td>1.7462499999999999</td></tr><tr><td>1.69545</td></tr><tr><td>1.73355</td></tr><tr><td>1.88595</td></tr><tr><td>1.7652999999999999</td></tr><tr><td>1.7399</td></tr><tr><td>1.67005</td></tr><tr><td>1.82245</td></tr><tr><td>1.8160999999999998</td></tr><tr><td>1.7081499999999998</td></tr><tr><td>1.7145</td></tr><tr><td>1.7145</td></tr><tr><td>1.8351499999999998</td></tr><tr><td>1.7652999999999999</td></tr><tr><td>1.7652999999999999</td></tr><tr><td>1.67005</td></tr><tr><td>1.67005</td></tr><tr><td>1.73355</td></tr><tr><td>1.8288</td></tr><tr><td>1.84785</td></tr><tr><td>1.7399</td></tr><tr><td>1.75895</td></tr><tr><td>1.7907</td></tr><tr><td>1.7018</td></tr><tr><td>1.77165</td></tr><tr><td>1.6764</td></tr><tr><td>1.7907</td></tr><tr><td>1.778</td></tr></tbody></table></div>
```

#### Histogram of Neck Circumference

``` python
%sql

SELECT Neck FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Neck</th></tr></thead><tbody><tr><td>36.2</td></tr><tr><td>38.5</td></tr><tr><td>34.0</td></tr><tr><td>37.4</td></tr><tr><td>34.4</td></tr><tr><td>39.0</td></tr><tr><td>36.4</td></tr><tr><td>37.8</td></tr><tr><td>38.1</td></tr><tr><td>42.1</td></tr><tr><td>38.5</td></tr><tr><td>39.4</td></tr><tr><td>38.4</td></tr><tr><td>39.4</td></tr><tr><td>40.5</td></tr><tr><td>36.4</td></tr><tr><td>38.9</td></tr><tr><td>42.1</td></tr><tr><td>38.0</td></tr><tr><td>40.0</td></tr><tr><td>39.1</td></tr><tr><td>41.3</td></tr><tr><td>33.9</td></tr><tr><td>35.5</td></tr><tr><td>34.5</td></tr><tr><td>35.7</td></tr><tr><td>36.2</td></tr><tr><td>38.8</td></tr><tr><td>36.4</td></tr><tr><td>36.7</td></tr><tr><td>38.7</td></tr><tr><td>37.3</td></tr><tr><td>38.1</td></tr><tr><td>39.8</td></tr><tr><td>42.1</td></tr><tr><td>38.4</td></tr><tr><td>38.5</td></tr><tr><td>42.1</td></tr><tr><td>51.2</td></tr><tr><td>40.2</td></tr><tr><td>43.2</td></tr><tr><td>36.6</td></tr><tr><td>37.3</td></tr><tr><td>41.5</td></tr><tr><td>31.5</td></tr><tr><td>35.7</td></tr><tr><td>33.6</td></tr><tr><td>34.6</td></tr><tr><td>32.8</td></tr><tr><td>34.0</td></tr><tr><td>34.9</td></tr><tr><td>34.3</td></tr><tr><td>36.5</td></tr><tr><td>35.1</td></tr><tr><td>37.8</td></tr><tr><td>39.9</td></tr><tr><td>39.1</td></tr><tr><td>40.5</td></tr><tr><td>40.5</td></tr><tr><td>38.4</td></tr><tr><td>41.4</td></tr><tr><td>35.6</td></tr><tr><td>38.0</td></tr><tr><td>37.4</td></tr><tr><td>40.1</td></tr><tr><td>40.9</td></tr><tr><td>35.6</td></tr><tr><td>36.9</td></tr><tr><td>37.5</td></tr><tr><td>36.3</td></tr><tr><td>35.5</td></tr><tr><td>38.7</td></tr><tr><td>36.4</td></tr><tr><td>33.2</td></tr><tr><td>36.5</td></tr><tr><td>36.0</td></tr><tr><td>38.7</td></tr><tr><td>38.7</td></tr><tr><td>37.8</td></tr><tr><td>37.4</td></tr><tr><td>38.4</td></tr><tr><td>38.1</td></tr><tr><td>39.3</td></tr><tr><td>38.7</td></tr><tr><td>38.5</td></tr><tr><td>36.5</td></tr><tr><td>37.7</td></tr><tr><td>36.5</td></tr><tr><td>38.0</td></tr><tr><td>36.7</td></tr><tr><td>37.2</td></tr><tr><td>39.2</td></tr><tr><td>37.5</td></tr><tr><td>38.0</td></tr><tr><td>37.3</td></tr><tr><td>41.1</td></tr><tr><td>37.5</td></tr><tr><td>38.7</td></tr><tr><td>35.9</td></tr><tr><td>40.0</td></tr><tr><td>40.1</td></tr><tr><td>37.0</td></tr><tr><td>36.3</td></tr><tr><td>40.7</td></tr><tr><td>39.6</td></tr><tr><td>31.1</td></tr><tr><td>38.6</td></tr><tr><td>42.0</td></tr><tr><td>38.5</td></tr><tr><td>34.2</td></tr><tr><td>37.2</td></tr><tr><td>37.1</td></tr><tr><td>40.2</td></tr><tr><td>35.3</td></tr><tr><td>38.0</td></tr><tr><td>36.3</td></tr><tr><td>36.8</td></tr><tr><td>41.0</td></tr><tr><td>38.3</td></tr><tr><td>38.0</td></tr><tr><td>40.8</td></tr><tr><td>39.5</td></tr><tr><td>36.9</td></tr><tr><td>36.9</td></tr><tr><td>37.7</td></tr><tr><td>36.6</td></tr><tr><td>38.9</td></tr><tr><td>37.5</td></tr><tr><td>39.8</td></tr><tr><td>38.3</td></tr><tr><td>35.5</td></tr><tr><td>36.3</td></tr><tr><td>37.8</td></tr><tr><td>37.8</td></tr><tr><td>36.5</td></tr><tr><td>37.8</td></tr><tr><td>37.0</td></tr><tr><td>37.7</td></tr><tr><td>34.3</td></tr><tr><td>40.8</td></tr><tr><td>37.4</td></tr><tr><td>36.5</td></tr><tr><td>37.5</td></tr><tr><td>35.5</td></tr><tr><td>38.0</td></tr><tr><td>35.7</td></tr><tr><td>39.2</td></tr><tr><td>40.9</td></tr><tr><td>35.2</td></tr><tr><td>40.6</td></tr><tr><td>35.4</td></tr><tr><td>41.8</td></tr><tr><td>34.1</td></tr><tr><td>37.9</td></tr><tr><td>38.2</td></tr><tr><td>35.6</td></tr><tr><td>38.5</td></tr><tr><td>37.0</td></tr><tr><td>35.9</td></tr><tr><td>36.2</td></tr><tr><td>35.0</td></tr><tr><td>38.5</td></tr><tr><td>40.7</td></tr><tr><td>36.0</td></tr><tr><td>39.5</td></tr><tr><td>40.5</td></tr><tr><td>38.5</td></tr><tr><td>43.9</td></tr><tr><td>40.4</td></tr><tr><td>37.6</td></tr><tr><td>37.0</td></tr><tr><td>34.0</td></tr><tr><td>38.4</td></tr><tr><td>38.7</td></tr><tr><td>41.5</td></tr><tr><td>36.0</td></tr><tr><td>35.3</td></tr><tr><td>42.1</td></tr><tr><td>38.0</td></tr><tr><td>42.8</td></tr><tr><td>40.0</td></tr><tr><td>33.8</td></tr><tr><td>35.5</td></tr><tr><td>35.3</td></tr><tr><td>37.7</td></tr><tr><td>39.4</td></tr><tr><td>41.9</td></tr><tr><td>38.5</td></tr><tr><td>40.8</td></tr><tr><td>38.0</td></tr><tr><td>36.4</td></tr><tr><td>41.8</td></tr><tr><td>40.7</td></tr><tr><td>38.5</td></tr><tr><td>35.4</td></tr><tr><td>38.5</td></tr><tr><td>35.5</td></tr><tr><td>36.5</td></tr><tr><td>37.6</td></tr><tr><td>37.4</td></tr><tr><td>37.8</td></tr><tr><td>35.2</td></tr><tr><td>37.9</td></tr><tr><td>37.9</td></tr><tr><td>40.9</td></tr><tr><td>41.9</td></tr><tr><td>39.1</td></tr><tr><td>40.2</td></tr><tr><td>36.0</td></tr><tr><td>34.5</td></tr><tr><td>35.8</td></tr><tr><td>40.2</td></tr><tr><td>38.3</td></tr><tr><td>39.0</td></tr><tr><td>37.4</td></tr><tr><td>41.2</td></tr><tr><td>34.8</td></tr><tr><td>36.9</td></tr><tr><td>39.4</td></tr><tr><td>37.6</td></tr><tr><td>38.5</td></tr><tr><td>42.5</td></tr><tr><td>37.4</td></tr><tr><td>35.2</td></tr><tr><td>41.1</td></tr><tr><td>33.4</td></tr><tr><td>37.2</td></tr><tr><td>38.3</td></tr><tr><td>38.1</td></tr><tr><td>37.4</td></tr><tr><td>35.2</td></tr><tr><td>39.4</td></tr><tr><td>38.0</td></tr><tr><td>35.1</td></tr><tr><td>40.4</td></tr><tr><td>38.3</td></tr><tr><td>40.6</td></tr><tr><td>40.2</td></tr><tr><td>37.9</td></tr><tr><td>40.8</td></tr><tr><td>34.7</td></tr><tr><td>38.8</td></tr><tr><td>41.4</td></tr><tr><td>41.3</td></tr><tr><td>40.7</td></tr><tr><td>36.3</td></tr><tr><td>40.8</td></tr><tr><td>34.9</td></tr><tr><td>40.9</td></tr><tr><td>38.9</td></tr><tr><td>38.9</td></tr><tr><td>40.8</td></tr></tbody></table></div>
```

#### Histogram of Chest Circumference

``` python
%sql

SELECT Chest FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Chest</th></tr></thead><tbody><tr><td>93.1</td></tr><tr><td>93.6</td></tr><tr><td>95.8</td></tr><tr><td>101.8</td></tr><tr><td>97.3</td></tr><tr><td>104.5</td></tr><tr><td>105.1</td></tr><tr><td>99.6</td></tr><tr><td>100.9</td></tr><tr><td>99.6</td></tr><tr><td>101.5</td></tr><tr><td>103.6</td></tr><tr><td>102.0</td></tr><tr><td>104.1</td></tr><tr><td>101.3</td></tr><tr><td>99.1</td></tr><tr><td>101.9</td></tr><tr><td>107.6</td></tr><tr><td>106.8</td></tr><tr><td>106.2</td></tr><tr><td>103.3</td></tr><tr><td>111.4</td></tr><tr><td>86.0</td></tr><tr><td>86.7</td></tr><tr><td>90.2</td></tr><tr><td>89.6</td></tr><tr><td>88.6</td></tr><tr><td>97.4</td></tr><tr><td>93.5</td></tr><tr><td>97.4</td></tr><tr><td>100.5</td></tr><tr><td>93.5</td></tr><tr><td>93.0</td></tr><tr><td>111.7</td></tr><tr><td>117.0</td></tr><tr><td>118.5</td></tr><tr><td>106.5</td></tr><tr><td>105.6</td></tr><tr><td>136.2</td></tr><tr><td>114.8</td></tr><tr><td>128.3</td></tr><tr><td>106.0</td></tr><tr><td>113.3</td></tr><tr><td>106.6</td></tr><tr><td>85.1</td></tr><tr><td>96.6</td></tr><tr><td>88.2</td></tr><tr><td>89.8</td></tr><tr><td>92.3</td></tr><tr><td>83.4</td></tr><tr><td>90.2</td></tr><tr><td>89.2</td></tr><tr><td>89.7</td></tr><tr><td>93.3</td></tr><tr><td>87.6</td></tr><tr><td>107.6</td></tr><tr><td>100.0</td></tr><tr><td>111.5</td></tr><tr><td>115.4</td></tr><tr><td>104.8</td></tr><tr><td>112.3</td></tr><tr><td>102.9</td></tr><tr><td>107.6</td></tr><tr><td>105.3</td></tr><tr><td>105.3</td></tr><tr><td>103.0</td></tr><tr><td>90.0</td></tr><tr><td>95.4</td></tr><tr><td>89.3</td></tr><tr><td>94.4</td></tr><tr><td>97.6</td></tr><tr><td>88.5</td></tr><tr><td>93.6</td></tr><tr><td>87.7</td></tr><tr><td>93.4</td></tr><tr><td>91.6</td></tr><tr><td>91.6</td></tr><tr><td>102.0</td></tr><tr><td>96.4</td></tr><tr><td>102.7</td></tr><tr><td>97.7</td></tr><tr><td>97.1</td></tr><tr><td>103.1</td></tr><tr><td>101.8</td></tr><tr><td>101.4</td></tr><tr><td>98.9</td></tr><tr><td>97.5</td></tr><tr><td>104.3</td></tr><tr><td>97.3</td></tr><tr><td>96.7</td></tr><tr><td>99.7</td></tr><tr><td>101.9</td></tr><tr><td>97.2</td></tr><tr><td>106.6</td></tr><tr><td>99.6</td></tr><tr><td>113.2</td></tr><tr><td>99.1</td></tr><tr><td>99.4</td></tr><tr><td>95.1</td></tr><tr><td>107.5</td></tr><tr><td>106.5</td></tr><tr><td>99.1</td></tr><tr><td>96.7</td></tr><tr><td>103.5</td></tr><tr><td>104.0</td></tr><tr><td>93.1</td></tr><tr><td>105.2</td></tr><tr><td>110.0</td></tr><tr><td>110.1</td></tr><tr><td>97.8</td></tr><tr><td>96.3</td></tr><tr><td>108.0</td></tr><tr><td>99.7</td></tr><tr><td>93.5</td></tr><tr><td>100.7</td></tr><tr><td>97.0</td></tr><tr><td>96.0</td></tr><tr><td>99.2</td></tr><tr><td>95.4</td></tr><tr><td>101.8</td></tr><tr><td>104.3</td></tr><tr><td>99.2</td></tr><tr><td>99.3</td></tr><tr><td>94.0</td></tr><tr><td>98.9</td></tr><tr><td>101.0</td></tr><tr><td>98.7</td></tr><tr><td>95.9</td></tr><tr><td>103.9</td></tr><tr><td>96.2</td></tr><tr><td>97.8</td></tr><tr><td>94.6</td></tr><tr><td>103.6</td></tr><tr><td>100.4</td></tr><tr><td>98.4</td></tr><tr><td>104.6</td></tr><tr><td>92.9</td></tr><tr><td>97.8</td></tr><tr><td>98.3</td></tr><tr><td>104.7</td></tr><tr><td>98.6</td></tr><tr><td>99.5</td></tr><tr><td>102.7</td></tr><tr><td>92.1</td></tr><tr><td>96.6</td></tr><tr><td>92.7</td></tr><tr><td>102.0</td></tr><tr><td>110.9</td></tr><tr><td>92.3</td></tr><tr><td>114.1</td></tr><tr><td>92.9</td></tr><tr><td>108.3</td></tr><tr><td>88.5</td></tr><tr><td>94.0</td></tr><tr><td>101.1</td></tr><tr><td>92.1</td></tr><tr><td>105.6</td></tr><tr><td>98.5</td></tr><tr><td>88.7</td></tr><tr><td>101.1</td></tr><tr><td>94.0</td></tr><tr><td>103.8</td></tr><tr><td>98.9</td></tr><tr><td>89.2</td></tr><tr><td>111.4</td></tr><tr><td>107.5</td></tr><tr><td>99.1</td></tr><tr><td>108.2</td></tr><tr><td>114.9</td></tr><tr><td>99.1</td></tr><tr><td>92.2</td></tr><tr><td>90.8</td></tr><tr><td>100.5</td></tr><tr><td>98.2</td></tr><tr><td>115.3</td></tr><tr><td>96.8</td></tr><tr><td>92.6</td></tr><tr><td>119.2</td></tr><tr><td>102.7</td></tr><tr><td>109.5</td></tr><tr><td>108.5</td></tr><tr><td>79.3</td></tr><tr><td>95.5</td></tr><tr><td>92.3</td></tr><tr><td>98.9</td></tr><tr><td>89.5</td></tr><tr><td>117.5</td></tr><tr><td>107.4</td></tr><tr><td>109.2</td></tr><tr><td>103.4</td></tr><tr><td>91.4</td></tr><tr><td>115.2</td></tr><tr><td>104.9</td></tr><tr><td>106.7</td></tr><tr><td>92.2</td></tr><tr><td>101.6</td></tr><tr><td>97.8</td></tr><tr><td>92.0</td></tr><tr><td>94.0</td></tr><tr><td>103.7</td></tr><tr><td>102.7</td></tr><tr><td>91.1</td></tr><tr><td>107.2</td></tr><tr><td>100.8</td></tr><tr><td>121.6</td></tr><tr><td>105.6</td></tr><tr><td>100.6</td></tr><tr><td>102.7</td></tr><tr><td>99.8</td></tr><tr><td>92.9</td></tr><tr><td>91.2</td></tr><tr><td>115.6</td></tr><tr><td>98.3</td></tr><tr><td>103.7</td></tr><tr><td>98.7</td></tr><tr><td>119.8</td></tr><tr><td>92.8</td></tr><tr><td>93.3</td></tr><tr><td>106.8</td></tr><tr><td>93.9</td></tr><tr><td>99.0</td></tr><tr><td>119.9</td></tr><tr><td>94.2</td></tr><tr><td>92.7</td></tr><tr><td>106.9</td></tr><tr><td>88.8</td></tr><tr><td>101.7</td></tr><tr><td>105.3</td></tr><tr><td>104.0</td></tr><tr><td>98.6</td></tr><tr><td>99.6</td></tr><tr><td>103.4</td></tr><tr><td>100.2</td></tr><tr><td>94.9</td></tr><tr><td>97.2</td></tr><tr><td>104.7</td></tr><tr><td>104.0</td></tr><tr><td>117.6</td></tr><tr><td>95.8</td></tr><tr><td>106.4</td></tr><tr><td>93.0</td></tr><tr><td>119.6</td></tr><tr><td>119.7</td></tr><tr><td>115.8</td></tr><tr><td>118.3</td></tr><tr><td>97.4</td></tr><tr><td>113.7</td></tr><tr><td>89.2</td></tr><tr><td>108.5</td></tr><tr><td>111.1</td></tr><tr><td>108.3</td></tr><tr><td>112.4</td></tr></tbody></table></div>
```

#### Histogram of Abdomen Circumference

``` python
%sql

SELECT Abdomen FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Abdomen</th></tr></thead><tbody><tr><td>85.2</td></tr><tr><td>83.0</td></tr><tr><td>87.9</td></tr><tr><td>86.4</td></tr><tr><td>100.0</td></tr><tr><td>94.4</td></tr><tr><td>90.7</td></tr><tr><td>88.5</td></tr><tr><td>82.5</td></tr><tr><td>88.6</td></tr><tr><td>83.6</td></tr><tr><td>90.9</td></tr><tr><td>91.6</td></tr><tr><td>101.8</td></tr><tr><td>96.4</td></tr><tr><td>92.8</td></tr><tr><td>96.4</td></tr><tr><td>97.5</td></tr><tr><td>89.6</td></tr><tr><td>100.5</td></tr><tr><td>95.9</td></tr><tr><td>98.8</td></tr><tr><td>76.4</td></tr><tr><td>80.0</td></tr><tr><td>76.3</td></tr><tr><td>79.7</td></tr><tr><td>74.6</td></tr><tr><td>88.7</td></tr><tr><td>73.9</td></tr><tr><td>83.5</td></tr><tr><td>88.7</td></tr><tr><td>84.5</td></tr><tr><td>79.1</td></tr><tr><td>100.5</td></tr><tr><td>115.6</td></tr><tr><td>113.1</td></tr><tr><td>100.9</td></tr><tr><td>98.8</td></tr><tr><td>148.1</td></tr><tr><td>108.1</td></tr><tr><td>126.2</td></tr><tr><td>104.3</td></tr><tr><td>111.2</td></tr><tr><td>104.3</td></tr><tr><td>76.0</td></tr><tr><td>81.5</td></tr><tr><td>73.7</td></tr><tr><td>79.5</td></tr><tr><td>83.4</td></tr><tr><td>70.4</td></tr><tr><td>86.7</td></tr><tr><td>77.9</td></tr><tr><td>82.0</td></tr><tr><td>79.6</td></tr><tr><td>77.6</td></tr><tr><td>100.0</td></tr><tr><td>99.8</td></tr><tr><td>104.2</td></tr><tr><td>105.3</td></tr><tr><td>98.3</td></tr><tr><td>104.8</td></tr><tr><td>94.7</td></tr><tr><td>102.4</td></tr><tr><td>99.7</td></tr><tr><td>105.5</td></tr><tr><td>100.3</td></tr><tr><td>83.9</td></tr><tr><td>86.6</td></tr><tr><td>78.4</td></tr><tr><td>84.6</td></tr><tr><td>91.5</td></tr><tr><td>82.8</td></tr><tr><td>82.9</td></tr><tr><td>76.0</td></tr><tr><td>83.3</td></tr><tr><td>81.8</td></tr><tr><td>78.8</td></tr><tr><td>95.0</td></tr><tr><td>95.4</td></tr><tr><td>98.6</td></tr><tr><td>95.8</td></tr><tr><td>89.0</td></tr><tr><td>97.8</td></tr><tr><td>94.9</td></tr><tr><td>99.8</td></tr><tr><td>89.7</td></tr><tr><td>88.1</td></tr><tr><td>90.9</td></tr><tr><td>86.0</td></tr><tr><td>86.5</td></tr><tr><td>95.6</td></tr><tr><td>93.2</td></tr><tr><td>83.1</td></tr><tr><td>97.5</td></tr><tr><td>88.8</td></tr><tr><td>99.2</td></tr><tr><td>91.6</td></tr><tr><td>86.7</td></tr><tr><td>88.2</td></tr><tr><td>94.0</td></tr><tr><td>95.0</td></tr><tr><td>92.0</td></tr><tr><td>89.2</td></tr><tr><td>95.5</td></tr><tr><td>98.6</td></tr><tr><td>87.3</td></tr><tr><td>102.8</td></tr><tr><td>101.6</td></tr><tr><td>88.7</td></tr><tr><td>92.3</td></tr><tr><td>90.6</td></tr><tr><td>105.0</td></tr><tr><td>95.0</td></tr><tr><td>89.6</td></tr><tr><td>92.4</td></tr><tr><td>86.6</td></tr><tr><td>90.0</td></tr><tr><td>90.0</td></tr><tr><td>92.4</td></tr><tr><td>87.5</td></tr><tr><td>99.2</td></tr><tr><td>98.1</td></tr><tr><td>83.3</td></tr><tr><td>86.1</td></tr><tr><td>84.1</td></tr><tr><td>89.9</td></tr><tr><td>92.1</td></tr><tr><td>78.0</td></tr><tr><td>93.5</td></tr><tr><td>87.0</td></tr><tr><td>90.1</td></tr><tr><td>90.3</td></tr><tr><td>99.8</td></tr><tr><td>89.4</td></tr><tr><td>87.2</td></tr><tr><td>101.1</td></tr><tr><td>86.1</td></tr><tr><td>98.6</td></tr><tr><td>88.5</td></tr><tr><td>106.6</td></tr><tr><td>93.1</td></tr><tr><td>93.0</td></tr><tr><td>91.0</td></tr><tr><td>77.1</td></tr><tr><td>85.3</td></tr><tr><td>81.9</td></tr><tr><td>99.1</td></tr><tr><td>100.5</td></tr><tr><td>76.5</td></tr><tr><td>106.8</td></tr><tr><td>77.6</td></tr><tr><td>102.9</td></tr><tr><td>72.8</td></tr><tr><td>88.2</td></tr><tr><td>100.1</td></tr><tr><td>83.5</td></tr><tr><td>105.0</td></tr><tr><td>90.8</td></tr><tr><td>76.6</td></tr><tr><td>92.4</td></tr><tr><td>81.2</td></tr><tr><td>95.6</td></tr><tr><td>92.1</td></tr><tr><td>83.4</td></tr><tr><td>106.0</td></tr><tr><td>95.1</td></tr><tr><td>90.4</td></tr><tr><td>100.4</td></tr><tr><td>115.9</td></tr><tr><td>90.8</td></tr><tr><td>81.9</td></tr><tr><td>75.0</td></tr><tr><td>90.3</td></tr><tr><td>90.3</td></tr><tr><td>108.8</td></tr><tr><td>79.4</td></tr><tr><td>83.2</td></tr><tr><td>110.3</td></tr><tr><td>92.7</td></tr><tr><td>104.5</td></tr><tr><td>104.6</td></tr><tr><td>69.4</td></tr><tr><td>83.6</td></tr><tr><td>86.8</td></tr><tr><td>90.4</td></tr><tr><td>83.7</td></tr><tr><td>109.3</td></tr><tr><td>98.9</td></tr><tr><td>98.0</td></tr><tr><td>101.2</td></tr><tr><td>80.6</td></tr><tr><td>113.7</td></tr><tr><td>94.1</td></tr><tr><td>105.7</td></tr><tr><td>85.6</td></tr><tr><td>96.6</td></tr><tr><td>86.0</td></tr><tr><td>89.7</td></tr><tr><td>78.0</td></tr><tr><td>89.7</td></tr><tr><td>89.2</td></tr><tr><td>85.7</td></tr><tr><td>103.1</td></tr><tr><td>89.1</td></tr><tr><td>113.9</td></tr><tr><td>96.3</td></tr><tr><td>93.9</td></tr><tr><td>101.3</td></tr><tr><td>83.9</td></tr><tr><td>84.4</td></tr><tr><td>79.4</td></tr><tr><td>104.0</td></tr><tr><td>89.7</td></tr><tr><td>97.6</td></tr><tr><td>87.6</td></tr><tr><td>122.1</td></tr><tr><td>81.1</td></tr><tr><td>81.5</td></tr><tr><td>100.0</td></tr><tr><td>88.7</td></tr><tr><td>91.8</td></tr><tr><td>110.4</td></tr><tr><td>87.6</td></tr><tr><td>82.8</td></tr><tr><td>95.3</td></tr><tr><td>78.2</td></tr><tr><td>91.1</td></tr><tr><td>96.7</td></tr><tr><td>89.4</td></tr><tr><td>93.0</td></tr><tr><td>86.4</td></tr><tr><td>96.7</td></tr><tr><td>88.1</td></tr><tr><td>94.9</td></tr><tr><td>93.3</td></tr><tr><td>95.6</td></tr><tr><td>98.2</td></tr><tr><td>113.8</td></tr><tr><td>82.8</td></tr><tr><td>100.5</td></tr><tr><td>79.7</td></tr><tr><td>118.0</td></tr><tr><td>109.0</td></tr><tr><td>113.4</td></tr><tr><td>106.1</td></tr><tr><td>84.3</td></tr><tr><td>107.6</td></tr><tr><td>83.6</td></tr><tr><td>105.0</td></tr><tr><td>111.5</td></tr><tr><td>101.3</td></tr><tr><td>108.5</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"3a50c0b9-fde0-4e56-a741-952ab0fcaa6e\",\"showTitle\":false,\"title\":\"\"}"}
#### Histogram of Hip Circumference
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"e393a81d-f156-4780-94ce-4875aa55486c\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT Hip FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Hip</th></tr></thead><tbody><tr><td>94.5</td></tr><tr><td>98.7</td></tr><tr><td>99.2</td></tr><tr><td>101.2</td></tr><tr><td>101.9</td></tr><tr><td>107.8</td></tr><tr><td>100.3</td></tr><tr><td>97.1</td></tr><tr><td>99.9</td></tr><tr><td>104.1</td></tr><tr><td>98.2</td></tr><tr><td>107.7</td></tr><tr><td>103.9</td></tr><tr><td>108.6</td></tr><tr><td>100.1</td></tr><tr><td>99.2</td></tr><tr><td>105.2</td></tr><tr><td>107.0</td></tr><tr><td>102.4</td></tr><tr><td>109.0</td></tr><tr><td>104.9</td></tr><tr><td>104.8</td></tr><tr><td>94.6</td></tr><tr><td>93.4</td></tr><tr><td>95.8</td></tr><tr><td>96.5</td></tr><tr><td>85.3</td></tr><tr><td>94.7</td></tr><tr><td>88.5</td></tr><tr><td>98.7</td></tr><tr><td>99.8</td></tr><tr><td>100.6</td></tr><tr><td>94.5</td></tr><tr><td>108.3</td></tr><tr><td>116.1</td></tr><tr><td>113.8</td></tr><tr><td>106.2</td></tr><tr><td>104.8</td></tr><tr><td>147.7</td></tr><tr><td>102.5</td></tr><tr><td>125.6</td></tr><tr><td>115.5</td></tr><tr><td>114.1</td></tr><tr><td>106.0</td></tr><tr><td>88.2</td></tr><tr><td>97.2</td></tr><tr><td>88.5</td></tr><tr><td>92.7</td></tr><tr><td>90.4</td></tr><tr><td>87.2</td></tr><tr><td>98.3</td></tr><tr><td>91.0</td></tr><tr><td>89.1</td></tr><tr><td>91.6</td></tr><tr><td>88.6</td></tr><tr><td>99.6</td></tr><tr><td>102.5</td></tr><tr><td>105.8</td></tr><tr><td>97.0</td></tr><tr><td>99.6</td></tr><tr><td>103.1</td></tr><tr><td>100.8</td></tr><tr><td>99.4</td></tr><tr><td>99.7</td></tr><tr><td>108.3</td></tr><tr><td>104.2</td></tr><tr><td>93.9</td></tr><tr><td>91.8</td></tr><tr><td>96.1</td></tr><tr><td>94.3</td></tr><tr><td>98.5</td></tr><tr><td>95.5</td></tr><tr><td>96.3</td></tr><tr><td>88.6</td></tr><tr><td>93.0</td></tr><tr><td>94.8</td></tr><tr><td>94.3</td></tr><tr><td>98.3</td></tr><tr><td>99.3</td></tr><tr><td>100.2</td></tr><tr><td>97.1</td></tr><tr><td>96.9</td></tr><tr><td>99.6</td></tr><tr><td>95.0</td></tr><tr><td>96.2</td></tr><tr><td>96.2</td></tr><tr><td>96.9</td></tr><tr><td>93.8</td></tr><tr><td>99.3</td></tr><tr><td>98.3</td></tr><tr><td>102.2</td></tr><tr><td>100.6</td></tr><tr><td>95.4</td></tr><tr><td>100.6</td></tr><tr><td>101.4</td></tr><tr><td>107.5</td></tr><tr><td>102.4</td></tr><tr><td>96.2</td></tr><tr><td>92.8</td></tr><tr><td>103.7</td></tr><tr><td>101.7</td></tr><tr><td>98.3</td></tr><tr><td>98.3</td></tr><tr><td>101.6</td></tr><tr><td>99.5</td></tr><tr><td>96.6</td></tr><tr><td>103.6</td></tr><tr><td>100.7</td></tr><tr><td>102.1</td></tr><tr><td>100.6</td></tr><tr><td>99.3</td></tr><tr><td>103.0</td></tr><tr><td>98.6</td></tr><tr><td>99.8</td></tr><tr><td>97.5</td></tr><tr><td>92.6</td></tr><tr><td>99.7</td></tr><tr><td>96.4</td></tr><tr><td>104.3</td></tr><tr><td>101.0</td></tr><tr><td>104.1</td></tr><tr><td>101.4</td></tr><tr><td>97.5</td></tr><tr><td>95.2</td></tr><tr><td>94.0</td></tr><tr><td>100.0</td></tr><tr><td>98.5</td></tr><tr><td>93.2</td></tr><tr><td>99.5</td></tr><tr><td>97.8</td></tr><tr><td>95.8</td></tr><tr><td>99.1</td></tr><tr><td>103.2</td></tr><tr><td>92.3</td></tr><tr><td>98.4</td></tr><tr><td>102.1</td></tr><tr><td>95.6</td></tr><tr><td>100.6</td></tr><tr><td>98.3</td></tr><tr><td>107.7</td></tr><tr><td>101.6</td></tr><tr><td>99.3</td></tr><tr><td>98.9</td></tr><tr><td>93.9</td></tr><tr><td>102.5</td></tr><tr><td>95.3</td></tr><tr><td>110.1</td></tr><tr><td>106.2</td></tr><tr><td>92.1</td></tr><tr><td>113.9</td></tr><tr><td>93.5</td></tr><tr><td>114.4</td></tr><tr><td>91.1</td></tr><tr><td>95.2</td></tr><tr><td>105.0</td></tr><tr><td>98.3</td></tr><tr><td>106.4</td></tr><tr><td>102.5</td></tr><tr><td>89.8</td></tr><tr><td>99.3</td></tr><tr><td>91.5</td></tr><tr><td>105.1</td></tr><tr><td>103.5</td></tr><tr><td>89.6</td></tr><tr><td>108.8</td></tr><tr><td>104.5</td></tr><tr><td>95.6</td></tr><tr><td>106.8</td></tr><tr><td>111.9</td></tr><tr><td>98.1</td></tr><tr><td>92.8</td></tr><tr><td>89.2</td></tr><tr><td>98.7</td></tr><tr><td>99.9</td></tr><tr><td>114.4</td></tr><tr><td>89.2</td></tr><tr><td>96.4</td></tr><tr><td>113.9</td></tr><tr><td>101.9</td></tr><tr><td>109.9</td></tr><tr><td>109.8</td></tr><tr><td>85.0</td></tr><tr><td>91.6</td></tr><tr><td>96.1</td></tr><tr><td>95.5</td></tr><tr><td>98.1</td></tr><tr><td>108.8</td></tr><tr><td>104.1</td></tr><tr><td>101.8</td></tr><tr><td>103.1</td></tr><tr><td>92.3</td></tr><tr><td>112.4</td></tr><tr><td>102.7</td></tr><tr><td>111.8</td></tr><tr><td>96.5</td></tr><tr><td>100.6</td></tr><tr><td>96.2</td></tr><tr><td>101.0</td></tr><tr><td>99.0</td></tr><tr><td>94.2</td></tr><tr><td>99.2</td></tr><tr><td>96.9</td></tr><tr><td>105.5</td></tr><tr><td>102.6</td></tr><tr><td>107.1</td></tr><tr><td>102.0</td></tr><tr><td>100.1</td></tr><tr><td>101.7</td></tr><tr><td>91.8</td></tr><tr><td>94.0</td></tr><tr><td>89.0</td></tr><tr><td>109.0</td></tr><tr><td>99.1</td></tr><tr><td>104.2</td></tr><tr><td>96.1</td></tr><tr><td>112.8</td></tr><tr><td>96.3</td></tr><tr><td>94.4</td></tr><tr><td>105.0</td></tr><tr><td>94.5</td></tr><tr><td>96.2</td></tr><tr><td>105.5</td></tr><tr><td>95.6</td></tr><tr><td>91.9</td></tr><tr><td>98.2</td></tr><tr><td>87.5</td></tr><tr><td>97.1</td></tr><tr><td>106.6</td></tr><tr><td>98.4</td></tr><tr><td>97.0</td></tr><tr><td>90.1</td></tr><tr><td>100.7</td></tr><tr><td>97.8</td></tr><tr><td>100.2</td></tr><tr><td>94.0</td></tr><tr><td>93.7</td></tr><tr><td>101.1</td></tr><tr><td>111.8</td></tr><tr><td>94.5</td></tr><tr><td>100.5</td></tr><tr><td>87.6</td></tr><tr><td>114.3</td></tr><tr><td>109.1</td></tr><tr><td>109.8</td></tr><tr><td>101.6</td></tr><tr><td>94.4</td></tr><tr><td>110.0</td></tr><tr><td>88.8</td></tr><tr><td>104.5</td></tr><tr><td>101.7</td></tr><tr><td>97.8</td></tr><tr><td>107.1</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"6850f4d4-abc8-4acf-bfb8-51ebe88f08d5\",\"showTitle\":false,\"title\":\"\"}"}
#### Histogram of Thigh Circumference
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"c64a175a-0820-4fac-b683-c8f6c26f5507\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT Thigh FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Thigh</th></tr></thead><tbody><tr><td>59.0</td></tr><tr><td>58.7</td></tr><tr><td>59.6</td></tr><tr><td>60.1</td></tr><tr><td>63.2</td></tr><tr><td>66.0</td></tr><tr><td>58.4</td></tr><tr><td>60.0</td></tr><tr><td>62.9</td></tr><tr><td>63.1</td></tr><tr><td>59.7</td></tr><tr><td>66.2</td></tr><tr><td>63.4</td></tr><tr><td>66.0</td></tr><tr><td>69.0</td></tr><tr><td>63.1</td></tr><tr><td>64.8</td></tr><tr><td>66.9</td></tr><tr><td>64.2</td></tr><tr><td>65.8</td></tr><tr><td>63.5</td></tr><tr><td>63.4</td></tr><tr><td>57.4</td></tr><tr><td>54.9</td></tr><tr><td>58.4</td></tr><tr><td>55.0</td></tr><tr><td>51.7</td></tr><tr><td>57.5</td></tr><tr><td>50.1</td></tr><tr><td>58.9</td></tr><tr><td>57.5</td></tr><tr><td>58.5</td></tr><tr><td>57.3</td></tr><tr><td>67.1</td></tr><tr><td>71.2</td></tr><tr><td>61.9</td></tr><tr><td>63.5</td></tr><tr><td>66.0</td></tr><tr><td>87.3</td></tr><tr><td>61.3</td></tr><tr><td>72.5</td></tr><tr><td>70.6</td></tr><tr><td>67.7</td></tr><tr><td>65.0</td></tr><tr><td>50.0</td></tr><tr><td>58.4</td></tr><tr><td>53.3</td></tr><tr><td>52.7</td></tr><tr><td>52.0</td></tr><tr><td>50.6</td></tr><tr><td>52.6</td></tr><tr><td>51.4</td></tr><tr><td>49.3</td></tr><tr><td>52.6</td></tr><tr><td>51.9</td></tr><tr><td>57.2</td></tr><tr><td>62.1</td></tr><tr><td>61.8</td></tr><tr><td>59.1</td></tr><tr><td>60.6</td></tr><tr><td>61.6</td></tr><tr><td>60.9</td></tr><tr><td>61.0</td></tr><tr><td>60.8</td></tr><tr><td>65.0</td></tr><tr><td>64.8</td></tr><tr><td>55.0</td></tr><tr><td>54.3</td></tr><tr><td>56.0</td></tr><tr><td>51.2</td></tr><tr><td>56.6</td></tr><tr><td>58.9</td></tr><tr><td>52.9</td></tr><tr><td>50.9</td></tr><tr><td>55.5</td></tr><tr><td>54.5</td></tr><tr><td>56.7</td></tr><tr><td>55.0</td></tr><tr><td>53.5</td></tr><tr><td>56.5</td></tr><tr><td>54.8</td></tr><tr><td>54.8</td></tr><tr><td>58.9</td></tr><tr><td>56.0</td></tr><tr><td>56.3</td></tr><tr><td>54.7</td></tr><tr><td>57.2</td></tr><tr><td>57.8</td></tr><tr><td>61.0</td></tr><tr><td>60.4</td></tr><tr><td>58.3</td></tr><tr><td>58.9</td></tr><tr><td>56.9</td></tr><tr><td>58.9</td></tr><tr><td>57.4</td></tr><tr><td>61.7</td></tr><tr><td>60.6</td></tr><tr><td>62.1</td></tr><tr><td>54.7</td></tr><tr><td>62.7</td></tr><tr><td>59.0</td></tr><tr><td>59.3</td></tr><tr><td>60.0</td></tr><tr><td>59.1</td></tr><tr><td>59.5</td></tr><tr><td>54.7</td></tr><tr><td>61.2</td></tr><tr><td>55.8</td></tr><tr><td>57.5</td></tr><tr><td>57.5</td></tr><tr><td>61.9</td></tr><tr><td>63.7</td></tr><tr><td>62.3</td></tr><tr><td>61.5</td></tr><tr><td>59.3</td></tr><tr><td>55.9</td></tr><tr><td>58.8</td></tr><tr><td>56.8</td></tr><tr><td>64.6</td></tr><tr><td>58.5</td></tr><tr><td>58.5</td></tr><tr><td>57.1</td></tr><tr><td>60.5</td></tr><tr><td>58.1</td></tr><tr><td>58.5</td></tr><tr><td>60.7</td></tr><tr><td>60.7</td></tr><tr><td>53.5</td></tr><tr><td>61.7</td></tr><tr><td>57.4</td></tr><tr><td>57.0</td></tr><tr><td>60.3</td></tr><tr><td>61.2</td></tr><tr><td>56.1</td></tr><tr><td>56.0</td></tr><tr><td>58.9</td></tr><tr><td>58.8</td></tr><tr><td>63.6</td></tr><tr><td>58.1</td></tr><tr><td>66.5</td></tr><tr><td>59.1</td></tr><tr><td>60.4</td></tr><tr><td>57.1</td></tr><tr><td>56.1</td></tr><tr><td>59.1</td></tr><tr><td>56.4</td></tr><tr><td>71.2</td></tr><tr><td>68.4</td></tr><tr><td>51.9</td></tr><tr><td>67.6</td></tr><tr><td>56.9</td></tr><tr><td>72.9</td></tr><tr><td>53.6</td></tr><tr><td>56.8</td></tr><tr><td>62.1</td></tr><tr><td>57.3</td></tr><tr><td>68.6</td></tr><tr><td>60.8</td></tr><tr><td>50.1</td></tr><tr><td>59.4</td></tr><tr><td>52.5</td></tr><tr><td>61.4</td></tr><tr><td>64.0</td></tr><tr><td>52.4</td></tr><tr><td>63.8</td></tr><tr><td>64.8</td></tr><tr><td>55.5</td></tr><tr><td>63.3</td></tr><tr><td>74.4</td></tr><tr><td>60.1</td></tr><tr><td>54.7</td></tr><tr><td>50.0</td></tr><tr><td>57.8</td></tr><tr><td>59.2</td></tr><tr><td>69.2</td></tr><tr><td>50.3</td></tr><tr><td>60.0</td></tr><tr><td>69.8</td></tr><tr><td>64.7</td></tr><tr><td>69.5</td></tr><tr><td>68.1</td></tr><tr><td>47.2</td></tr><tr><td>54.1</td></tr><tr><td>58.0</td></tr><tr><td>55.4</td></tr><tr><td>57.3</td></tr><tr><td>67.7</td></tr><tr><td>63.5</td></tr><tr><td>62.8</td></tr><tr><td>61.5</td></tr><tr><td>54.3</td></tr><tr><td>68.5</td></tr><tr><td>60.6</td></tr><tr><td>65.3</td></tr><tr><td>60.2</td></tr><tr><td>61.1</td></tr><tr><td>57.7</td></tr><tr><td>62.3</td></tr><tr><td>57.5</td></tr><tr><td>58.5</td></tr><tr><td>60.2</td></tr><tr><td>55.5</td></tr><tr><td>68.8</td></tr><tr><td>60.6</td></tr><tr><td>63.5</td></tr><tr><td>63.3</td></tr><tr><td>58.9</td></tr><tr><td>60.7</td></tr><tr><td>53.0</td></tr><tr><td>56.0</td></tr><tr><td>51.1</td></tr><tr><td>63.7</td></tr><tr><td>56.3</td></tr><tr><td>60.0</td></tr><tr><td>57.1</td></tr><tr><td>62.5</td></tr><tr><td>53.8</td></tr><tr><td>54.7</td></tr><tr><td>63.9</td></tr><tr><td>53.7</td></tr><tr><td>57.7</td></tr><tr><td>64.2</td></tr><tr><td>59.7</td></tr><tr><td>54.4</td></tr><tr><td>57.4</td></tr><tr><td>50.8</td></tr><tr><td>56.6</td></tr><tr><td>64.0</td></tr><tr><td>58.4</td></tr><tr><td>55.4</td></tr><tr><td>53.0</td></tr><tr><td>59.3</td></tr><tr><td>57.1</td></tr><tr><td>56.8</td></tr><tr><td>54.3</td></tr><tr><td>54.4</td></tr><tr><td>59.3</td></tr><tr><td>63.4</td></tr><tr><td>61.2</td></tr><tr><td>59.2</td></tr><tr><td>50.7</td></tr><tr><td>61.3</td></tr><tr><td>63.7</td></tr><tr><td>65.6</td></tr><tr><td>58.2</td></tr><tr><td>54.3</td></tr><tr><td>63.3</td></tr><tr><td>49.6</td></tr><tr><td>59.6</td></tr><tr><td>60.3</td></tr><tr><td>56.0</td></tr><tr><td>59.3</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"b6ee1842-b59f-4233-aa91-2c682f58f514\",\"showTitle\":false,\"title\":\"\"}"}
#### Histogram of Knee Circumference
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"21702fe8-64f6-4f1d-a3ef-de29d3f46d6b\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT Knee FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Knee</th></tr></thead><tbody><tr><td>37.3</td></tr><tr><td>37.3</td></tr><tr><td>38.9</td></tr><tr><td>37.3</td></tr><tr><td>42.2</td></tr><tr><td>42.0</td></tr><tr><td>38.3</td></tr><tr><td>39.4</td></tr><tr><td>38.3</td></tr><tr><td>41.7</td></tr><tr><td>39.7</td></tr><tr><td>39.2</td></tr><tr><td>38.3</td></tr><tr><td>41.5</td></tr><tr><td>39.0</td></tr><tr><td>38.7</td></tr><tr><td>40.8</td></tr><tr><td>40.0</td></tr><tr><td>38.7</td></tr><tr><td>40.6</td></tr><tr><td>38.0</td></tr><tr><td>40.6</td></tr><tr><td>35.3</td></tr><tr><td>36.2</td></tr><tr><td>35.5</td></tr><tr><td>36.7</td></tr><tr><td>34.7</td></tr><tr><td>36.0</td></tr><tr><td>34.5</td></tr><tr><td>35.3</td></tr><tr><td>38.7</td></tr><tr><td>38.8</td></tr><tr><td>36.2</td></tr><tr><td>44.2</td></tr><tr><td>43.3</td></tr><tr><td>38.3</td></tr><tr><td>39.9</td></tr><tr><td>41.5</td></tr><tr><td>49.1</td></tr><tr><td>41.1</td></tr><tr><td>39.6</td></tr><tr><td>42.5</td></tr><tr><td>40.9</td></tr><tr><td>40.2</td></tr><tr><td>34.7</td></tr><tr><td>38.2</td></tr><tr><td>34.5</td></tr><tr><td>37.5</td></tr><tr><td>35.8</td></tr><tr><td>34.4</td></tr><tr><td>37.2</td></tr><tr><td>34.9</td></tr><tr><td>33.7</td></tr><tr><td>37.6</td></tr><tr><td>34.9</td></tr><tr><td>38.0</td></tr><tr><td>39.6</td></tr><tr><td>39.8</td></tr><tr><td>38.0</td></tr><tr><td>37.7</td></tr><tr><td>40.9</td></tr><tr><td>38.0</td></tr><tr><td>39.4</td></tr><tr><td>40.1</td></tr><tr><td>41.2</td></tr><tr><td>40.2</td></tr><tr><td>36.1</td></tr><tr><td>35.4</td></tr><tr><td>37.4</td></tr><tr><td>37.4</td></tr><tr><td>38.6</td></tr><tr><td>37.6</td></tr><tr><td>37.5</td></tr><tr><td>35.4</td></tr><tr><td>35.2</td></tr><tr><td>37.0</td></tr><tr><td>39.7</td></tr><tr><td>38.3</td></tr><tr><td>37.5</td></tr><tr><td>39.3</td></tr><tr><td>38.2</td></tr><tr><td>38.0</td></tr><tr><td>39.0</td></tr><tr><td>36.5</td></tr><tr><td>36.6</td></tr><tr><td>37.8</td></tr><tr><td>37.7</td></tr><tr><td>39.5</td></tr><tr><td>38.4</td></tr><tr><td>39.9</td></tr><tr><td>38.2</td></tr><tr><td>39.7</td></tr><tr><td>38.3</td></tr><tr><td>40.5</td></tr><tr><td>39.6</td></tr><tr><td>42.3</td></tr><tr><td>39.4</td></tr><tr><td>39.3</td></tr><tr><td>37.3</td></tr><tr><td>39.0</td></tr><tr><td>39.4</td></tr><tr><td>38.4</td></tr><tr><td>38.4</td></tr><tr><td>39.8</td></tr><tr><td>36.1</td></tr><tr><td>39.0</td></tr><tr><td>39.3</td></tr><tr><td>38.7</td></tr><tr><td>40.0</td></tr><tr><td>36.8</td></tr><tr><td>38.0</td></tr><tr><td>40.0</td></tr><tr><td>38.1</td></tr><tr><td>37.8</td></tr><tr><td>38.1</td></tr><tr><td>36.3</td></tr><tr><td>38.4</td></tr><tr><td>38.8</td></tr><tr><td>41.1</td></tr><tr><td>39.2</td></tr><tr><td>39.3</td></tr><tr><td>40.5</td></tr><tr><td>38.7</td></tr><tr><td>36.5</td></tr><tr><td>36.6</td></tr><tr><td>36.0</td></tr><tr><td>36.8</td></tr><tr><td>35.8</td></tr><tr><td>39.0</td></tr><tr><td>36.9</td></tr><tr><td>38.7</td></tr><tr><td>38.5</td></tr><tr><td>38.1</td></tr><tr><td>35.6</td></tr><tr><td>36.9</td></tr><tr><td>37.9</td></tr><tr><td>36.1</td></tr><tr><td>39.2</td></tr><tr><td>38.4</td></tr><tr><td>42.5</td></tr><tr><td>39.6</td></tr><tr><td>38.2</td></tr><tr><td>36.7</td></tr><tr><td>36.1</td></tr><tr><td>37.6</td></tr><tr><td>36.5</td></tr><tr><td>43.5</td></tr><tr><td>40.8</td></tr><tr><td>35.7</td></tr><tr><td>42.7</td></tr><tr><td>35.9</td></tr><tr><td>43.5</td></tr><tr><td>36.8</td></tr><tr><td>37.4</td></tr><tr><td>40.0</td></tr><tr><td>37.8</td></tr><tr><td>40.0</td></tr><tr><td>38.5</td></tr><tr><td>34.8</td></tr><tr><td>39.0</td></tr><tr><td>36.6</td></tr><tr><td>40.6</td></tr><tr><td>37.3</td></tr><tr><td>35.6</td></tr><tr><td>42.0</td></tr><tr><td>41.3</td></tr><tr><td>34.2</td></tr><tr><td>41.7</td></tr><tr><td>40.6</td></tr><tr><td>39.1</td></tr><tr><td>36.2</td></tr><tr><td>34.8</td></tr><tr><td>37.3</td></tr><tr><td>37.7</td></tr><tr><td>42.4</td></tr><tr><td>34.8</td></tr><tr><td>38.1</td></tr><tr><td>42.6</td></tr><tr><td>39.5</td></tr><tr><td>43.1</td></tr><tr><td>42.8</td></tr><tr><td>33.5</td></tr><tr><td>36.2</td></tr><tr><td>39.4</td></tr><tr><td>38.9</td></tr><tr><td>39.7</td></tr><tr><td>41.3</td></tr><tr><td>39.8</td></tr><tr><td>41.3</td></tr><tr><td>40.4</td></tr><tr><td>36.3</td></tr><tr><td>45.0</td></tr><tr><td>38.6</td></tr><tr><td>43.3</td></tr><tr><td>38.9</td></tr><tr><td>38.4</td></tr><tr><td>38.6</td></tr><tr><td>38.0</td></tr><tr><td>40.0</td></tr><tr><td>39.0</td></tr><tr><td>39.2</td></tr><tr><td>35.7</td></tr><tr><td>38.3</td></tr><tr><td>39.0</td></tr><tr><td>40.3</td></tr><tr><td>39.8</td></tr><tr><td>37.6</td></tr><tr><td>39.4</td></tr><tr><td>36.2</td></tr><tr><td>38.2</td></tr><tr><td>35.0</td></tr><tr><td>40.3</td></tr><tr><td>38.8</td></tr><tr><td>40.9</td></tr><tr><td>38.1</td></tr><tr><td>36.9</td></tr><tr><td>36.5</td></tr><tr><td>39.0</td></tr><tr><td>39.2</td></tr><tr><td>36.2</td></tr><tr><td>38.1</td></tr><tr><td>42.7</td></tr><tr><td>40.2</td></tr><tr><td>35.2</td></tr><tr><td>37.1</td></tr><tr><td>33.0</td></tr><tr><td>38.5</td></tr><tr><td>42.6</td></tr><tr><td>37.4</td></tr><tr><td>38.8</td></tr><tr><td>35.0</td></tr><tr><td>38.6</td></tr><tr><td>38.9</td></tr><tr><td>35.9</td></tr><tr><td>35.7</td></tr><tr><td>37.1</td></tr><tr><td>40.3</td></tr><tr><td>41.1</td></tr><tr><td>39.1</td></tr><tr><td>38.1</td></tr><tr><td>33.4</td></tr><tr><td>42.1</td></tr><tr><td>42.4</td></tr><tr><td>46.0</td></tr><tr><td>38.8</td></tr><tr><td>37.5</td></tr><tr><td>44.0</td></tr><tr><td>34.8</td></tr><tr><td>40.8</td></tr><tr><td>37.3</td></tr><tr><td>41.6</td></tr><tr><td>42.2</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"361dd3d2-b886-4967-bef6-32ab879c5945\",\"showTitle\":false,\"title\":\"\"}"}
#### Histogram of Ankle Circumference
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"fab145ad-9f22-487c-bc32-c19dd46ef2fa\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT Ankle FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Ankle</th></tr></thead><tbody><tr><td>21.9</td></tr><tr><td>23.4</td></tr><tr><td>24.0</td></tr><tr><td>22.8</td></tr><tr><td>24.0</td></tr><tr><td>25.6</td></tr><tr><td>22.9</td></tr><tr><td>23.2</td></tr><tr><td>23.8</td></tr><tr><td>25.0</td></tr><tr><td>25.2</td></tr><tr><td>25.9</td></tr><tr><td>21.5</td></tr><tr><td>23.7</td></tr><tr><td>23.1</td></tr><tr><td>21.7</td></tr><tr><td>23.1</td></tr><tr><td>24.4</td></tr><tr><td>22.9</td></tr><tr><td>24.0</td></tr><tr><td>22.1</td></tr><tr><td>24.6</td></tr><tr><td>22.2</td></tr><tr><td>22.1</td></tr><tr><td>22.9</td></tr><tr><td>22.5</td></tr><tr><td>21.4</td></tr><tr><td>21.0</td></tr><tr><td>21.3</td></tr><tr><td>22.6</td></tr><tr><td>33.9</td></tr><tr><td>21.5</td></tr><tr><td>24.5</td></tr><tr><td>25.2</td></tr><tr><td>26.3</td></tr><tr><td>21.9</td></tr><tr><td>22.6</td></tr><tr><td>24.7</td></tr><tr><td>29.6</td></tr><tr><td>24.7</td></tr><tr><td>26.6</td></tr><tr><td>23.7</td></tr><tr><td>25.0</td></tr><tr><td>23.0</td></tr><tr><td>21.0</td></tr><tr><td>23.4</td></tr><tr><td>22.5</td></tr><tr><td>21.9</td></tr><tr><td>20.6</td></tr><tr><td>21.9</td></tr><tr><td>22.4</td></tr><tr><td>21.0</td></tr><tr><td>21.4</td></tr><tr><td>22.6</td></tr><tr><td>22.5</td></tr><tr><td>22.0</td></tr><tr><td>22.5</td></tr><tr><td>22.7</td></tr><tr><td>22.5</td></tr><tr><td>22.9</td></tr><tr><td>23.1</td></tr><tr><td>22.1</td></tr><tr><td>23.6</td></tr><tr><td>22.7</td></tr><tr><td>24.7</td></tr><tr><td>22.7</td></tr><tr><td>21.7</td></tr><tr><td>21.5</td></tr><tr><td>22.4</td></tr><tr><td>21.6</td></tr><tr><td>22.4</td></tr><tr><td>21.6</td></tr><tr><td>23.1</td></tr><tr><td>19.1</td></tr><tr><td>20.9</td></tr><tr><td>21.4</td></tr><tr><td>24.2</td></tr><tr><td>21.8</td></tr><tr><td>21.5</td></tr><tr><td>22.7</td></tr><tr><td>23.7</td></tr><tr><td>22.0</td></tr><tr><td>23.0</td></tr><tr><td>24.1</td></tr><tr><td>22.0</td></tr><tr><td>33.7</td></tr><tr><td>21.8</td></tr><tr><td>23.3</td></tr><tr><td>23.8</td></tr><tr><td>24.4</td></tr><tr><td>22.5</td></tr><tr><td>23.1</td></tr><tr><td>22.1</td></tr><tr><td>24.5</td></tr><tr><td>24.6</td></tr><tr><td>23.2</td></tr><tr><td>22.9</td></tr><tr><td>23.3</td></tr><tr><td>21.9</td></tr><tr><td>22.3</td></tr><tr><td>22.3</td></tr><tr><td>22.4</td></tr><tr><td>23.2</td></tr><tr><td>25.4</td></tr><tr><td>22.0</td></tr><tr><td>24.8</td></tr><tr><td>23.5</td></tr><tr><td>23.4</td></tr><tr><td>24.8</td></tr><tr><td>22.8</td></tr><tr><td>22.3</td></tr><tr><td>23.6</td></tr><tr><td>23.9</td></tr><tr><td>21.9</td></tr><tr><td>21.8</td></tr><tr><td>22.1</td></tr><tr><td>22.8</td></tr><tr><td>23.3</td></tr><tr><td>24.8</td></tr><tr><td>24.5</td></tr><tr><td>24.6</td></tr><tr><td>23.2</td></tr><tr><td>22.6</td></tr><tr><td>22.1</td></tr><tr><td>23.5</td></tr><tr><td>21.9</td></tr><tr><td>22.2</td></tr><tr><td>20.8</td></tr><tr><td>21.8</td></tr><tr><td>22.2</td></tr><tr><td>23.2</td></tr><tr><td>23.0</td></tr><tr><td>22.6</td></tr><tr><td>20.5</td></tr><tr><td>23.0</td></tr><tr><td>22.7</td></tr><tr><td>22.4</td></tr><tr><td>23.8</td></tr><tr><td>22.5</td></tr><tr><td>24.5</td></tr><tr><td>21.6</td></tr><tr><td>22.0</td></tr><tr><td>22.3</td></tr><tr><td>22.7</td></tr><tr><td>23.2</td></tr><tr><td>22.0</td></tr><tr><td>25.2</td></tr><tr><td>24.6</td></tr><tr><td>22.0</td></tr><tr><td>24.7</td></tr><tr><td>20.4</td></tr><tr><td>25.1</td></tr><tr><td>23.8</td></tr><tr><td>22.8</td></tr><tr><td>24.9</td></tr><tr><td>21.7</td></tr><tr><td>25.2</td></tr><tr><td>25.0</td></tr><tr><td>21.8</td></tr><tr><td>24.6</td></tr><tr><td>21.0</td></tr><tr><td>25.0</td></tr><tr><td>23.5</td></tr><tr><td>20.4</td></tr><tr><td>23.4</td></tr><tr><td>25.6</td></tr><tr><td>21.9</td></tr><tr><td>24.6</td></tr><tr><td>24.0</td></tr><tr><td>23.4</td></tr><tr><td>22.1</td></tr><tr><td>22.0</td></tr><tr><td>22.4</td></tr><tr><td>21.5</td></tr><tr><td>24.0</td></tr><tr><td>22.2</td></tr><tr><td>22.0</td></tr><tr><td>24.8</td></tr><tr><td>24.7</td></tr><tr><td>25.8</td></tr><tr><td>24.1</td></tr><tr><td>20.2</td></tr><tr><td>21.8</td></tr><tr><td>22.7</td></tr><tr><td>22.4</td></tr><tr><td>22.6</td></tr><tr><td>24.7</td></tr><tr><td>23.5</td></tr><tr><td>24.8</td></tr><tr><td>22.9</td></tr><tr><td>21.8</td></tr><tr><td>25.5</td></tr><tr><td>24.7</td></tr><tr><td>26.0</td></tr><tr><td>22.4</td></tr><tr><td>24.1</td></tr><tr><td>24.0</td></tr><tr><td>22.3</td></tr><tr><td>22.5</td></tr><tr><td>24.1</td></tr><tr><td>23.8</td></tr><tr><td>22.0</td></tr><tr><td>23.7</td></tr><tr><td>24.0</td></tr><tr><td>21.8</td></tr><tr><td>24.1</td></tr><tr><td>21.4</td></tr><tr><td>23.3</td></tr><tr><td>22.5</td></tr><tr><td>22.6</td></tr><tr><td>21.7</td></tr><tr><td>23.2</td></tr><tr><td>23.0</td></tr><tr><td>25.5</td></tr><tr><td>21.8</td></tr><tr><td>23.6</td></tr><tr><td>21.5</td></tr><tr><td>22.6</td></tr><tr><td>22.9</td></tr><tr><td>22.0</td></tr><tr><td>23.9</td></tr><tr><td>27.0</td></tr><tr><td>23.4</td></tr><tr><td>22.5</td></tr><tr><td>21.8</td></tr><tr><td>19.7</td></tr><tr><td>22.6</td></tr><tr><td>23.4</td></tr><tr><td>22.5</td></tr><tr><td>23.2</td></tr><tr><td>21.3</td></tr><tr><td>22.8</td></tr><tr><td>23.6</td></tr><tr><td>21.0</td></tr><tr><td>21.0</td></tr><tr><td>22.7</td></tr><tr><td>23.0</td></tr><tr><td>22.3</td></tr><tr><td>22.3</td></tr><tr><td>24.0</td></tr><tr><td>20.1</td></tr><tr><td>23.4</td></tr><tr><td>24.6</td></tr><tr><td>25.4</td></tr><tr><td>24.1</td></tr><tr><td>22.6</td></tr><tr><td>22.6</td></tr><tr><td>21.5</td></tr><tr><td>23.2</td></tr><tr><td>21.5</td></tr><tr><td>22.7</td></tr><tr><td>24.6</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"b5263c10-a956-4805-8405-3401d8c5db2b\",\"showTitle\":false,\"title\":\"\"}"}
#### Histogram of Biceps Circumference
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"1e7b2fd6-defc-46a5-89ea-abe8886d1170\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT Biceps FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Biceps</th></tr></thead><tbody><tr><td>32.0</td></tr><tr><td>30.5</td></tr><tr><td>28.8</td></tr><tr><td>32.4</td></tr><tr><td>32.2</td></tr><tr><td>35.7</td></tr><tr><td>31.9</td></tr><tr><td>30.5</td></tr><tr><td>35.9</td></tr><tr><td>35.6</td></tr><tr><td>32.8</td></tr><tr><td>37.2</td></tr><tr><td>32.5</td></tr><tr><td>36.9</td></tr><tr><td>36.1</td></tr><tr><td>31.1</td></tr><tr><td>36.2</td></tr><tr><td>38.2</td></tr><tr><td>37.2</td></tr><tr><td>37.1</td></tr><tr><td>32.5</td></tr><tr><td>33.0</td></tr><tr><td>27.9</td></tr><tr><td>29.8</td></tr><tr><td>31.1</td></tr><tr><td>29.9</td></tr><tr><td>28.7</td></tr><tr><td>29.2</td></tr><tr><td>30.5</td></tr><tr><td>30.1</td></tr><tr><td>32.5</td></tr><tr><td>30.1</td></tr><tr><td>29.0</td></tr><tr><td>37.5</td></tr><tr><td>37.3</td></tr><tr><td>32.0</td></tr><tr><td>35.1</td></tr><tr><td>33.2</td></tr><tr><td>45.0</td></tr><tr><td>34.1</td></tr><tr><td>36.4</td></tr><tr><td>33.6</td></tr><tr><td>36.7</td></tr><tr><td>35.8</td></tr><tr><td>26.1</td></tr><tr><td>29.7</td></tr><tr><td>27.9</td></tr><tr><td>28.8</td></tr><tr><td>28.8</td></tr><tr><td>26.8</td></tr><tr><td>26.0</td></tr><tr><td>26.7</td></tr><tr><td>29.6</td></tr><tr><td>38.5</td></tr><tr><td>27.7</td></tr><tr><td>35.9</td></tr><tr><td>33.1</td></tr><tr><td>37.7</td></tr><tr><td>31.6</td></tr><tr><td>34.5</td></tr><tr><td>36.2</td></tr><tr><td>32.5</td></tr><tr><td>32.7</td></tr><tr><td>33.6</td></tr><tr><td>35.3</td></tr><tr><td>34.8</td></tr><tr><td>29.6</td></tr><tr><td>32.8</td></tr><tr><td>32.6</td></tr><tr><td>27.3</td></tr><tr><td>31.5</td></tr><tr><td>30.3</td></tr><tr><td>29.7</td></tr><tr><td>29.3</td></tr><tr><td>29.4</td></tr><tr><td>29.3</td></tr><tr><td>30.2</td></tr><tr><td>30.8</td></tr><tr><td>31.4</td></tr><tr><td>30.3</td></tr><tr><td>29.4</td></tr><tr><td>29.9</td></tr><tr><td>34.3</td></tr><tr><td>31.2</td></tr><tr><td>29.7</td></tr><tr><td>32.4</td></tr><tr><td>32.6</td></tr><tr><td>29.2</td></tr><tr><td>30.2</td></tr><tr><td>28.8</td></tr><tr><td>29.1</td></tr><tr><td>31.4</td></tr><tr><td>30.1</td></tr><tr><td>33.3</td></tr><tr><td>30.3</td></tr><tr><td>32.9</td></tr><tr><td>31.6</td></tr><tr><td>30.6</td></tr><tr><td>31.6</td></tr><tr><td>35.3</td></tr><tr><td>32.2</td></tr><tr><td>27.9</td></tr><tr><td>31.0</td></tr><tr><td>31.0</td></tr><tr><td>30.1</td></tr><tr><td>31.0</td></tr><tr><td>30.5</td></tr><tr><td>35.1</td></tr><tr><td>35.1</td></tr><tr><td>32.1</td></tr><tr><td>33.3</td></tr><tr><td>33.5</td></tr><tr><td>35.3</td></tr><tr><td>30.7</td></tr><tr><td>31.8</td></tr><tr><td>29.8</td></tr><tr><td>29.9</td></tr><tr><td>33.4</td></tr><tr><td>33.6</td></tr><tr><td>32.1</td></tr><tr><td>33.9</td></tr><tr><td>33.0</td></tr><tr><td>34.4</td></tr><tr><td>30.6</td></tr><tr><td>34.4</td></tr><tr><td>35.6</td></tr><tr><td>33.8</td></tr><tr><td>33.9</td></tr><tr><td>33.3</td></tr><tr><td>31.6</td></tr><tr><td>27.5</td></tr><tr><td>31.2</td></tr><tr><td>33.5</td></tr><tr><td>33.6</td></tr><tr><td>34.0</td></tr><tr><td>30.9</td></tr><tr><td>32.7</td></tr><tr><td>34.3</td></tr><tr><td>31.7</td></tr><tr><td>35.5</td></tr><tr><td>30.8</td></tr><tr><td>32.0</td></tr><tr><td>31.6</td></tr><tr><td>30.5</td></tr><tr><td>31.8</td></tr><tr><td>33.5</td></tr><tr><td>36.1</td></tr><tr><td>33.3</td></tr><tr><td>25.8</td></tr><tr><td>36.0</td></tr><tr><td>31.6</td></tr><tr><td>38.5</td></tr><tr><td>27.8</td></tr><tr><td>30.6</td></tr><tr><td>33.7</td></tr><tr><td>32.2</td></tr><tr><td>35.2</td></tr><tr><td>31.6</td></tr><tr><td>27.0</td></tr><tr><td>30.1</td></tr><tr><td>27.0</td></tr><tr><td>31.3</td></tr><tr><td>33.5</td></tr><tr><td>28.3</td></tr><tr><td>34.0</td></tr><tr><td>36.4</td></tr><tr><td>30.2</td></tr><tr><td>37.2</td></tr><tr><td>36.1</td></tr><tr><td>32.5</td></tr><tr><td>30.4</td></tr><tr><td>24.8</td></tr><tr><td>31.0</td></tr><tr><td>32.4</td></tr><tr><td>35.4</td></tr><tr><td>31.0</td></tr><tr><td>31.5</td></tr><tr><td>34.4</td></tr><tr><td>34.8</td></tr><tr><td>39.1</td></tr><tr><td>35.6</td></tr><tr><td>27.7</td></tr><tr><td>31.4</td></tr><tr><td>30.0</td></tr><tr><td>30.5</td></tr><tr><td>32.9</td></tr><tr><td>37.2</td></tr><tr><td>36.4</td></tr><tr><td>36.6</td></tr><tr><td>33.4</td></tr><tr><td>29.6</td></tr><tr><td>37.1</td></tr><tr><td>34.0</td></tr><tr><td>33.7</td></tr><tr><td>31.7</td></tr><tr><td>32.9</td></tr><tr><td>31.2</td></tr><tr><td>30.8</td></tr><tr><td>30.6</td></tr><tr><td>33.8</td></tr><tr><td>31.7</td></tr><tr><td>29.4</td></tr><tr><td>32.1</td></tr><tr><td>32.9</td></tr><tr><td>34.8</td></tr><tr><td>37.3</td></tr><tr><td>33.1</td></tr><tr><td>36.7</td></tr><tr><td>31.4</td></tr><tr><td>29.0</td></tr><tr><td>30.9</td></tr><tr><td>36.8</td></tr><tr><td>29.5</td></tr><tr><td>32.7</td></tr><tr><td>28.6</td></tr><tr><td>34.7</td></tr><tr><td>31.3</td></tr><tr><td>27.5</td></tr><tr><td>35.7</td></tr><tr><td>28.5</td></tr><tr><td>31.4</td></tr><tr><td>38.4</td></tr><tr><td>27.9</td></tr><tr><td>29.4</td></tr><tr><td>34.1</td></tr><tr><td>25.3</td></tr><tr><td>33.4</td></tr><tr><td>33.2</td></tr><tr><td>34.6</td></tr><tr><td>32.4</td></tr><tr><td>31.7</td></tr><tr><td>31.8</td></tr><tr><td>30.9</td></tr><tr><td>27.8</td></tr><tr><td>31.3</td></tr><tr><td>30.3</td></tr><tr><td>32.6</td></tr><tr><td>35.1</td></tr><tr><td>29.8</td></tr><tr><td>35.9</td></tr><tr><td>28.5</td></tr><tr><td>34.9</td></tr><tr><td>35.6</td></tr><tr><td>35.3</td></tr><tr><td>32.1</td></tr><tr><td>29.2</td></tr><tr><td>37.5</td></tr><tr><td>25.6</td></tr><tr><td>35.2</td></tr><tr><td>31.3</td></tr><tr><td>30.5</td></tr><tr><td>33.7</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"2bbbe1d4-a8f6-45b5-8828-196d3ace7d86\",\"showTitle\":false,\"title\":\"\"}"}
#### Histogram of Forearm Circumference
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"51707c96-3d8c-4d52-8a4a-8a7131b8247a\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT Forearm FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Forearm</th></tr></thead><tbody><tr><td>27.4</td></tr><tr><td>28.9</td></tr><tr><td>25.2</td></tr><tr><td>29.4</td></tr><tr><td>27.7</td></tr><tr><td>30.6</td></tr><tr><td>27.8</td></tr><tr><td>29.0</td></tr><tr><td>31.1</td></tr><tr><td>30.0</td></tr><tr><td>29.4</td></tr><tr><td>30.2</td></tr><tr><td>28.6</td></tr><tr><td>31.6</td></tr><tr><td>30.5</td></tr><tr><td>26.4</td></tr><tr><td>30.8</td></tr><tr><td>31.6</td></tr><tr><td>30.5</td></tr><tr><td>30.1</td></tr><tr><td>30.3</td></tr><tr><td>32.8</td></tr><tr><td>25.9</td></tr><tr><td>26.7</td></tr><tr><td>28.0</td></tr><tr><td>28.2</td></tr><tr><td>27.0</td></tr><tr><td>26.6</td></tr><tr><td>27.9</td></tr><tr><td>26.7</td></tr><tr><td>27.7</td></tr><tr><td>26.4</td></tr><tr><td>30.0</td></tr><tr><td>31.5</td></tr><tr><td>31.7</td></tr><tr><td>29.8</td></tr><tr><td>30.6</td></tr><tr><td>30.5</td></tr><tr><td>29.0</td></tr><tr><td>31.0</td></tr><tr><td>32.7</td></tr><tr><td>28.7</td></tr><tr><td>29.8</td></tr><tr><td>31.5</td></tr><tr><td>23.1</td></tr><tr><td>27.4</td></tr><tr><td>26.2</td></tr><tr><td>26.8</td></tr><tr><td>25.5</td></tr><tr><td>25.8</td></tr><tr><td>25.8</td></tr><tr><td>26.1</td></tr><tr><td>26.0</td></tr><tr><td>27.4</td></tr><tr><td>27.5</td></tr><tr><td>30.2</td></tr><tr><td>28.3</td></tr><tr><td>30.9</td></tr><tr><td>28.8</td></tr><tr><td>29.6</td></tr><tr><td>31.8</td></tr><tr><td>29.8</td></tr><tr><td>29.9</td></tr><tr><td>29.0</td></tr><tr><td>31.1</td></tr><tr><td>30.1</td></tr><tr><td>27.4</td></tr><tr><td>27.4</td></tr><tr><td>28.1</td></tr><tr><td>27.1</td></tr><tr><td>27.3</td></tr><tr><td>27.3</td></tr><tr><td>27.3</td></tr><tr><td>25.7</td></tr><tr><td>27.0</td></tr><tr><td>27.0</td></tr><tr><td>29.2</td></tr><tr><td>25.7</td></tr><tr><td>26.8</td></tr><tr><td>28.7</td></tr><tr><td>27.2</td></tr><tr><td>25.2</td></tr><tr><td>29.6</td></tr><tr><td>27.3</td></tr><tr><td>26.3</td></tr><tr><td>27.7</td></tr><tr><td>28.0</td></tr><tr><td>28.4</td></tr><tr><td>29.3</td></tr><tr><td>29.6</td></tr><tr><td>27.7</td></tr><tr><td>28.4</td></tr><tr><td>28.2</td></tr><tr><td>29.6</td></tr><tr><td>27.9</td></tr><tr><td>30.8</td></tr><tr><td>30.1</td></tr><tr><td>27.8</td></tr><tr><td>27.5</td></tr><tr><td>30.9</td></tr><tr><td>31.0</td></tr><tr><td>26.2</td></tr><tr><td>29.2</td></tr><tr><td>30.3</td></tr><tr><td>27.2</td></tr><tr><td>29.4</td></tr><tr><td>28.5</td></tr><tr><td>29.6</td></tr><tr><td>30.7</td></tr><tr><td>26.0</td></tr><tr><td>28.2</td></tr><tr><td>27.8</td></tr><tr><td>31.1</td></tr><tr><td>27.6</td></tr><tr><td>27.3</td></tr><tr><td>26.3</td></tr><tr><td>28.0</td></tr><tr><td>29.8</td></tr><tr><td>29.5</td></tr><tr><td>28.6</td></tr><tr><td>31.2</td></tr><tr><td>29.6</td></tr><tr><td>28.0</td></tr><tr><td>27.5</td></tr><tr><td>29.2</td></tr><tr><td>30.2</td></tr><tr><td>30.3</td></tr><tr><td>28.2</td></tr><tr><td>29.6</td></tr><tr><td>27.8</td></tr><tr><td>26.5</td></tr><tr><td>28.4</td></tr><tr><td>28.6</td></tr><tr><td>29.3</td></tr><tr><td>29.8</td></tr><tr><td>28.8</td></tr><tr><td>28.3</td></tr><tr><td>28.4</td></tr><tr><td>27.4</td></tr><tr><td>29.8</td></tr><tr><td>27.9</td></tr><tr><td>28.5</td></tr><tr><td>27.5</td></tr><tr><td>27.2</td></tr><tr><td>29.7</td></tr><tr><td>28.3</td></tr><tr><td>30.3</td></tr><tr><td>29.7</td></tr><tr><td>25.2</td></tr><tr><td>30.4</td></tr><tr><td>29.0</td></tr><tr><td>33.8</td></tr><tr><td>26.3</td></tr><tr><td>28.3</td></tr><tr><td>29.2</td></tr><tr><td>27.7</td></tr><tr><td>30.7</td></tr><tr><td>28.0</td></tr><tr><td>34.9</td></tr><tr><td>28.2</td></tr><tr><td>26.3</td></tr><tr><td>29.2</td></tr><tr><td>30.6</td></tr><tr><td>26.2</td></tr><tr><td>31.2</td></tr><tr><td>33.7</td></tr><tr><td>28.7</td></tr><tr><td>33.1</td></tr><tr><td>31.8</td></tr><tr><td>29.8</td></tr><tr><td>27.4</td></tr><tr><td>25.9</td></tr><tr><td>28.7</td></tr><tr><td>28.4</td></tr><tr><td>21.0</td></tr><tr><td>26.9</td></tr><tr><td>26.6</td></tr><tr><td>29.5</td></tr><tr><td>30.3</td></tr><tr><td>32.5</td></tr><tr><td>29.0</td></tr><tr><td>24.6</td></tr><tr><td>28.3</td></tr><tr><td>26.4</td></tr><tr><td>28.9</td></tr><tr><td>29.3</td></tr><tr><td>31.8</td></tr><tr><td>30.4</td></tr><tr><td>32.4</td></tr><tr><td>29.2</td></tr><tr><td>27.3</td></tr><tr><td>31.2</td></tr><tr><td>30.1</td></tr><tr><td>29.9</td></tr><tr><td>27.1</td></tr><tr><td>29.8</td></tr><tr><td>27.3</td></tr><tr><td>27.8</td></tr><tr><td>30.0</td></tr><tr><td>28.8</td></tr><tr><td>28.4</td></tr><tr><td>26.6</td></tr><tr><td>28.9</td></tr><tr><td>29.2</td></tr><tr><td>30.7</td></tr><tr><td>23.1</td></tr><tr><td>29.5</td></tr><tr><td>31.6</td></tr><tr><td>27.5</td></tr><tr><td>26.2</td></tr><tr><td>28.8</td></tr><tr><td>31.0</td></tr><tr><td>27.9</td></tr><tr><td>30.0</td></tr><tr><td>26.7</td></tr><tr><td>29.1</td></tr><tr><td>26.3</td></tr><tr><td>25.9</td></tr><tr><td>30.4</td></tr><tr><td>25.7</td></tr><tr><td>29.9</td></tr><tr><td>32.0</td></tr><tr><td>27.0</td></tr><tr><td>26.8</td></tr><tr><td>31.1</td></tr><tr><td>22.0</td></tr><tr><td>29.3</td></tr><tr><td>30.0</td></tr><tr><td>30.1</td></tr><tr><td>29.7</td></tr><tr><td>27.3</td></tr><tr><td>29.1</td></tr><tr><td>29.6</td></tr><tr><td>26.1</td></tr><tr><td>28.7</td></tr><tr><td>26.3</td></tr><tr><td>28.5</td></tr><tr><td>29.6</td></tr><tr><td>28.9</td></tr><tr><td>30.5</td></tr><tr><td>24.8</td></tr><tr><td>30.1</td></tr><tr><td>30.7</td></tr><tr><td>29.8</td></tr><tr><td>29.3</td></tr><tr><td>27.3</td></tr><tr><td>32.6</td></tr><tr><td>25.7</td></tr><tr><td>28.6</td></tr><tr><td>27.2</td></tr><tr><td>29.4</td></tr><tr><td>30.0</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"dbb4bf09-fc15-4aaf-a904-6e994ee864e2\",\"showTitle\":false,\"title\":\"\"}"}
#### Histogram of Wrist Circumference
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"538734f7-ac3b-476e-ab8e-893775cc9008\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT Wrist FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Wrist</th></tr></thead><tbody><tr><td>17.1</td></tr><tr><td>18.2</td></tr><tr><td>16.6</td></tr><tr><td>18.2</td></tr><tr><td>17.7</td></tr><tr><td>18.8</td></tr><tr><td>17.7</td></tr><tr><td>18.8</td></tr><tr><td>18.2</td></tr><tr><td>19.2</td></tr><tr><td>18.5</td></tr><tr><td>19.0</td></tr><tr><td>17.7</td></tr><tr><td>18.8</td></tr><tr><td>18.2</td></tr><tr><td>16.9</td></tr><tr><td>17.3</td></tr><tr><td>19.3</td></tr><tr><td>18.5</td></tr><tr><td>18.2</td></tr><tr><td>18.4</td></tr><tr><td>19.9</td></tr><tr><td>16.7</td></tr><tr><td>17.1</td></tr><tr><td>17.6</td></tr><tr><td>17.7</td></tr><tr><td>16.5</td></tr><tr><td>17.0</td></tr><tr><td>17.2</td></tr><tr><td>17.6</td></tr><tr><td>18.4</td></tr><tr><td>17.9</td></tr><tr><td>18.8</td></tr><tr><td>18.7</td></tr><tr><td>19.7</td></tr><tr><td>17.0</td></tr><tr><td>19.0</td></tr><tr><td>19.4</td></tr><tr><td>21.4</td></tr><tr><td>18.3</td></tr><tr><td>21.4</td></tr><tr><td>17.4</td></tr><tr><td>18.4</td></tr><tr><td>18.8</td></tr><tr><td>16.1</td></tr><tr><td>18.3</td></tr><tr><td>17.3</td></tr><tr><td>17.9</td></tr><tr><td>16.3</td></tr><tr><td>16.8</td></tr><tr><td>17.3</td></tr><tr><td>17.2</td></tr><tr><td>16.9</td></tr><tr><td>18.5</td></tr><tr><td>18.5</td></tr><tr><td>18.9</td></tr><tr><td>18.5</td></tr><tr><td>19.2</td></tr><tr><td>18.2</td></tr><tr><td>18.5</td></tr><tr><td>20.2</td></tr><tr><td>18.3</td></tr><tr><td>19.1</td></tr><tr><td>18.8</td></tr><tr><td>18.4</td></tr><tr><td>18.7</td></tr><tr><td>17.4</td></tr><tr><td>18.7</td></tr><tr><td>18.1</td></tr><tr><td>17.3</td></tr><tr><td>18.6</td></tr><tr><td>18.3</td></tr><tr><td>18.2</td></tr><tr><td>16.9</td></tr><tr><td>16.8</td></tr><tr><td>18.3</td></tr><tr><td>18.1</td></tr><tr><td>18.8</td></tr><tr><td>18.3</td></tr><tr><td>19.0</td></tr><tr><td>19.0</td></tr><tr><td>17.7</td></tr><tr><td>19.0</td></tr><tr><td>19.2</td></tr><tr><td>18.0</td></tr><tr><td>18.2</td></tr><tr><td>18.8</td></tr><tr><td>18.1</td></tr><tr><td>18.8</td></tr><tr><td>18.7</td></tr><tr><td>17.7</td></tr><tr><td>18.8</td></tr><tr><td>18.4</td></tr><tr><td>19.1</td></tr><tr><td>17.8</td></tr><tr><td>20.4</td></tr><tr><td>18.5</td></tr><tr><td>18.2</td></tr><tr><td>18.2</td></tr><tr><td>18.3</td></tr><tr><td>18.6</td></tr><tr><td>17.0</td></tr><tr><td>18.4</td></tr><tr><td>19.7</td></tr><tr><td>17.7</td></tr><tr><td>18.8</td></tr><tr><td>18.1</td></tr><tr><td>19.1</td></tr><tr><td>19.2</td></tr><tr><td>17.3</td></tr><tr><td>18.1</td></tr><tr><td>17.4</td></tr><tr><td>19.8</td></tr><tr><td>17.4</td></tr><tr><td>17.5</td></tr><tr><td>17.3</td></tr><tr><td>18.1</td></tr><tr><td>19.5</td></tr><tr><td>18.5</td></tr><tr><td>18.0</td></tr><tr><td>19.5</td></tr><tr><td>18.4</td></tr><tr><td>17.6</td></tr><tr><td>17.6</td></tr><tr><td>18.0</td></tr><tr><td>17.6</td></tr><tr><td>17.2</td></tr><tr><td>17.4</td></tr><tr><td>18.1</td></tr><tr><td>17.7</td></tr><tr><td>17.6</td></tr><tr><td>17.1</td></tr><tr><td>17.9</td></tr><tr><td>17.3</td></tr><tr><td>18.1</td></tr><tr><td>17.6</td></tr><tr><td>17.1</td></tr><tr><td>17.7</td></tr><tr><td>17.6</td></tr><tr><td>18.7</td></tr><tr><td>16.6</td></tr><tr><td>17.8</td></tr><tr><td>17.9</td></tr><tr><td>18.2</td></tr><tr><td>18.3</td></tr><tr><td>17.3</td></tr><tr><td>18.7</td></tr><tr><td>18.4</td></tr><tr><td>16.9</td></tr><tr><td>18.4</td></tr><tr><td>17.8</td></tr><tr><td>19.6</td></tr><tr><td>17.4</td></tr><tr><td>17.9</td></tr><tr><td>19.4</td></tr><tr><td>17.7</td></tr><tr><td>19.1</td></tr><tr><td>18.6</td></tr><tr><td>16.9</td></tr><tr><td>18.2</td></tr><tr><td>16.5</td></tr><tr><td>19.1</td></tr><tr><td>19.7</td></tr><tr><td>16.5</td></tr><tr><td>18.5</td></tr><tr><td>19.4</td></tr><tr><td>17.7</td></tr><tr><td>19.8</td></tr><tr><td>18.8</td></tr><tr><td>17.4</td></tr><tr><td>17.7</td></tr><tr><td>16.9</td></tr><tr><td>17.7</td></tr><tr><td>17.8</td></tr><tr><td>20.1</td></tr><tr><td>16.9</td></tr><tr><td>16.7</td></tr><tr><td>18.4</td></tr><tr><td>18.1</td></tr><tr><td>19.9</td></tr><tr><td>19.0</td></tr><tr><td>16.5</td></tr><tr><td>17.2</td></tr><tr><td>17.4</td></tr><tr><td>17.7</td></tr><tr><td>18.2</td></tr><tr><td>20.0</td></tr><tr><td>19.1</td></tr><tr><td>18.8</td></tr><tr><td>18.5</td></tr><tr><td>17.9</td></tr><tr><td>19.9</td></tr><tr><td>18.7</td></tr><tr><td>18.5</td></tr><tr><td>17.1</td></tr><tr><td>18.8</td></tr><tr><td>17.4</td></tr><tr><td>16.9</td></tr><tr><td>18.5</td></tr><tr><td>18.8</td></tr><tr><td>18.6</td></tr><tr><td>17.4</td></tr><tr><td>18.7</td></tr><tr><td>18.4</td></tr><tr><td>17.4</td></tr><tr><td>19.4</td></tr><tr><td>17.3</td></tr><tr><td>18.4</td></tr><tr><td>17.7</td></tr><tr><td>17.6</td></tr><tr><td>17.4</td></tr><tr><td>18.9</td></tr><tr><td>18.6</td></tr><tr><td>19.0</td></tr><tr><td>18.0</td></tr><tr><td>18.4</td></tr><tr><td>17.8</td></tr><tr><td>18.6</td></tr><tr><td>19.2</td></tr><tr><td>17.1</td></tr><tr><td>18.9</td></tr><tr><td>19.6</td></tr><tr><td>17.8</td></tr><tr><td>17.0</td></tr><tr><td>19.2</td></tr><tr><td>15.8</td></tr><tr><td>18.8</td></tr><tr><td>18.4</td></tr><tr><td>18.8</td></tr><tr><td>19.0</td></tr><tr><td>16.9</td></tr><tr><td>19.0</td></tr><tr><td>18.0</td></tr><tr><td>17.6</td></tr><tr><td>18.3</td></tr><tr><td>18.3</td></tr><tr><td>19.0</td></tr><tr><td>18.5</td></tr><tr><td>18.3</td></tr><tr><td>19.1</td></tr><tr><td>16.5</td></tr><tr><td>19.4</td></tr><tr><td>19.5</td></tr><tr><td>19.5</td></tr><tr><td>18.5</td></tr><tr><td>18.5</td></tr><tr><td>18.8</td></tr><tr><td>18.5</td></tr><tr><td>20.1</td></tr><tr><td>18.0</td></tr><tr><td>19.8</td></tr><tr><td>20.9</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"d1caa974-9c54-460d-ba9a-7cde96f1a7bc\",\"showTitle\":false,\"title\":\"\"}"}
## Correlation Analysis
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"0f082388-52b4-495f-9ac5-fa137bdc68ca\",\"showTitle\":false,\"title\":\"\"}"}
``` python
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=df.columns, outputCol=vector_col)
df_vector = assembler.transform(df).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
corrmatrix = matrix.toArray().tolist()

corrdf = spark.createDataFrame(corrmatrix,df.columns)
corrdf.head()
```

::: {.output .stream .stdout}
    Out[28]: Row(Density=1.0, BodyFat=-0.9877824021639864, Age=-0.27763721075009434, Weight=-0.5940618756988721, Height=0.09788114294593664, Neck=-0.4729663617987716, Chest=-0.6825986513963713, Abdomen=-0.7989546315453323, Hip=-0.6093314301876932, Thigh=-0.5530909789787456, Knee=-0.49504035298897714, Ankle=-0.2648900333564061, Biceps=-0.4871087232659662, Forearm=-0.3516484181331154, Wrist=-0.3257159810204718)
:::
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"daa23ac8-6945-4a1e-b7d1-9d3e10c0b459\",\"showTitle\":false,\"title\":\"\"}"}
``` python
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
ax.set_title("Correlation Matrix for Specified Attributes")

# plot the correlation matrix
cax = ax.imshow(corrmatrix, vmax=1, vmin=-1, cmap='coolwarm')

# set labels
ax.set_xticks(np.arange(len(df.columns)))
ax.set_yticks(np.arange(len(df.columns)))
ax.set_xticklabels(df.columns, rotation=90, ha='center')
ax.set_yticklabels(df.columns)

# add colorbar
fig.colorbar(cax)

plt.tight_layout()
plt.show()

```

::: {.output .display_data}
![](vertopal_12d9cc67628e4826b52142c82e1b17ce/15374280b0251cf7c17cdf15dd48d386ce65b618.png)
:::
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"ba22552b-16ea-4c12-88c3-7a3cf13da8f7\",\"showTitle\":false,\"title\":\"\"}"}
``` python
import six

for i in df.columns:
    if not( isinstance( df.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to BodyFat for ", i, df.stat.corr('BodyFat',i))
```

::: {.output .stream .stdout}
    Correlation to BodyFat for  Density -0.9877824021639853
    Correlation to BodyFat for  BodyFat 1.0
    Correlation to BodyFat for  Age 0.29145844013522204
    Correlation to BodyFat for  Weight 0.6124140022026475
    Correlation to BodyFat for  Height -0.08949537985440173
    Correlation to BodyFat for  Neck 0.4905918534410396
    Correlation to BodyFat for  Chest 0.7026203388938641
    Correlation to BodyFat for  Abdomen 0.813432284781049
    Correlation to BodyFat for  Hip 0.6252009175086624
    Correlation to BodyFat for  Thigh 0.5596075319940894
    Correlation to BodyFat for  Knee 0.5086652428854677
    Correlation to BodyFat for  Ankle 0.265969770306373
    Correlation to BodyFat for  Biceps 0.49327112589161554
    Correlation to BodyFat for  Forearm 0.3613869031997192
    Correlation to BodyFat for  Wrist 0.34657486452658576
:::
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"8a0b9bfb-9dde-4743-93ac-68c37fe44b42\",\"showTitle\":false,\"title\":\"\"}"}
#### Conclusion

The correlation coefficient ranges from --1 to 1. When it is close to 1,
it means that there is a strong positive correlation; for example, the
body fat tends to go up when the abdomen circumference goes up.

When the coefficient is close to --1, it means that there is a strong
negative correlation; the body fat tends to go down when the density
goes up.

Finally, coefficients close to zero mean that there is no linear
correlation.
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"a2ff757a-93c7-4674-932f-7b7c4c3d42b3\",\"showTitle\":false,\"title\":\"\"}"}
## Exploring Dependencies of Life Expectancy
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"f7b7b512-50b4-4643-b1ef-dc12d5fac1db\",\"showTitle\":false,\"title\":\"\"}"}
Scatter Plot (BodyFat VS Density)
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"03837190-962a-4fe7-adb3-cdd3d064c3e7\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT BodyFat, Density FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>BodyFat</th><th>Density</th></tr></thead><tbody><tr><td>12.3</td><td>1.0708</td></tr><tr><td>6.1</td><td>1.0853</td></tr><tr><td>25.3</td><td>1.0414</td></tr><tr><td>10.4</td><td>1.0751</td></tr><tr><td>28.7</td><td>1.034</td></tr><tr><td>20.9</td><td>1.0502</td></tr><tr><td>19.2</td><td>1.0549</td></tr><tr><td>12.4</td><td>1.0704</td></tr><tr><td>4.1</td><td>1.09</td></tr><tr><td>11.7</td><td>1.0722</td></tr><tr><td>7.1</td><td>1.083</td></tr><tr><td>7.8</td><td>1.0812</td></tr><tr><td>20.8</td><td>1.0513</td></tr><tr><td>21.2</td><td>1.0505</td></tr><tr><td>22.1</td><td>1.0484</td></tr><tr><td>20.9</td><td>1.0512</td></tr><tr><td>29.0</td><td>1.0333</td></tr><tr><td>22.9</td><td>1.0468</td></tr><tr><td>16.0</td><td>1.0622</td></tr><tr><td>16.5</td><td>1.061</td></tr><tr><td>19.1</td><td>1.0551</td></tr><tr><td>15.2</td><td>1.064</td></tr><tr><td>15.6</td><td>1.0631</td></tr><tr><td>17.7</td><td>1.0584</td></tr><tr><td>14.0</td><td>1.0668</td></tr><tr><td>3.7</td><td>1.0911</td></tr><tr><td>7.9</td><td>1.0811</td></tr><tr><td>22.9</td><td>1.0468</td></tr><tr><td>3.7</td><td>1.091</td></tr><tr><td>8.8</td><td>1.079</td></tr><tr><td>11.9</td><td>1.0716</td></tr><tr><td>5.7</td><td>1.0862</td></tr><tr><td>11.8</td><td>1.0719</td></tr><tr><td>21.3</td><td>1.0502</td></tr><tr><td>32.3</td><td>1.0263</td></tr><tr><td>40.1</td><td>1.0101</td></tr><tr><td>24.2</td><td>1.0438</td></tr><tr><td>28.4</td><td>1.0346</td></tr><tr><td>35.2</td><td>1.0202</td></tr><tr><td>32.6</td><td>1.0258</td></tr><tr><td>34.5</td><td>1.0217</td></tr><tr><td>32.9</td><td>1.025</td></tr><tr><td>31.6</td><td>1.0279</td></tr><tr><td>32.0</td><td>1.0269</td></tr><tr><td>7.7</td><td>1.0814</td></tr><tr><td>13.9</td><td>1.067</td></tr><tr><td>10.8</td><td>1.0742</td></tr><tr><td>5.6</td><td>1.0665</td></tr><tr><td>13.6</td><td>1.0678</td></tr><tr><td>4.0</td><td>1.0903</td></tr><tr><td>10.2</td><td>1.0756</td></tr><tr><td>6.6</td><td>1.084</td></tr><tr><td>8.0</td><td>1.0807</td></tr><tr><td>6.3</td><td>1.0848</td></tr><tr><td>3.9</td><td>1.0906</td></tr><tr><td>22.6</td><td>1.0473</td></tr><tr><td>20.4</td><td>1.0524</td></tr><tr><td>28.0</td><td>1.0356</td></tr><tr><td>31.5</td><td>1.028</td></tr><tr><td>24.6</td><td>1.043</td></tr><tr><td>26.1</td><td>1.0396</td></tr><tr><td>29.8</td><td>1.0317</td></tr><tr><td>30.7</td><td>1.0298</td></tr><tr><td>25.8</td><td>1.0403</td></tr><tr><td>32.3</td><td>1.0264</td></tr><tr><td>30.0</td><td>1.0313</td></tr><tr><td>21.5</td><td>1.0499</td></tr><tr><td>13.8</td><td>1.0673</td></tr><tr><td>6.3</td><td>1.0847</td></tr><tr><td>12.9</td><td>1.0693</td></tr><tr><td>24.3</td><td>1.0439</td></tr><tr><td>8.8</td><td>1.0788</td></tr><tr><td>8.5</td><td>1.0796</td></tr><tr><td>13.5</td><td>1.068</td></tr><tr><td>11.8</td><td>1.072</td></tr><tr><td>18.5</td><td>1.0666</td></tr><tr><td>8.8</td><td>1.079</td></tr><tr><td>22.2</td><td>1.0483</td></tr><tr><td>21.5</td><td>1.0498</td></tr><tr><td>18.8</td><td>1.056</td></tr><tr><td>31.4</td><td>1.0283</td></tr><tr><td>26.8</td><td>1.0382</td></tr><tr><td>18.4</td><td>1.0568</td></tr><tr><td>27.0</td><td>1.0377</td></tr><tr><td>27.0</td><td>1.0378</td></tr><tr><td>26.6</td><td>1.0386</td></tr><tr><td>14.9</td><td>1.0648</td></tr><tr><td>23.1</td><td>1.0462</td></tr><tr><td>8.3</td><td>1.08</td></tr><tr><td>14.1</td><td>1.0666</td></tr><tr><td>20.5</td><td>1.052</td></tr><tr><td>18.2</td><td>1.0573</td></tr><tr><td>8.5</td><td>1.0795</td></tr><tr><td>24.9</td><td>1.0424</td></tr><tr><td>9.0</td><td>1.0785</td></tr><tr><td>17.4</td><td>1.0991</td></tr><tr><td>9.6</td><td>1.077</td></tr><tr><td>11.3</td><td>1.073</td></tr><tr><td>17.8</td><td>1.0582</td></tr><tr><td>22.2</td><td>1.0484</td></tr><tr><td>21.2</td><td>1.0506</td></tr><tr><td>20.4</td><td>1.0524</td></tr><tr><td>20.1</td><td>1.053</td></tr><tr><td>22.3</td><td>1.048</td></tr><tr><td>25.4</td><td>1.0412</td></tr><tr><td>18.0</td><td>1.0578</td></tr><tr><td>19.3</td><td>1.0547</td></tr><tr><td>18.3</td><td>1.0569</td></tr><tr><td>17.3</td><td>1.0593</td></tr><tr><td>21.4</td><td>1.05</td></tr><tr><td>19.7</td><td>1.0538</td></tr><tr><td>28.0</td><td>1.0355</td></tr><tr><td>22.1</td><td>1.0486</td></tr><tr><td>21.3</td><td>1.0503</td></tr><tr><td>26.7</td><td>1.0384</td></tr><tr><td>16.7</td><td>1.0607</td></tr><tr><td>20.1</td><td>1.0529</td></tr><tr><td>13.9</td><td>1.0671</td></tr><tr><td>25.8</td><td>1.0404</td></tr><tr><td>18.1</td><td>1.0575</td></tr><tr><td>27.9</td><td>1.0358</td></tr><tr><td>25.3</td><td>1.0414</td></tr><tr><td>14.7</td><td>1.0652</td></tr><tr><td>16.0</td><td>1.0623</td></tr><tr><td>13.8</td><td>1.0674</td></tr><tr><td>17.5</td><td>1.0587</td></tr><tr><td>27.2</td><td>1.0373</td></tr><tr><td>17.4</td><td>1.059</td></tr><tr><td>20.8</td><td>1.0515</td></tr><tr><td>14.9</td><td>1.0648</td></tr><tr><td>18.1</td><td>1.0575</td></tr><tr><td>22.7</td><td>1.0472</td></tr><tr><td>23.6</td><td>1.0452</td></tr><tr><td>26.1</td><td>1.0398</td></tr><tr><td>24.4</td><td>1.0435</td></tr><tr><td>27.1</td><td>1.0374</td></tr><tr><td>21.8</td><td>1.0491</td></tr><tr><td>29.4</td><td>1.0325</td></tr><tr><td>22.4</td><td>1.0481</td></tr><tr><td>20.4</td><td>1.0522</td></tr><tr><td>24.9</td><td>1.0422</td></tr><tr><td>18.3</td><td>1.0571</td></tr><tr><td>23.3</td><td>1.0459</td></tr><tr><td>9.4</td><td>1.0775</td></tr><tr><td>10.3</td><td>1.0754</td></tr><tr><td>14.2</td><td>1.0664</td></tr><tr><td>19.2</td><td>1.055</td></tr><tr><td>29.6</td><td>1.0322</td></tr><tr><td>5.3</td><td>1.0873</td></tr><tr><td>25.2</td><td>1.0416</td></tr><tr><td>9.4</td><td>1.0776</td></tr><tr><td>19.6</td><td>1.0542</td></tr><tr><td>10.1</td><td>1.0758</td></tr><tr><td>16.5</td><td>1.061</td></tr><tr><td>21.0</td><td>1.051</td></tr><tr><td>17.3</td><td>1.0594</td></tr><tr><td>31.2</td><td>1.0287</td></tr><tr><td>10.0</td><td>1.0761</td></tr><tr><td>12.5</td><td>1.0704</td></tr><tr><td>22.5</td><td>1.0477</td></tr><tr><td>9.4</td><td>1.0775</td></tr><tr><td>14.6</td><td>1.0653</td></tr><tr><td>13.0</td><td>1.069</td></tr><tr><td>15.1</td><td>1.0644</td></tr><tr><td>27.3</td><td>1.037</td></tr><tr><td>19.2</td><td>1.0549</td></tr><tr><td>21.8</td><td>1.0492</td></tr><tr><td>20.3</td><td>1.0525</td></tr><tr><td>34.3</td><td>1.018</td></tr><tr><td>16.5</td><td>1.061</td></tr><tr><td>3.0</td><td>1.0926</td></tr><tr><td>0.7</td><td>1.0983</td></tr><tr><td>20.5</td><td>1.0521</td></tr><tr><td>16.9</td><td>1.0603</td></tr><tr><td>25.3</td><td>1.0414</td></tr><tr><td>9.9</td><td>1.0763</td></tr><tr><td>13.1</td><td>1.0689</td></tr><tr><td>29.9</td><td>1.0316</td></tr><tr><td>22.5</td><td>1.0477</td></tr><tr><td>16.9</td><td>1.0603</td></tr><tr><td>26.6</td><td>1.0387</td></tr><tr><td>0.0</td><td>1.1089</td></tr><tr><td>11.5</td><td>1.0725</td></tr><tr><td>12.1</td><td>1.0713</td></tr><tr><td>17.5</td><td>1.0587</td></tr><tr><td>8.6</td><td>1.0794</td></tr><tr><td>23.6</td><td>1.0453</td></tr><tr><td>20.4</td><td>1.0524</td></tr><tr><td>20.5</td><td>1.052</td></tr><tr><td>24.4</td><td>1.0434</td></tr><tr><td>11.4</td><td>1.0728</td></tr><tr><td>38.1</td><td>1.014</td></tr><tr><td>15.9</td><td>1.0624</td></tr><tr><td>24.7</td><td>1.0429</td></tr><tr><td>22.8</td><td>1.047</td></tr><tr><td>25.5</td><td>1.0411</td></tr><tr><td>22.0</td><td>1.0488</td></tr><tr><td>17.7</td><td>1.0583</td></tr><tr><td>6.6</td><td>1.0841</td></tr><tr><td>23.6</td><td>1.0462</td></tr><tr><td>12.2</td><td>1.0709</td></tr><tr><td>22.1</td><td>1.0484</td></tr><tr><td>28.7</td><td>1.034</td></tr><tr><td>6.0</td><td>1.0854</td></tr><tr><td>34.8</td><td>1.0209</td></tr><tr><td>16.6</td><td>1.061</td></tr><tr><td>32.9</td><td>1.025</td></tr><tr><td>32.8</td><td>1.0254</td></tr><tr><td>9.6</td><td>1.0771</td></tr><tr><td>10.8</td><td>1.0742</td></tr><tr><td>7.1</td><td>1.0829</td></tr><tr><td>27.2</td><td>1.0373</td></tr><tr><td>19.5</td><td>1.0543</td></tr><tr><td>18.7</td><td>1.0561</td></tr><tr><td>19.5</td><td>1.0543</td></tr><tr><td>47.5</td><td>0.995</td></tr><tr><td>13.6</td><td>1.0678</td></tr><tr><td>7.5</td><td>1.0819</td></tr><tr><td>24.5</td><td>1.0433</td></tr><tr><td>15.0</td><td>1.0646</td></tr><tr><td>12.4</td><td>1.0706</td></tr><tr><td>26.0</td><td>1.0399</td></tr><tr><td>11.5</td><td>1.0726</td></tr><tr><td>5.2</td><td>1.0874</td></tr><tr><td>10.9</td><td>1.074</td></tr><tr><td>12.5</td><td>1.0703</td></tr><tr><td>14.8</td><td>1.065</td></tr><tr><td>25.2</td><td>1.0418</td></tr><tr><td>14.9</td><td>1.0647</td></tr><tr><td>17.0</td><td>1.0601</td></tr><tr><td>10.6</td><td>1.0745</td></tr><tr><td>16.1</td><td>1.062</td></tr><tr><td>15.4</td><td>1.0636</td></tr><tr><td>26.7</td><td>1.0384</td></tr><tr><td>25.8</td><td>1.0403</td></tr><tr><td>18.6</td><td>1.0563</td></tr><tr><td>24.8</td><td>1.0424</td></tr><tr><td>27.3</td><td>1.0372</td></tr><tr><td>12.4</td><td>1.0705</td></tr><tr><td>29.9</td><td>1.0316</td></tr><tr><td>17.0</td><td>1.0599</td></tr><tr><td>35.0</td><td>1.0207</td></tr><tr><td>30.4</td><td>1.0304</td></tr><tr><td>32.6</td><td>1.0256</td></tr><tr><td>29.0</td><td>1.0334</td></tr><tr><td>15.2</td><td>1.0641</td></tr><tr><td>30.2</td><td>1.0308</td></tr><tr><td>11.0</td><td>1.0736</td></tr><tr><td>33.6</td><td>1.0236</td></tr><tr><td>29.3</td><td>1.0328</td></tr><tr><td>26.0</td><td>1.0399</td></tr><tr><td>31.9</td><td>1.0271</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"71f26e10-91a9-4ed5-af59-e64d02af210a\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT BodyFat, Age FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>BodyFat</th><th>Age</th></tr></thead><tbody><tr><td>12.3</td><td>23</td></tr><tr><td>6.1</td><td>22</td></tr><tr><td>25.3</td><td>22</td></tr><tr><td>10.4</td><td>26</td></tr><tr><td>28.7</td><td>24</td></tr><tr><td>20.9</td><td>24</td></tr><tr><td>19.2</td><td>26</td></tr><tr><td>12.4</td><td>25</td></tr><tr><td>4.1</td><td>25</td></tr><tr><td>11.7</td><td>23</td></tr><tr><td>7.1</td><td>26</td></tr><tr><td>7.8</td><td>27</td></tr><tr><td>20.8</td><td>32</td></tr><tr><td>21.2</td><td>30</td></tr><tr><td>22.1</td><td>35</td></tr><tr><td>20.9</td><td>35</td></tr><tr><td>29.0</td><td>34</td></tr><tr><td>22.9</td><td>32</td></tr><tr><td>16.0</td><td>28</td></tr><tr><td>16.5</td><td>33</td></tr><tr><td>19.1</td><td>28</td></tr><tr><td>15.2</td><td>28</td></tr><tr><td>15.6</td><td>31</td></tr><tr><td>17.7</td><td>32</td></tr><tr><td>14.0</td><td>28</td></tr><tr><td>3.7</td><td>27</td></tr><tr><td>7.9</td><td>34</td></tr><tr><td>22.9</td><td>31</td></tr><tr><td>3.7</td><td>27</td></tr><tr><td>8.8</td><td>29</td></tr><tr><td>11.9</td><td>32</td></tr><tr><td>5.7</td><td>29</td></tr><tr><td>11.8</td><td>27</td></tr><tr><td>21.3</td><td>41</td></tr><tr><td>32.3</td><td>41</td></tr><tr><td>40.1</td><td>49</td></tr><tr><td>24.2</td><td>40</td></tr><tr><td>28.4</td><td>50</td></tr><tr><td>35.2</td><td>46</td></tr><tr><td>32.6</td><td>50</td></tr><tr><td>34.5</td><td>45</td></tr><tr><td>32.9</td><td>44</td></tr><tr><td>31.6</td><td>48</td></tr><tr><td>32.0</td><td>41</td></tr><tr><td>7.7</td><td>39</td></tr><tr><td>13.9</td><td>43</td></tr><tr><td>10.8</td><td>40</td></tr><tr><td>5.6</td><td>39</td></tr><tr><td>13.6</td><td>45</td></tr><tr><td>4.0</td><td>47</td></tr><tr><td>10.2</td><td>47</td></tr><tr><td>6.6</td><td>40</td></tr><tr><td>8.0</td><td>51</td></tr><tr><td>6.3</td><td>49</td></tr><tr><td>3.9</td><td>42</td></tr><tr><td>22.6</td><td>54</td></tr><tr><td>20.4</td><td>58</td></tr><tr><td>28.0</td><td>62</td></tr><tr><td>31.5</td><td>54</td></tr><tr><td>24.6</td><td>61</td></tr><tr><td>26.1</td><td>62</td></tr><tr><td>29.8</td><td>56</td></tr><tr><td>30.7</td><td>54</td></tr><tr><td>25.8</td><td>61</td></tr><tr><td>32.3</td><td>57</td></tr><tr><td>30.0</td><td>55</td></tr><tr><td>21.5</td><td>54</td></tr><tr><td>13.8</td><td>55</td></tr><tr><td>6.3</td><td>54</td></tr><tr><td>12.9</td><td>55</td></tr><tr><td>24.3</td><td>62</td></tr><tr><td>8.8</td><td>55</td></tr><tr><td>8.5</td><td>56</td></tr><tr><td>13.5</td><td>55</td></tr><tr><td>11.8</td><td>61</td></tr><tr><td>18.5</td><td>61</td></tr><tr><td>8.8</td><td>57</td></tr><tr><td>22.2</td><td>69</td></tr><tr><td>21.5</td><td>81</td></tr><tr><td>18.8</td><td>66</td></tr><tr><td>31.4</td><td>67</td></tr><tr><td>26.8</td><td>64</td></tr><tr><td>18.4</td><td>64</td></tr><tr><td>27.0</td><td>70</td></tr><tr><td>27.0</td><td>72</td></tr><tr><td>26.6</td><td>67</td></tr><tr><td>14.9</td><td>72</td></tr><tr><td>23.1</td><td>64</td></tr><tr><td>8.3</td><td>46</td></tr><tr><td>14.1</td><td>48</td></tr><tr><td>20.5</td><td>46</td></tr><tr><td>18.2</td><td>44</td></tr><tr><td>8.5</td><td>47</td></tr><tr><td>24.9</td><td>46</td></tr><tr><td>9.0</td><td>47</td></tr><tr><td>17.4</td><td>53</td></tr><tr><td>9.6</td><td>38</td></tr><tr><td>11.3</td><td>50</td></tr><tr><td>17.8</td><td>46</td></tr><tr><td>22.2</td><td>47</td></tr><tr><td>21.2</td><td>49</td></tr><tr><td>20.4</td><td>48</td></tr><tr><td>20.1</td><td>41</td></tr><tr><td>22.3</td><td>49</td></tr><tr><td>25.4</td><td>43</td></tr><tr><td>18.0</td><td>43</td></tr><tr><td>19.3</td><td>43</td></tr><tr><td>18.3</td><td>52</td></tr><tr><td>17.3</td><td>43</td></tr><tr><td>21.4</td><td>40</td></tr><tr><td>19.7</td><td>43</td></tr><tr><td>28.0</td><td>43</td></tr><tr><td>22.1</td><td>47</td></tr><tr><td>21.3</td><td>42</td></tr><tr><td>26.7</td><td>48</td></tr><tr><td>16.7</td><td>40</td></tr><tr><td>20.1</td><td>48</td></tr><tr><td>13.9</td><td>51</td></tr><tr><td>25.8</td><td>40</td></tr><tr><td>18.1</td><td>44</td></tr><tr><td>27.9</td><td>52</td></tr><tr><td>25.3</td><td>44</td></tr><tr><td>14.7</td><td>40</td></tr><tr><td>16.0</td><td>47</td></tr><tr><td>13.8</td><td>50</td></tr><tr><td>17.5</td><td>46</td></tr><tr><td>27.2</td><td>42</td></tr><tr><td>17.4</td><td>43</td></tr><tr><td>20.8</td><td>40</td></tr><tr><td>14.9</td><td>42</td></tr><tr><td>18.1</td><td>49</td></tr><tr><td>22.7</td><td>40</td></tr><tr><td>23.6</td><td>47</td></tr><tr><td>26.1</td><td>50</td></tr><tr><td>24.4</td><td>41</td></tr><tr><td>27.1</td><td>44</td></tr><tr><td>21.8</td><td>39</td></tr><tr><td>29.4</td><td>43</td></tr><tr><td>22.4</td><td>40</td></tr><tr><td>20.4</td><td>49</td></tr><tr><td>24.9</td><td>40</td></tr><tr><td>18.3</td><td>40</td></tr><tr><td>23.3</td><td>52</td></tr><tr><td>9.4</td><td>23</td></tr><tr><td>10.3</td><td>23</td></tr><tr><td>14.2</td><td>24</td></tr><tr><td>19.2</td><td>24</td></tr><tr><td>29.6</td><td>25</td></tr><tr><td>5.3</td><td>25</td></tr><tr><td>25.2</td><td>26</td></tr><tr><td>9.4</td><td>26</td></tr><tr><td>19.6</td><td>26</td></tr><tr><td>10.1</td><td>27</td></tr><tr><td>16.5</td><td>27</td></tr><tr><td>21.0</td><td>27</td></tr><tr><td>17.3</td><td>28</td></tr><tr><td>31.2</td><td>28</td></tr><tr><td>10.0</td><td>28</td></tr><tr><td>12.5</td><td>30</td></tr><tr><td>22.5</td><td>31</td></tr><tr><td>9.4</td><td>31</td></tr><tr><td>14.6</td><td>33</td></tr><tr><td>13.0</td><td>33</td></tr><tr><td>15.1</td><td>34</td></tr><tr><td>27.3</td><td>34</td></tr><tr><td>19.2</td><td>35</td></tr><tr><td>21.8</td><td>35</td></tr><tr><td>20.3</td><td>35</td></tr><tr><td>34.3</td><td>35</td></tr><tr><td>16.5</td><td>35</td></tr><tr><td>3.0</td><td>35</td></tr><tr><td>0.7</td><td>35</td></tr><tr><td>20.5</td><td>35</td></tr><tr><td>16.9</td><td>36</td></tr><tr><td>25.3</td><td>36</td></tr><tr><td>9.9</td><td>37</td></tr><tr><td>13.1</td><td>37</td></tr><tr><td>29.9</td><td>37</td></tr><tr><td>22.5</td><td>38</td></tr><tr><td>16.9</td><td>39</td></tr><tr><td>26.6</td><td>39</td></tr><tr><td>0.0</td><td>40</td></tr><tr><td>11.5</td><td>40</td></tr><tr><td>12.1</td><td>40</td></tr><tr><td>17.5</td><td>40</td></tr><tr><td>8.6</td><td>40</td></tr><tr><td>23.6</td><td>41</td></tr><tr><td>20.4</td><td>41</td></tr><tr><td>20.5</td><td>41</td></tr><tr><td>24.4</td><td>41</td></tr><tr><td>11.4</td><td>41</td></tr><tr><td>38.1</td><td>42</td></tr><tr><td>15.9</td><td>42</td></tr><tr><td>24.7</td><td>42</td></tr><tr><td>22.8</td><td>42</td></tr><tr><td>25.5</td><td>42</td></tr><tr><td>22.0</td><td>42</td></tr><tr><td>17.7</td><td>42</td></tr><tr><td>6.6</td><td>42</td></tr><tr><td>23.6</td><td>43</td></tr><tr><td>12.2</td><td>43</td></tr><tr><td>22.1</td><td>43</td></tr><tr><td>28.7</td><td>43</td></tr><tr><td>6.0</td><td>44</td></tr><tr><td>34.8</td><td>44</td></tr><tr><td>16.6</td><td>44</td></tr><tr><td>32.9</td><td>44</td></tr><tr><td>32.8</td><td>47</td></tr><tr><td>9.6</td><td>47</td></tr><tr><td>10.8</td><td>47</td></tr><tr><td>7.1</td><td>49</td></tr><tr><td>27.2</td><td>49</td></tr><tr><td>19.5</td><td>49</td></tr><tr><td>18.7</td><td>50</td></tr><tr><td>19.5</td><td>50</td></tr><tr><td>47.5</td><td>51</td></tr><tr><td>13.6</td><td>51</td></tr><tr><td>7.5</td><td>51</td></tr><tr><td>24.5</td><td>52</td></tr><tr><td>15.0</td><td>53</td></tr><tr><td>12.4</td><td>54</td></tr><tr><td>26.0</td><td>54</td></tr><tr><td>11.5</td><td>54</td></tr><tr><td>5.2</td><td>55</td></tr><tr><td>10.9</td><td>55</td></tr><tr><td>12.5</td><td>55</td></tr><tr><td>14.8</td><td>55</td></tr><tr><td>25.2</td><td>55</td></tr><tr><td>14.9</td><td>56</td></tr><tr><td>17.0</td><td>56</td></tr><tr><td>10.6</td><td>57</td></tr><tr><td>16.1</td><td>57</td></tr><tr><td>15.4</td><td>58</td></tr><tr><td>26.7</td><td>58</td></tr><tr><td>25.8</td><td>60</td></tr><tr><td>18.6</td><td>62</td></tr><tr><td>24.8</td><td>62</td></tr><tr><td>27.3</td><td>63</td></tr><tr><td>12.4</td><td>64</td></tr><tr><td>29.9</td><td>65</td></tr><tr><td>17.0</td><td>65</td></tr><tr><td>35.0</td><td>65</td></tr><tr><td>30.4</td><td>66</td></tr><tr><td>32.6</td><td>67</td></tr><tr><td>29.0</td><td>67</td></tr><tr><td>15.2</td><td>68</td></tr><tr><td>30.2</td><td>69</td></tr><tr><td>11.0</td><td>70</td></tr><tr><td>33.6</td><td>72</td></tr><tr><td>29.3</td><td>72</td></tr><tr><td>26.0</td><td>72</td></tr><tr><td>31.9</td><td>74</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"ef54a02f-eb20-4eeb-84f0-4edd3e4987a4\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT Weight, BodyFat FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Weight</th><th>BodyFat</th></tr></thead><tbody><tr><td>69.966566</td><td>12.3</td></tr><tr><td>78.584814</td><td>6.1</td></tr><tr><td>69.853168</td><td>25.3</td></tr><tr><td>83.80112199999999</td><td>10.4</td></tr><tr><td>83.574326</td><td>28.7</td></tr><tr><td>95.367718</td><td>20.9</td></tr><tr><td>82.100152</td><td>19.2</td></tr><tr><td>79.83219199999999</td><td>12.4</td></tr><tr><td>86.636072</td><td>4.1</td></tr><tr><td>89.924614</td><td>11.7</td></tr><tr><td>84.48151</td><td>7.1</td></tr><tr><td>97.975872</td><td>7.8</td></tr><tr><td>81.873356</td><td>20.8</td></tr><tr><td>93.099758</td><td>21.2</td></tr><tr><td>85.161898</td><td>22.1</td></tr><tr><td>73.822098</td><td>20.9</td></tr><tr><td>88.790634</td><td>29.0</td></tr><tr><td>94.914126</td><td>22.9</td></tr><tr><td>83.34753</td><td>16.0</td></tr><tr><td>96.048106</td><td>16.5</td></tr><tr><td>81.192968</td><td>19.1</td></tr><tr><td>90.945196</td><td>15.2</td></tr><tr><td>63.616278</td><td>15.6</td></tr><tr><td>67.47181</td><td>17.7</td></tr><tr><td>68.60579</td><td>14.0</td></tr><tr><td>72.234526</td><td>3.7</td></tr><tr><td>59.647348</td><td>7.9</td></tr><tr><td>67.131616</td><td>22.9</td></tr><tr><td>60.441134</td><td>3.7</td></tr><tr><td>72.914914</td><td>8.8</td></tr><tr><td>82.553744</td><td>11.9</td></tr><tr><td>72.688118</td><td>5.7</td></tr><tr><td>76.203456</td><td>11.8</td></tr><tr><td>99.109852</td><td>21.3</td></tr><tr><td>112.150622</td><td>32.3</td></tr><tr><td>86.976266</td><td>40.1</td></tr><tr><td>91.738982</td><td>24.2</td></tr><tr><td>89.244226</td><td>28.4</td></tr><tr><td>164.72193479999999</td><td>35.2</td></tr><tr><td>92.079176</td><td>32.6</td></tr><tr><td>119.181298</td><td>34.5</td></tr><tr><td>92.98636</td><td>32.9</td></tr><tr><td>98.429464</td><td>31.6</td></tr><tr><td>96.161504</td><td>32.0</td></tr><tr><td>56.812398</td><td>7.7</td></tr><tr><td>74.502486</td><td>13.9</td></tr><tr><td>60.554532</td><td>10.8</td></tr><tr><td>67.358412</td><td>5.6</td></tr><tr><td>61.575114</td><td>13.6</td></tr><tr><td>57.83298</td><td>4.0</td></tr><tr><td>71.780934</td><td>10.2</td></tr><tr><td>63.162686</td><td>6.6</td></tr><tr><td>62.255502</td><td>8.0</td></tr><tr><td>69.28617799999999</td><td>6.3</td></tr><tr><td>61.80191</td><td>3.9</td></tr><tr><td>89.811216</td><td>22.6</td></tr><tr><td>82.326948</td><td>20.4</td></tr><tr><td>91.28538999999999</td><td>28.0</td></tr><tr><td>91.85238</td><td>31.5</td></tr><tr><td>81.533162</td><td>24.6</td></tr><tr><td>97.975872</td><td>26.1</td></tr><tr><td>81.07957</td><td>29.8</td></tr><tr><td>87.656654</td><td>30.7</td></tr><tr><td>80.739376</td><td>25.8</td></tr><tr><td>93.213156</td><td>32.3</td></tr><tr><td>83.234132</td><td>30.0</td></tr><tr><td>68.719188</td><td>21.5</td></tr><tr><td>70.193362</td><td>13.8</td></tr><tr><td>70.420158</td><td>6.3</td></tr><tr><td>71.100546</td><td>12.9</td></tr><tr><td>75.97666</td><td>24.3</td></tr><tr><td>66.564626</td><td>8.8</td></tr><tr><td>72.914914</td><td>8.5</td></tr><tr><td>56.699</td><td>13.5</td></tr><tr><td>64.863656</td><td>11.8</td></tr><tr><td>67.245014</td><td>18.5</td></tr><tr><td>73.7087</td><td>8.8</td></tr><tr><td>80.625978</td><td>22.2</td></tr><tr><td>73.14171</td><td>21.5</td></tr><tr><td>77.67763</td><td>18.8</td></tr><tr><td>74.27569</td><td>31.4</td></tr><tr><td>68.152198</td><td>26.8</td></tr><tr><td>86.295878</td><td>18.4</td></tr><tr><td>77.450834</td><td>27.0</td></tr><tr><td>76.203456</td><td>27.0</td></tr><tr><td>75.749864</td><td>26.6</td></tr><tr><td>71.554138</td><td>14.9</td></tr><tr><td>72.57472</td><td>23.1</td></tr><tr><td>80.172386</td><td>8.3</td></tr><tr><td>79.83219199999999</td><td>14.1</td></tr><tr><td>80.28578399999999</td><td>20.5</td></tr><tr><td>81.533162</td><td>18.2</td></tr><tr><td>74.956078</td><td>8.5</td></tr><tr><td>87.31645999999999</td><td>24.9</td></tr><tr><td>83.574326</td><td>9.0</td></tr><tr><td>101.83140399999999</td><td>17.4</td></tr><tr><td>85.61549</td><td>9.6</td></tr><tr><td>73.7087</td><td>11.3</td></tr><tr><td>70.987148</td><td>17.8</td></tr><tr><td>89.357624</td><td>22.2</td></tr><tr><td>90.038012</td><td>21.2</td></tr><tr><td>78.81161</td><td>20.4</td></tr><tr><td>78.358018</td><td>20.1</td></tr><tr><td>89.244226</td><td>22.3</td></tr><tr><td>80.28578399999999</td><td>25.4</td></tr><tr><td>75.069476</td><td>18.0</td></tr><tr><td>90.83179799999999</td><td>19.3</td></tr><tr><td>92.192574</td><td>18.3</td></tr><tr><td>87.996848</td><td>17.3</td></tr><tr><td>76.430252</td><td>21.4</td></tr><tr><td>77.450834</td><td>19.7</td></tr><tr><td>83.120734</td><td>28.0</td></tr><tr><td>80.852774</td><td>22.1</td></tr><tr><td>73.935496</td><td>21.3</td></tr><tr><td>79.491998</td><td>26.7</td></tr><tr><td>71.667536</td><td>16.7</td></tr><tr><td>80.399182</td><td>20.1</td></tr><tr><td>81.192968</td><td>13.9</td></tr><tr><td>86.636072</td><td>25.8</td></tr><tr><td>85.0485</td><td>18.1</td></tr><tr><td>93.666748</td><td>27.9</td></tr><tr><td>84.027918</td><td>25.3</td></tr><tr><td>72.688118</td><td>14.7</td></tr><tr><td>68.719188</td><td>16.0</td></tr><tr><td>73.028312</td><td>13.8</td></tr><tr><td>75.749864</td><td>17.5</td></tr><tr><td>80.51258</td><td>27.2</td></tr><tr><td>69.059382</td><td>17.4</td></tr><tr><td>87.203062</td><td>20.8</td></tr><tr><td>74.956078</td><td>14.9</td></tr><tr><td>77.904426</td><td>18.1</td></tr><tr><td>77.67763</td><td>22.7</td></tr><tr><td>89.357624</td><td>23.6</td></tr><tr><td>71.213944</td><td>26.1</td></tr><tr><td>76.31685399999999</td><td>24.4</td></tr><tr><td>84.368112</td><td>27.1</td></tr><tr><td>75.636466</td><td>21.8</td></tr><tr><td>85.161898</td><td>29.4</td></tr><tr><td>76.31685399999999</td><td>22.4</td></tr><tr><td>96.501698</td><td>20.4</td></tr><tr><td>80.172386</td><td>24.9</td></tr><tr><td>78.584814</td><td>18.3</td></tr><tr><td>75.749864</td><td>23.3</td></tr><tr><td>72.461322</td><td>9.4</td></tr><tr><td>85.34333480000001</td><td>10.3</td></tr><tr><td>70.760352</td><td>14.2</td></tr><tr><td>94.573932</td><td>19.2</td></tr><tr><td>93.666748</td><td>29.6</td></tr><tr><td>65.20385</td><td>5.3</td></tr><tr><td>101.151016</td><td>25.2</td></tr><tr><td>69.059382</td><td>9.4</td></tr><tr><td>109.655866</td><td>19.6</td></tr><tr><td>66.224432</td><td>10.1</td></tr><tr><td>71.100546</td><td>16.5</td></tr><tr><td>90.83179799999999</td><td>21.0</td></tr><tr><td>77.791028</td><td>17.3</td></tr><tr><td>93.326554</td><td>31.2</td></tr><tr><td>82.78054</td><td>10.0</td></tr><tr><td>61.915307999999996</td><td>12.5</td></tr><tr><td>80.399182</td><td>22.5</td></tr><tr><td>68.60579</td><td>9.4</td></tr><tr><td>88.904032</td><td>14.6</td></tr><tr><td>83.574326</td><td>13.0</td></tr><tr><td>63.50288</td><td>15.1</td></tr><tr><td>99.22325</td><td>27.3</td></tr><tr><td>98.429464</td><td>19.2</td></tr><tr><td>75.40967</td><td>21.8</td></tr><tr><td>101.944802</td><td>20.3</td></tr><tr><td>103.532374</td><td>34.3</td></tr><tr><td>78.358018</td><td>16.5</td></tr><tr><td>69.059382</td><td>3.0</td></tr><tr><td>57.039194</td><td>0.7</td></tr><tr><td>80.399182</td><td>20.5</td></tr><tr><td>79.94559</td><td>16.9</td></tr><tr><td>102.851986</td><td>25.3</td></tr><tr><td>65.884238</td><td>9.9</td></tr><tr><td>68.492392</td><td>13.1</td></tr><tr><td>109.42907</td><td>29.9</td></tr><tr><td>84.935102</td><td>22.5</td></tr><tr><td>106.480722</td><td>16.9</td></tr><tr><td>99.450046</td><td>26.6</td></tr><tr><td>53.750652</td><td>0.0</td></tr><tr><td>66.111034</td><td>11.5</td></tr><tr><td>72.234526</td><td>12.1</td></tr><tr><td>77.337436</td><td>17.5</td></tr><tr><td>75.97666</td><td>8.6</td></tr><tr><td>105.573538</td><td>23.6</td></tr><tr><td>95.481116</td><td>20.4</td></tr><tr><td>91.738982</td><td>20.5</td></tr><tr><td>83.91452</td><td>24.4</td></tr><tr><td>69.399576</td><td>11.4</td></tr><tr><td>110.789846</td><td>38.1</td></tr><tr><td>87.77005199999999</td><td>15.9</td></tr><tr><td>101.944802</td><td>24.7</td></tr><tr><td>73.822098</td><td>22.8</td></tr><tr><td>81.64656</td><td>25.5</td></tr><tr><td>70.87375</td><td>22.0</td></tr><tr><td>76.203456</td><td>17.7</td></tr><tr><td>75.863262</td><td>6.6</td></tr><tr><td>77.450834</td><td>23.6</td></tr><tr><td>80.852774</td><td>12.2</td></tr><tr><td>68.0388</td><td>22.1</td></tr><tr><td>90.945196</td><td>28.7</td></tr><tr><td>83.460928</td><td>6.0</td></tr><tr><td>101.151016</td><td>34.8</td></tr><tr><td>94.68733</td><td>16.6</td></tr><tr><td>75.296272</td><td>32.9</td></tr><tr><td>88.45044</td><td>32.8</td></tr><tr><td>72.80151599999999</td><td>9.6</td></tr><tr><td>72.461322</td><td>10.8</td></tr><tr><td>63.729676</td><td>7.1</td></tr><tr><td>98.08927</td><td>27.2</td></tr><tr><td>76.31685399999999</td><td>19.5</td></tr><tr><td>88.337042</td><td>18.7</td></tr><tr><td>78.358018</td><td>19.5</td></tr><tr><td>99.336648</td><td>47.5</td></tr><tr><td>67.698606</td><td>13.6</td></tr><tr><td>70.079964</td><td>7.5</td></tr><tr><td>90.378206</td><td>24.5</td></tr><tr><td>70.079964</td><td>15.0</td></tr><tr><td>69.512974</td><td>12.4</td></tr><tr><td>104.32616</td><td>26.0</td></tr><tr><td>73.368506</td><td>11.5</td></tr><tr><td>64.523462</td><td>5.2</td></tr><tr><td>81.533162</td><td>10.9</td></tr><tr><td>57.379388</td><td>12.5</td></tr><tr><td>76.883844</td><td>14.8</td></tr><tr><td>90.038012</td><td>25.2</td></tr><tr><td>79.151804</td><td>14.9</td></tr><tr><td>76.090058</td><td>17.0</td></tr><tr><td>67.018218</td><td>10.6</td></tr><tr><td>82.667142</td><td>16.1</td></tr><tr><td>79.605396</td><td>15.4</td></tr><tr><td>73.368506</td><td>26.7</td></tr><tr><td>71.554138</td><td>25.8</td></tr><tr><td>76.54365</td><td>18.6</td></tr><tr><td>86.862868</td><td>24.8</td></tr><tr><td>99.40468680000001</td><td>27.3</td></tr><tr><td>70.420158</td><td>12.4</td></tr><tr><td>86.069082</td><td>29.9</td></tr><tr><td>57.83298</td><td>17.0</td></tr><tr><td>101.83140399999999</td><td>35.0</td></tr><tr><td>106.25392599999999</td><td>30.4</td></tr><tr><td>103.305578</td><td>32.6</td></tr><tr><td>90.491604</td><td>29.0</td></tr><tr><td>70.533556</td><td>15.2</td></tr><tr><td>97.749076</td><td>30.2</td></tr><tr><td>60.894726</td><td>11.0</td></tr><tr><td>91.171992</td><td>33.6</td></tr><tr><td>84.708306</td><td>29.3</td></tr><tr><td>86.522674</td><td>26.0</td></tr><tr><td>94.12034</td><td>31.9</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"704d7278-a930-4473-b4fe-a7184b3ae00e\",\"showTitle\":false,\"title\":\"\"}"}
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"e3a1c678-98d1-429a-a8f2-2eb2b50a317c\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT BodyFat, Height FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>BodyFat</th><th>Height</th></tr></thead><tbody><tr><td>12.3</td><td>1.72085</td></tr><tr><td>6.1</td><td>1.8351499999999998</td></tr><tr><td>25.3</td><td>1.68275</td></tr><tr><td>10.4</td><td>1.8351499999999998</td></tr><tr><td>28.7</td><td>1.80975</td></tr><tr><td>20.9</td><td>1.89865</td></tr><tr><td>19.2</td><td>1.77165</td></tr><tr><td>12.4</td><td>1.8415</td></tr><tr><td>4.1</td><td>1.8796</td></tr><tr><td>11.7</td><td>1.8669</td></tr><tr><td>7.1</td><td>1.8922999999999999</td></tr><tr><td>7.8</td><td>1.9304</td></tr><tr><td>20.8</td><td>1.7652999999999999</td></tr><tr><td>21.2</td><td>1.80975</td></tr><tr><td>22.1</td><td>1.7652999999999999</td></tr><tr><td>20.9</td><td>1.6764</td></tr><tr><td>29.0</td><td>1.8034</td></tr><tr><td>22.9</td><td>1.8034</td></tr><tr><td>16.0</td><td>1.72085</td></tr><tr><td>16.5</td><td>1.8669</td></tr><tr><td>19.1</td><td>1.7271999999999998</td></tr><tr><td>15.2</td><td>1.77165</td></tr><tr><td>15.6</td><td>1.73355</td></tr><tr><td>17.7</td><td>1.778</td></tr><tr><td>14.0</td><td>1.72085</td></tr><tr><td>3.7</td><td>1.8160999999999998</td></tr><tr><td>7.9</td><td>1.7145</td></tr><tr><td>22.9</td><td>1.7145</td></tr><tr><td>3.7</td><td>1.64465</td></tr><tr><td>8.8</td><td>1.7526</td></tr><tr><td>11.9</td><td>1.8732499999999999</td></tr><tr><td>5.7</td><td>1.80975</td></tr><tr><td>11.8</td><td>1.80975</td></tr><tr><td>21.3</td><td>1.8034</td></tr><tr><td>32.3</td><td>1.8669</td></tr><tr><td>40.1</td><td>1.651</td></tr><tr><td>24.2</td><td>1.778</td></tr><tr><td>28.4</td><td>1.73355</td></tr><tr><td>35.2</td><td>1.8351499999999998</td></tr><tr><td>32.6</td><td>1.7018</td></tr><tr><td>34.5</td><td>1.7462499999999999</td></tr><tr><td>32.9</td><td>0.7493</td></tr><tr><td>31.6</td><td>1.778</td></tr><tr><td>32.0</td><td>1.8160999999999998</td></tr><tr><td>7.7</td><td>1.7271999999999998</td></tr><tr><td>13.9</td><td>1.86055</td></tr><tr><td>10.8</td><td>1.7145</td></tr><tr><td>5.6</td><td>1.80975</td></tr><tr><td>13.6</td><td>1.7399</td></tr><tr><td>4.0</td><td>1.69545</td></tr><tr><td>10.2</td><td>1.8351499999999998</td></tr><tr><td>6.6</td><td>1.7526</td></tr><tr><td>8.0</td><td>1.72085</td></tr><tr><td>6.3</td><td>1.8669</td></tr><tr><td>3.9</td><td>1.7145</td></tr><tr><td>22.6</td><td>1.8288</td></tr><tr><td>20.4</td><td>1.7271999999999998</td></tr><tr><td>28.0</td><td>1.7652999999999999</td></tr><tr><td>31.5</td><td>1.79705</td></tr><tr><td>24.6</td><td>1.67005</td></tr><tr><td>26.1</td><td>1.86055</td></tr><tr><td>29.8</td><td>1.7399</td></tr><tr><td>30.7</td><td>1.7843499999999999</td></tr><tr><td>25.8</td><td>1.7018</td></tr><tr><td>32.3</td><td>1.778</td></tr><tr><td>30.0</td><td>1.7145</td></tr><tr><td>21.5</td><td>1.79705</td></tr><tr><td>13.8</td><td>1.8160999999999998</td></tr><tr><td>6.3</td><td>1.75895</td></tr><tr><td>12.9</td><td>1.8160999999999998</td></tr><tr><td>24.3</td><td>1.8160999999999998</td></tr><tr><td>8.8</td><td>1.7462499999999999</td></tr><tr><td>8.5</td><td>1.8732499999999999</td></tr><tr><td>13.5</td><td>1.6256</td></tr><tr><td>11.8</td><td>1.67005</td></tr><tr><td>18.5</td><td>1.7145</td></tr><tr><td>8.8</td><td>1.7652999999999999</td></tr><tr><td>22.2</td><td>1.7399</td></tr><tr><td>21.5</td><td>1.7843499999999999</td></tr><tr><td>18.8</td><td>1.75895</td></tr><tr><td>31.4</td><td>1.72085</td></tr><tr><td>26.8</td><td>1.7081499999999998</td></tr><tr><td>18.4</td><td>1.84785</td></tr><tr><td>27.0</td><td>1.778</td></tr><tr><td>27.0</td><td>1.75895</td></tr><tr><td>26.6</td><td>1.7145</td></tr><tr><td>14.9</td><td>1.7081499999999998</td></tr><tr><td>23.1</td><td>1.67005</td></tr><tr><td>8.3</td><td>1.8415</td></tr><tr><td>14.1</td><td>1.8541999999999998</td></tr><tr><td>20.5</td><td>1.778</td></tr><tr><td>18.2</td><td>1.7652999999999999</td></tr><tr><td>8.5</td><td>1.7907</td></tr><tr><td>24.9</td><td>1.82245</td></tr><tr><td>9.0</td><td>1.8922999999999999</td></tr><tr><td>17.4</td><td>1.97485</td></tr><tr><td>9.6</td><td>1.86055</td></tr><tr><td>11.3</td><td>1.6890999999999998</td></tr><tr><td>17.8</td><td>1.73355</td></tr><tr><td>22.2</td><td>1.8288</td></tr><tr><td>21.2</td><td>1.8669</td></tr><tr><td>20.4</td><td>1.8288</td></tr><tr><td>20.1</td><td>1.80975</td></tr><tr><td>22.3</td><td>1.8732499999999999</td></tr><tr><td>25.4</td><td>1.75895</td></tr><tr><td>18.0</td><td>1.7399</td></tr><tr><td>19.3</td><td>1.8669</td></tr><tr><td>18.3</td><td>1.88595</td></tr><tr><td>17.3</td><td>1.9177</td></tr><tr><td>21.4</td><td>1.75895</td></tr><tr><td>19.7</td><td>1.7399</td></tr><tr><td>28.0</td><td>1.778</td></tr><tr><td>22.1</td><td>1.778</td></tr><tr><td>21.3</td><td>1.7843499999999999</td></tr><tr><td>26.7</td><td>1.82245</td></tr><tr><td>16.7</td><td>1.75895</td></tr><tr><td>20.1</td><td>1.84785</td></tr><tr><td>13.9</td><td>1.8288</td></tr><tr><td>25.8</td><td>1.8796</td></tr><tr><td>18.1</td><td>1.8351499999999998</td></tr><tr><td>27.9</td><td>1.8922999999999999</td></tr><tr><td>25.3</td><td>1.8160999999999998</td></tr><tr><td>14.7</td><td>1.7462499999999999</td></tr><tr><td>16.0</td><td>1.69545</td></tr><tr><td>13.8</td><td>1.6890999999999998</td></tr><tr><td>17.5</td><td>1.7018</td></tr><tr><td>27.2</td><td>1.7462499999999999</td></tr><tr><td>17.4</td><td>1.72085</td></tr><tr><td>20.8</td><td>1.86055</td></tr><tr><td>14.9</td><td>1.77165</td></tr><tr><td>18.1</td><td>1.8160999999999998</td></tr><tr><td>22.7</td><td>1.7907</td></tr><tr><td>23.6</td><td>1.86055</td></tr><tr><td>26.1</td><td>1.69545</td></tr><tr><td>24.4</td><td>1.7652999999999999</td></tr><tr><td>27.1</td><td>1.77165</td></tr><tr><td>21.8</td><td>1.79705</td></tr><tr><td>29.4</td><td>1.8796</td></tr><tr><td>22.4</td><td>1.80975</td></tr><tr><td>20.4</td><td>1.905</td></tr><tr><td>24.9</td><td>1.8034</td></tr><tr><td>18.3</td><td>1.7652999999999999</td></tr><tr><td>23.3</td><td>1.72085</td></tr><tr><td>9.4</td><td>1.8351499999999998</td></tr><tr><td>10.3</td><td>1.9685</td></tr><tr><td>14.2</td><td>1.79705</td></tr><tr><td>19.2</td><td>1.84785</td></tr><tr><td>29.6</td><td>1.77165</td></tr><tr><td>5.3</td><td>1.8415</td></tr><tr><td>25.2</td><td>1.7843499999999999</td></tr><tr><td>9.4</td><td>1.7526</td></tr><tr><td>19.6</td><td>1.8922999999999999</td></tr><tr><td>10.1</td><td>1.8351499999999998</td></tr><tr><td>16.5</td><td>1.7081499999999998</td></tr><tr><td>21.0</td><td>1.8669</td></tr><tr><td>17.3</td><td>1.9113499999999999</td></tr><tr><td>31.2</td><td>1.7526</td></tr><tr><td>10.0</td><td>1.8351499999999998</td></tr><tr><td>12.5</td><td>1.7462499999999999</td></tr><tr><td>22.5</td><td>1.8160999999999998</td></tr><tr><td>9.4</td><td>1.8351499999999998</td></tr><tr><td>14.6</td><td>1.8541999999999998</td></tr><tr><td>13.0</td><td>1.7462499999999999</td></tr><tr><td>15.1</td><td>1.7907</td></tr><tr><td>27.3</td><td>1.8288</td></tr><tr><td>19.2</td><td>1.8732499999999999</td></tr><tr><td>21.8</td><td>1.7271999999999998</td></tr><tr><td>20.3</td><td>1.8351499999999998</td></tr><tr><td>34.3</td><td>1.7652999999999999</td></tr><tr><td>16.5</td><td>1.7652999999999999</td></tr><tr><td>3.0</td><td>1.72085</td></tr><tr><td>0.7</td><td>1.6637</td></tr><tr><td>20.5</td><td>1.8034</td></tr><tr><td>16.9</td><td>1.8160999999999998</td></tr><tr><td>25.3</td><td>1.82245</td></tr><tr><td>9.9</td><td>1.75895</td></tr><tr><td>13.1</td><td>1.7018</td></tr><tr><td>29.9</td><td>1.8160999999999998</td></tr><tr><td>22.5</td><td>1.75895</td></tr><tr><td>16.9</td><td>1.8922999999999999</td></tr><tr><td>26.6</td><td>1.88595</td></tr><tr><td>0.0</td><td>1.7271999999999998</td></tr><tr><td>11.5</td><td>1.7081499999999998</td></tr><tr><td>12.1</td><td>1.77165</td></tr><tr><td>17.5</td><td>1.88595</td></tr><tr><td>8.6</td><td>1.8160999999999998</td></tr><tr><td>23.6</td><td>1.88595</td></tr><tr><td>20.4</td><td>1.8288</td></tr><tr><td>20.5</td><td>1.8415</td></tr><tr><td>24.4</td><td>1.73355</td></tr><tr><td>11.4</td><td>1.75895</td></tr><tr><td>38.1</td><td>1.9304</td></tr><tr><td>15.9</td><td>1.7907</td></tr><tr><td>24.7</td><td>1.89865</td></tr><tr><td>22.8</td><td>1.84785</td></tr><tr><td>25.5</td><td>1.73355</td></tr><tr><td>22.0</td><td>1.7526</td></tr><tr><td>17.7</td><td>1.8160999999999998</td></tr><tr><td>6.6</td><td>1.84785</td></tr><tr><td>23.6</td><td>1.7145</td></tr><tr><td>12.2</td><td>1.7843499999999999</td></tr><tr><td>22.1</td><td>1.75895</td></tr><tr><td>28.7</td><td>1.8160999999999998</td></tr><tr><td>6.0</td><td>1.8796</td></tr><tr><td>34.8</td><td>1.77165</td></tr><tr><td>16.6</td><td>1.8541999999999998</td></tr><tr><td>32.9</td><td>1.6637</td></tr><tr><td>32.8</td><td>1.8415</td></tr><tr><td>9.6</td><td>1.7843499999999999</td></tr><tr><td>10.8</td><td>1.79705</td></tr><tr><td>7.1</td><td>1.7271999999999998</td></tr><tr><td>27.2</td><td>1.8922999999999999</td></tr><tr><td>19.5</td><td>1.82245</td></tr><tr><td>18.7</td><td>1.79705</td></tr><tr><td>19.5</td><td>1.8541999999999998</td></tr><tr><td>47.5</td><td>1.6256</td></tr><tr><td>13.6</td><td>1.77165</td></tr><tr><td>7.5</td><td>1.778</td></tr><tr><td>24.5</td><td>1.82245</td></tr><tr><td>15.0</td><td>1.75895</td></tr><tr><td>12.4</td><td>1.7907</td></tr><tr><td>26.0</td><td>1.8351499999999998</td></tr><tr><td>11.5</td><td>1.7145</td></tr><tr><td>5.2</td><td>1.7081499999999998</td></tr><tr><td>10.9</td><td>1.7462499999999999</td></tr><tr><td>12.5</td><td>1.69545</td></tr><tr><td>14.8</td><td>1.73355</td></tr><tr><td>25.2</td><td>1.88595</td></tr><tr><td>14.9</td><td>1.7652999999999999</td></tr><tr><td>17.0</td><td>1.7399</td></tr><tr><td>10.6</td><td>1.67005</td></tr><tr><td>16.1</td><td>1.82245</td></tr><tr><td>15.4</td><td>1.8160999999999998</td></tr><tr><td>26.7</td><td>1.7081499999999998</td></tr><tr><td>25.8</td><td>1.7145</td></tr><tr><td>18.6</td><td>1.7145</td></tr><tr><td>24.8</td><td>1.8351499999999998</td></tr><tr><td>27.3</td><td>1.7652999999999999</td></tr><tr><td>12.4</td><td>1.7652999999999999</td></tr><tr><td>29.9</td><td>1.67005</td></tr><tr><td>17.0</td><td>1.67005</td></tr><tr><td>35.0</td><td>1.73355</td></tr><tr><td>30.4</td><td>1.8288</td></tr><tr><td>32.6</td><td>1.84785</td></tr><tr><td>29.0</td><td>1.7399</td></tr><tr><td>15.2</td><td>1.75895</td></tr><tr><td>30.2</td><td>1.7907</td></tr><tr><td>11.0</td><td>1.7018</td></tr><tr><td>33.6</td><td>1.77165</td></tr><tr><td>29.3</td><td>1.6764</td></tr><tr><td>26.0</td><td>1.7907</td></tr><tr><td>31.9</td><td>1.778</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"17630334-7bf4-452e-9844-42060e1e40f1\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT Neck, BodyFat FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Chest</th><th>BodyFat</th></tr></thead><tbody><tr><td>93.1</td><td>12.3</td></tr><tr><td>93.6</td><td>6.1</td></tr><tr><td>95.8</td><td>25.3</td></tr><tr><td>101.8</td><td>10.4</td></tr><tr><td>97.3</td><td>28.7</td></tr><tr><td>104.5</td><td>20.9</td></tr><tr><td>105.1</td><td>19.2</td></tr><tr><td>99.6</td><td>12.4</td></tr><tr><td>100.9</td><td>4.1</td></tr><tr><td>99.6</td><td>11.7</td></tr><tr><td>101.5</td><td>7.1</td></tr><tr><td>103.6</td><td>7.8</td></tr><tr><td>102.0</td><td>20.8</td></tr><tr><td>104.1</td><td>21.2</td></tr><tr><td>101.3</td><td>22.1</td></tr><tr><td>99.1</td><td>20.9</td></tr><tr><td>101.9</td><td>29.0</td></tr><tr><td>107.6</td><td>22.9</td></tr><tr><td>106.8</td><td>16.0</td></tr><tr><td>106.2</td><td>16.5</td></tr><tr><td>103.3</td><td>19.1</td></tr><tr><td>111.4</td><td>15.2</td></tr><tr><td>86.0</td><td>15.6</td></tr><tr><td>86.7</td><td>17.7</td></tr><tr><td>90.2</td><td>14.0</td></tr><tr><td>89.6</td><td>3.7</td></tr><tr><td>88.6</td><td>7.9</td></tr><tr><td>97.4</td><td>22.9</td></tr><tr><td>93.5</td><td>3.7</td></tr><tr><td>97.4</td><td>8.8</td></tr><tr><td>100.5</td><td>11.9</td></tr><tr><td>93.5</td><td>5.7</td></tr><tr><td>93.0</td><td>11.8</td></tr><tr><td>111.7</td><td>21.3</td></tr><tr><td>117.0</td><td>32.3</td></tr><tr><td>118.5</td><td>40.1</td></tr><tr><td>106.5</td><td>24.2</td></tr><tr><td>105.6</td><td>28.4</td></tr><tr><td>136.2</td><td>35.2</td></tr><tr><td>114.8</td><td>32.6</td></tr><tr><td>128.3</td><td>34.5</td></tr><tr><td>106.0</td><td>32.9</td></tr><tr><td>113.3</td><td>31.6</td></tr><tr><td>106.6</td><td>32.0</td></tr><tr><td>85.1</td><td>7.7</td></tr><tr><td>96.6</td><td>13.9</td></tr><tr><td>88.2</td><td>10.8</td></tr><tr><td>89.8</td><td>5.6</td></tr><tr><td>92.3</td><td>13.6</td></tr><tr><td>83.4</td><td>4.0</td></tr><tr><td>90.2</td><td>10.2</td></tr><tr><td>89.2</td><td>6.6</td></tr><tr><td>89.7</td><td>8.0</td></tr><tr><td>93.3</td><td>6.3</td></tr><tr><td>87.6</td><td>3.9</td></tr><tr><td>107.6</td><td>22.6</td></tr><tr><td>100.0</td><td>20.4</td></tr><tr><td>111.5</td><td>28.0</td></tr><tr><td>115.4</td><td>31.5</td></tr><tr><td>104.8</td><td>24.6</td></tr><tr><td>112.3</td><td>26.1</td></tr><tr><td>102.9</td><td>29.8</td></tr><tr><td>107.6</td><td>30.7</td></tr><tr><td>105.3</td><td>25.8</td></tr><tr><td>105.3</td><td>32.3</td></tr><tr><td>103.0</td><td>30.0</td></tr><tr><td>90.0</td><td>21.5</td></tr><tr><td>95.4</td><td>13.8</td></tr><tr><td>89.3</td><td>6.3</td></tr><tr><td>94.4</td><td>12.9</td></tr><tr><td>97.6</td><td>24.3</td></tr><tr><td>88.5</td><td>8.8</td></tr><tr><td>93.6</td><td>8.5</td></tr><tr><td>87.7</td><td>13.5</td></tr><tr><td>93.4</td><td>11.8</td></tr><tr><td>91.6</td><td>18.5</td></tr><tr><td>91.6</td><td>8.8</td></tr><tr><td>102.0</td><td>22.2</td></tr><tr><td>96.4</td><td>21.5</td></tr><tr><td>102.7</td><td>18.8</td></tr><tr><td>97.7</td><td>31.4</td></tr><tr><td>97.1</td><td>26.8</td></tr><tr><td>103.1</td><td>18.4</td></tr><tr><td>101.8</td><td>27.0</td></tr><tr><td>101.4</td><td>27.0</td></tr><tr><td>98.9</td><td>26.6</td></tr><tr><td>97.5</td><td>14.9</td></tr><tr><td>104.3</td><td>23.1</td></tr><tr><td>97.3</td><td>8.3</td></tr><tr><td>96.7</td><td>14.1</td></tr><tr><td>99.7</td><td>20.5</td></tr><tr><td>101.9</td><td>18.2</td></tr><tr><td>97.2</td><td>8.5</td></tr><tr><td>106.6</td><td>24.9</td></tr><tr><td>99.6</td><td>9.0</td></tr><tr><td>113.2</td><td>17.4</td></tr><tr><td>99.1</td><td>9.6</td></tr><tr><td>99.4</td><td>11.3</td></tr><tr><td>95.1</td><td>17.8</td></tr><tr><td>107.5</td><td>22.2</td></tr><tr><td>106.5</td><td>21.2</td></tr><tr><td>99.1</td><td>20.4</td></tr><tr><td>96.7</td><td>20.1</td></tr><tr><td>103.5</td><td>22.3</td></tr><tr><td>104.0</td><td>25.4</td></tr><tr><td>93.1</td><td>18.0</td></tr><tr><td>105.2</td><td>19.3</td></tr><tr><td>110.0</td><td>18.3</td></tr><tr><td>110.1</td><td>17.3</td></tr><tr><td>97.8</td><td>21.4</td></tr><tr><td>96.3</td><td>19.7</td></tr><tr><td>108.0</td><td>28.0</td></tr><tr><td>99.7</td><td>22.1</td></tr><tr><td>93.5</td><td>21.3</td></tr><tr><td>100.7</td><td>26.7</td></tr><tr><td>97.0</td><td>16.7</td></tr><tr><td>96.0</td><td>20.1</td></tr><tr><td>99.2</td><td>13.9</td></tr><tr><td>95.4</td><td>25.8</td></tr><tr><td>101.8</td><td>18.1</td></tr><tr><td>104.3</td><td>27.9</td></tr><tr><td>99.2</td><td>25.3</td></tr><tr><td>99.3</td><td>14.7</td></tr><tr><td>94.0</td><td>16.0</td></tr><tr><td>98.9</td><td>13.8</td></tr><tr><td>101.0</td><td>17.5</td></tr><tr><td>98.7</td><td>27.2</td></tr><tr><td>95.9</td><td>17.4</td></tr><tr><td>103.9</td><td>20.8</td></tr><tr><td>96.2</td><td>14.9</td></tr><tr><td>97.8</td><td>18.1</td></tr><tr><td>94.6</td><td>22.7</td></tr><tr><td>103.6</td><td>23.6</td></tr><tr><td>100.4</td><td>26.1</td></tr><tr><td>98.4</td><td>24.4</td></tr><tr><td>104.6</td><td>27.1</td></tr><tr><td>92.9</td><td>21.8</td></tr><tr><td>97.8</td><td>29.4</td></tr><tr><td>98.3</td><td>22.4</td></tr><tr><td>104.7</td><td>20.4</td></tr><tr><td>98.6</td><td>24.9</td></tr><tr><td>99.5</td><td>18.3</td></tr><tr><td>102.7</td><td>23.3</td></tr><tr><td>92.1</td><td>9.4</td></tr><tr><td>96.6</td><td>10.3</td></tr><tr><td>92.7</td><td>14.2</td></tr><tr><td>102.0</td><td>19.2</td></tr><tr><td>110.9</td><td>29.6</td></tr><tr><td>92.3</td><td>5.3</td></tr><tr><td>114.1</td><td>25.2</td></tr><tr><td>92.9</td><td>9.4</td></tr><tr><td>108.3</td><td>19.6</td></tr><tr><td>88.5</td><td>10.1</td></tr><tr><td>94.0</td><td>16.5</td></tr><tr><td>101.1</td><td>21.0</td></tr><tr><td>92.1</td><td>17.3</td></tr><tr><td>105.6</td><td>31.2</td></tr><tr><td>98.5</td><td>10.0</td></tr><tr><td>88.7</td><td>12.5</td></tr><tr><td>101.1</td><td>22.5</td></tr><tr><td>94.0</td><td>9.4</td></tr><tr><td>103.8</td><td>14.6</td></tr><tr><td>98.9</td><td>13.0</td></tr><tr><td>89.2</td><td>15.1</td></tr><tr><td>111.4</td><td>27.3</td></tr><tr><td>107.5</td><td>19.2</td></tr><tr><td>99.1</td><td>21.8</td></tr><tr><td>108.2</td><td>20.3</td></tr><tr><td>114.9</td><td>34.3</td></tr><tr><td>99.1</td><td>16.5</td></tr><tr><td>92.2</td><td>3.0</td></tr><tr><td>90.8</td><td>0.7</td></tr><tr><td>100.5</td><td>20.5</td></tr><tr><td>98.2</td><td>16.9</td></tr><tr><td>115.3</td><td>25.3</td></tr><tr><td>96.8</td><td>9.9</td></tr><tr><td>92.6</td><td>13.1</td></tr><tr><td>119.2</td><td>29.9</td></tr><tr><td>102.7</td><td>22.5</td></tr><tr><td>109.5</td><td>16.9</td></tr><tr><td>108.5</td><td>26.6</td></tr><tr><td>79.3</td><td>0.0</td></tr><tr><td>95.5</td><td>11.5</td></tr><tr><td>92.3</td><td>12.1</td></tr><tr><td>98.9</td><td>17.5</td></tr><tr><td>89.5</td><td>8.6</td></tr><tr><td>117.5</td><td>23.6</td></tr><tr><td>107.4</td><td>20.4</td></tr><tr><td>109.2</td><td>20.5</td></tr><tr><td>103.4</td><td>24.4</td></tr><tr><td>91.4</td><td>11.4</td></tr><tr><td>115.2</td><td>38.1</td></tr><tr><td>104.9</td><td>15.9</td></tr><tr><td>106.7</td><td>24.7</td></tr><tr><td>92.2</td><td>22.8</td></tr><tr><td>101.6</td><td>25.5</td></tr><tr><td>97.8</td><td>22.0</td></tr><tr><td>92.0</td><td>17.7</td></tr><tr><td>94.0</td><td>6.6</td></tr><tr><td>103.7</td><td>23.6</td></tr><tr><td>102.7</td><td>12.2</td></tr><tr><td>91.1</td><td>22.1</td></tr><tr><td>107.2</td><td>28.7</td></tr><tr><td>100.8</td><td>6.0</td></tr><tr><td>121.6</td><td>34.8</td></tr><tr><td>105.6</td><td>16.6</td></tr><tr><td>100.6</td><td>32.9</td></tr><tr><td>102.7</td><td>32.8</td></tr><tr><td>99.8</td><td>9.6</td></tr><tr><td>92.9</td><td>10.8</td></tr><tr><td>91.2</td><td>7.1</td></tr><tr><td>115.6</td><td>27.2</td></tr><tr><td>98.3</td><td>19.5</td></tr><tr><td>103.7</td><td>18.7</td></tr><tr><td>98.7</td><td>19.5</td></tr><tr><td>119.8</td><td>47.5</td></tr><tr><td>92.8</td><td>13.6</td></tr><tr><td>93.3</td><td>7.5</td></tr><tr><td>106.8</td><td>24.5</td></tr><tr><td>93.9</td><td>15.0</td></tr><tr><td>99.0</td><td>12.4</td></tr><tr><td>119.9</td><td>26.0</td></tr><tr><td>94.2</td><td>11.5</td></tr><tr><td>92.7</td><td>5.2</td></tr><tr><td>106.9</td><td>10.9</td></tr><tr><td>88.8</td><td>12.5</td></tr><tr><td>101.7</td><td>14.8</td></tr><tr><td>105.3</td><td>25.2</td></tr><tr><td>104.0</td><td>14.9</td></tr><tr><td>98.6</td><td>17.0</td></tr><tr><td>99.6</td><td>10.6</td></tr><tr><td>103.4</td><td>16.1</td></tr><tr><td>100.2</td><td>15.4</td></tr><tr><td>94.9</td><td>26.7</td></tr><tr><td>97.2</td><td>25.8</td></tr><tr><td>104.7</td><td>18.6</td></tr><tr><td>104.0</td><td>24.8</td></tr><tr><td>117.6</td><td>27.3</td></tr><tr><td>95.8</td><td>12.4</td></tr><tr><td>106.4</td><td>29.9</td></tr><tr><td>93.0</td><td>17.0</td></tr><tr><td>119.6</td><td>35.0</td></tr><tr><td>119.7</td><td>30.4</td></tr><tr><td>115.8</td><td>32.6</td></tr><tr><td>118.3</td><td>29.0</td></tr><tr><td>97.4</td><td>15.2</td></tr><tr><td>113.7</td><td>30.2</td></tr><tr><td>89.2</td><td>11.0</td></tr><tr><td>108.5</td><td>33.6</td></tr><tr><td>111.1</td><td>29.3</td></tr><tr><td>108.3</td><td>26.0</td></tr><tr><td>112.4</td><td>31.9</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"7045d8be-84a5-47ac-9696-b51aab744803\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT Abdomen, BodyFat FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Abdomen</th><th>BodyFat</th></tr></thead><tbody><tr><td>85.2</td><td>12.3</td></tr><tr><td>83.0</td><td>6.1</td></tr><tr><td>87.9</td><td>25.3</td></tr><tr><td>86.4</td><td>10.4</td></tr><tr><td>100.0</td><td>28.7</td></tr><tr><td>94.4</td><td>20.9</td></tr><tr><td>90.7</td><td>19.2</td></tr><tr><td>88.5</td><td>12.4</td></tr><tr><td>82.5</td><td>4.1</td></tr><tr><td>88.6</td><td>11.7</td></tr><tr><td>83.6</td><td>7.1</td></tr><tr><td>90.9</td><td>7.8</td></tr><tr><td>91.6</td><td>20.8</td></tr><tr><td>101.8</td><td>21.2</td></tr><tr><td>96.4</td><td>22.1</td></tr><tr><td>92.8</td><td>20.9</td></tr><tr><td>96.4</td><td>29.0</td></tr><tr><td>97.5</td><td>22.9</td></tr><tr><td>89.6</td><td>16.0</td></tr><tr><td>100.5</td><td>16.5</td></tr><tr><td>95.9</td><td>19.1</td></tr><tr><td>98.8</td><td>15.2</td></tr><tr><td>76.4</td><td>15.6</td></tr><tr><td>80.0</td><td>17.7</td></tr><tr><td>76.3</td><td>14.0</td></tr><tr><td>79.7</td><td>3.7</td></tr><tr><td>74.6</td><td>7.9</td></tr><tr><td>88.7</td><td>22.9</td></tr><tr><td>73.9</td><td>3.7</td></tr><tr><td>83.5</td><td>8.8</td></tr><tr><td>88.7</td><td>11.9</td></tr><tr><td>84.5</td><td>5.7</td></tr><tr><td>79.1</td><td>11.8</td></tr><tr><td>100.5</td><td>21.3</td></tr><tr><td>115.6</td><td>32.3</td></tr><tr><td>113.1</td><td>40.1</td></tr><tr><td>100.9</td><td>24.2</td></tr><tr><td>98.8</td><td>28.4</td></tr><tr><td>148.1</td><td>35.2</td></tr><tr><td>108.1</td><td>32.6</td></tr><tr><td>126.2</td><td>34.5</td></tr><tr><td>104.3</td><td>32.9</td></tr><tr><td>111.2</td><td>31.6</td></tr><tr><td>104.3</td><td>32.0</td></tr><tr><td>76.0</td><td>7.7</td></tr><tr><td>81.5</td><td>13.9</td></tr><tr><td>73.7</td><td>10.8</td></tr><tr><td>79.5</td><td>5.6</td></tr><tr><td>83.4</td><td>13.6</td></tr><tr><td>70.4</td><td>4.0</td></tr><tr><td>86.7</td><td>10.2</td></tr><tr><td>77.9</td><td>6.6</td></tr><tr><td>82.0</td><td>8.0</td></tr><tr><td>79.6</td><td>6.3</td></tr><tr><td>77.6</td><td>3.9</td></tr><tr><td>100.0</td><td>22.6</td></tr><tr><td>99.8</td><td>20.4</td></tr><tr><td>104.2</td><td>28.0</td></tr><tr><td>105.3</td><td>31.5</td></tr><tr><td>98.3</td><td>24.6</td></tr><tr><td>104.8</td><td>26.1</td></tr><tr><td>94.7</td><td>29.8</td></tr><tr><td>102.4</td><td>30.7</td></tr><tr><td>99.7</td><td>25.8</td></tr><tr><td>105.5</td><td>32.3</td></tr><tr><td>100.3</td><td>30.0</td></tr><tr><td>83.9</td><td>21.5</td></tr><tr><td>86.6</td><td>13.8</td></tr><tr><td>78.4</td><td>6.3</td></tr><tr><td>84.6</td><td>12.9</td></tr><tr><td>91.5</td><td>24.3</td></tr><tr><td>82.8</td><td>8.8</td></tr><tr><td>82.9</td><td>8.5</td></tr><tr><td>76.0</td><td>13.5</td></tr><tr><td>83.3</td><td>11.8</td></tr><tr><td>81.8</td><td>18.5</td></tr><tr><td>78.8</td><td>8.8</td></tr><tr><td>95.0</td><td>22.2</td></tr><tr><td>95.4</td><td>21.5</td></tr><tr><td>98.6</td><td>18.8</td></tr><tr><td>95.8</td><td>31.4</td></tr><tr><td>89.0</td><td>26.8</td></tr><tr><td>97.8</td><td>18.4</td></tr><tr><td>94.9</td><td>27.0</td></tr><tr><td>99.8</td><td>27.0</td></tr><tr><td>89.7</td><td>26.6</td></tr><tr><td>88.1</td><td>14.9</td></tr><tr><td>90.9</td><td>23.1</td></tr><tr><td>86.0</td><td>8.3</td></tr><tr><td>86.5</td><td>14.1</td></tr><tr><td>95.6</td><td>20.5</td></tr><tr><td>93.2</td><td>18.2</td></tr><tr><td>83.1</td><td>8.5</td></tr><tr><td>97.5</td><td>24.9</td></tr><tr><td>88.8</td><td>9.0</td></tr><tr><td>99.2</td><td>17.4</td></tr><tr><td>91.6</td><td>9.6</td></tr><tr><td>86.7</td><td>11.3</td></tr><tr><td>88.2</td><td>17.8</td></tr><tr><td>94.0</td><td>22.2</td></tr><tr><td>95.0</td><td>21.2</td></tr><tr><td>92.0</td><td>20.4</td></tr><tr><td>89.2</td><td>20.1</td></tr><tr><td>95.5</td><td>22.3</td></tr><tr><td>98.6</td><td>25.4</td></tr><tr><td>87.3</td><td>18.0</td></tr><tr><td>102.8</td><td>19.3</td></tr><tr><td>101.6</td><td>18.3</td></tr><tr><td>88.7</td><td>17.3</td></tr><tr><td>92.3</td><td>21.4</td></tr><tr><td>90.6</td><td>19.7</td></tr><tr><td>105.0</td><td>28.0</td></tr><tr><td>95.0</td><td>22.1</td></tr><tr><td>89.6</td><td>21.3</td></tr><tr><td>92.4</td><td>26.7</td></tr><tr><td>86.6</td><td>16.7</td></tr><tr><td>90.0</td><td>20.1</td></tr><tr><td>90.0</td><td>13.9</td></tr><tr><td>92.4</td><td>25.8</td></tr><tr><td>87.5</td><td>18.1</td></tr><tr><td>99.2</td><td>27.9</td></tr><tr><td>98.1</td><td>25.3</td></tr><tr><td>83.3</td><td>14.7</td></tr><tr><td>86.1</td><td>16.0</td></tr><tr><td>84.1</td><td>13.8</td></tr><tr><td>89.9</td><td>17.5</td></tr><tr><td>92.1</td><td>27.2</td></tr><tr><td>78.0</td><td>17.4</td></tr><tr><td>93.5</td><td>20.8</td></tr><tr><td>87.0</td><td>14.9</td></tr><tr><td>90.1</td><td>18.1</td></tr><tr><td>90.3</td><td>22.7</td></tr><tr><td>99.8</td><td>23.6</td></tr><tr><td>89.4</td><td>26.1</td></tr><tr><td>87.2</td><td>24.4</td></tr><tr><td>101.1</td><td>27.1</td></tr><tr><td>86.1</td><td>21.8</td></tr><tr><td>98.6</td><td>29.4</td></tr><tr><td>88.5</td><td>22.4</td></tr><tr><td>106.6</td><td>20.4</td></tr><tr><td>93.1</td><td>24.9</td></tr><tr><td>93.0</td><td>18.3</td></tr><tr><td>91.0</td><td>23.3</td></tr><tr><td>77.1</td><td>9.4</td></tr><tr><td>85.3</td><td>10.3</td></tr><tr><td>81.9</td><td>14.2</td></tr><tr><td>99.1</td><td>19.2</td></tr><tr><td>100.5</td><td>29.6</td></tr><tr><td>76.5</td><td>5.3</td></tr><tr><td>106.8</td><td>25.2</td></tr><tr><td>77.6</td><td>9.4</td></tr><tr><td>102.9</td><td>19.6</td></tr><tr><td>72.8</td><td>10.1</td></tr><tr><td>88.2</td><td>16.5</td></tr><tr><td>100.1</td><td>21.0</td></tr><tr><td>83.5</td><td>17.3</td></tr><tr><td>105.0</td><td>31.2</td></tr><tr><td>90.8</td><td>10.0</td></tr><tr><td>76.6</td><td>12.5</td></tr><tr><td>92.4</td><td>22.5</td></tr><tr><td>81.2</td><td>9.4</td></tr><tr><td>95.6</td><td>14.6</td></tr><tr><td>92.1</td><td>13.0</td></tr><tr><td>83.4</td><td>15.1</td></tr><tr><td>106.0</td><td>27.3</td></tr><tr><td>95.1</td><td>19.2</td></tr><tr><td>90.4</td><td>21.8</td></tr><tr><td>100.4</td><td>20.3</td></tr><tr><td>115.9</td><td>34.3</td></tr><tr><td>90.8</td><td>16.5</td></tr><tr><td>81.9</td><td>3.0</td></tr><tr><td>75.0</td><td>0.7</td></tr><tr><td>90.3</td><td>20.5</td></tr><tr><td>90.3</td><td>16.9</td></tr><tr><td>108.8</td><td>25.3</td></tr><tr><td>79.4</td><td>9.9</td></tr><tr><td>83.2</td><td>13.1</td></tr><tr><td>110.3</td><td>29.9</td></tr><tr><td>92.7</td><td>22.5</td></tr><tr><td>104.5</td><td>16.9</td></tr><tr><td>104.6</td><td>26.6</td></tr><tr><td>69.4</td><td>0.0</td></tr><tr><td>83.6</td><td>11.5</td></tr><tr><td>86.8</td><td>12.1</td></tr><tr><td>90.4</td><td>17.5</td></tr><tr><td>83.7</td><td>8.6</td></tr><tr><td>109.3</td><td>23.6</td></tr><tr><td>98.9</td><td>20.4</td></tr><tr><td>98.0</td><td>20.5</td></tr><tr><td>101.2</td><td>24.4</td></tr><tr><td>80.6</td><td>11.4</td></tr><tr><td>113.7</td><td>38.1</td></tr><tr><td>94.1</td><td>15.9</td></tr><tr><td>105.7</td><td>24.7</td></tr><tr><td>85.6</td><td>22.8</td></tr><tr><td>96.6</td><td>25.5</td></tr><tr><td>86.0</td><td>22.0</td></tr><tr><td>89.7</td><td>17.7</td></tr><tr><td>78.0</td><td>6.6</td></tr><tr><td>89.7</td><td>23.6</td></tr><tr><td>89.2</td><td>12.2</td></tr><tr><td>85.7</td><td>22.1</td></tr><tr><td>103.1</td><td>28.7</td></tr><tr><td>89.1</td><td>6.0</td></tr><tr><td>113.9</td><td>34.8</td></tr><tr><td>96.3</td><td>16.6</td></tr><tr><td>93.9</td><td>32.9</td></tr><tr><td>101.3</td><td>32.8</td></tr><tr><td>83.9</td><td>9.6</td></tr><tr><td>84.4</td><td>10.8</td></tr><tr><td>79.4</td><td>7.1</td></tr><tr><td>104.0</td><td>27.2</td></tr><tr><td>89.7</td><td>19.5</td></tr><tr><td>97.6</td><td>18.7</td></tr><tr><td>87.6</td><td>19.5</td></tr><tr><td>122.1</td><td>47.5</td></tr><tr><td>81.1</td><td>13.6</td></tr><tr><td>81.5</td><td>7.5</td></tr><tr><td>100.0</td><td>24.5</td></tr><tr><td>88.7</td><td>15.0</td></tr><tr><td>91.8</td><td>12.4</td></tr><tr><td>110.4</td><td>26.0</td></tr><tr><td>87.6</td><td>11.5</td></tr><tr><td>82.8</td><td>5.2</td></tr><tr><td>95.3</td><td>10.9</td></tr><tr><td>78.2</td><td>12.5</td></tr><tr><td>91.1</td><td>14.8</td></tr><tr><td>96.7</td><td>25.2</td></tr><tr><td>89.4</td><td>14.9</td></tr><tr><td>93.0</td><td>17.0</td></tr><tr><td>86.4</td><td>10.6</td></tr><tr><td>96.7</td><td>16.1</td></tr><tr><td>88.1</td><td>15.4</td></tr><tr><td>94.9</td><td>26.7</td></tr><tr><td>93.3</td><td>25.8</td></tr><tr><td>95.6</td><td>18.6</td></tr><tr><td>98.2</td><td>24.8</td></tr><tr><td>113.8</td><td>27.3</td></tr><tr><td>82.8</td><td>12.4</td></tr><tr><td>100.5</td><td>29.9</td></tr><tr><td>79.7</td><td>17.0</td></tr><tr><td>118.0</td><td>35.0</td></tr><tr><td>109.0</td><td>30.4</td></tr><tr><td>113.4</td><td>32.6</td></tr><tr><td>106.1</td><td>29.0</td></tr><tr><td>84.3</td><td>15.2</td></tr><tr><td>107.6</td><td>30.2</td></tr><tr><td>83.6</td><td>11.0</td></tr><tr><td>105.0</td><td>33.6</td></tr><tr><td>111.5</td><td>29.3</td></tr><tr><td>101.3</td><td>26.0</td></tr><tr><td>108.5</td><td>31.9</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"744899ed-b93b-43be-82e9-184f488bad21\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT Hip, BodyFat FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Hip</th><th>BodyFat</th></tr></thead><tbody><tr><td>94.5</td><td>12.3</td></tr><tr><td>98.7</td><td>6.1</td></tr><tr><td>99.2</td><td>25.3</td></tr><tr><td>101.2</td><td>10.4</td></tr><tr><td>101.9</td><td>28.7</td></tr><tr><td>107.8</td><td>20.9</td></tr><tr><td>100.3</td><td>19.2</td></tr><tr><td>97.1</td><td>12.4</td></tr><tr><td>99.9</td><td>4.1</td></tr><tr><td>104.1</td><td>11.7</td></tr><tr><td>98.2</td><td>7.1</td></tr><tr><td>107.7</td><td>7.8</td></tr><tr><td>103.9</td><td>20.8</td></tr><tr><td>108.6</td><td>21.2</td></tr><tr><td>100.1</td><td>22.1</td></tr><tr><td>99.2</td><td>20.9</td></tr><tr><td>105.2</td><td>29.0</td></tr><tr><td>107.0</td><td>22.9</td></tr><tr><td>102.4</td><td>16.0</td></tr><tr><td>109.0</td><td>16.5</td></tr><tr><td>104.9</td><td>19.1</td></tr><tr><td>104.8</td><td>15.2</td></tr><tr><td>94.6</td><td>15.6</td></tr><tr><td>93.4</td><td>17.7</td></tr><tr><td>95.8</td><td>14.0</td></tr><tr><td>96.5</td><td>3.7</td></tr><tr><td>85.3</td><td>7.9</td></tr><tr><td>94.7</td><td>22.9</td></tr><tr><td>88.5</td><td>3.7</td></tr><tr><td>98.7</td><td>8.8</td></tr><tr><td>99.8</td><td>11.9</td></tr><tr><td>100.6</td><td>5.7</td></tr><tr><td>94.5</td><td>11.8</td></tr><tr><td>108.3</td><td>21.3</td></tr><tr><td>116.1</td><td>32.3</td></tr><tr><td>113.8</td><td>40.1</td></tr><tr><td>106.2</td><td>24.2</td></tr><tr><td>104.8</td><td>28.4</td></tr><tr><td>147.7</td><td>35.2</td></tr><tr><td>102.5</td><td>32.6</td></tr><tr><td>125.6</td><td>34.5</td></tr><tr><td>115.5</td><td>32.9</td></tr><tr><td>114.1</td><td>31.6</td></tr><tr><td>106.0</td><td>32.0</td></tr><tr><td>88.2</td><td>7.7</td></tr><tr><td>97.2</td><td>13.9</td></tr><tr><td>88.5</td><td>10.8</td></tr><tr><td>92.7</td><td>5.6</td></tr><tr><td>90.4</td><td>13.6</td></tr><tr><td>87.2</td><td>4.0</td></tr><tr><td>98.3</td><td>10.2</td></tr><tr><td>91.0</td><td>6.6</td></tr><tr><td>89.1</td><td>8.0</td></tr><tr><td>91.6</td><td>6.3</td></tr><tr><td>88.6</td><td>3.9</td></tr><tr><td>99.6</td><td>22.6</td></tr><tr><td>102.5</td><td>20.4</td></tr><tr><td>105.8</td><td>28.0</td></tr><tr><td>97.0</td><td>31.5</td></tr><tr><td>99.6</td><td>24.6</td></tr><tr><td>103.1</td><td>26.1</td></tr><tr><td>100.8</td><td>29.8</td></tr><tr><td>99.4</td><td>30.7</td></tr><tr><td>99.7</td><td>25.8</td></tr><tr><td>108.3</td><td>32.3</td></tr><tr><td>104.2</td><td>30.0</td></tr><tr><td>93.9</td><td>21.5</td></tr><tr><td>91.8</td><td>13.8</td></tr><tr><td>96.1</td><td>6.3</td></tr><tr><td>94.3</td><td>12.9</td></tr><tr><td>98.5</td><td>24.3</td></tr><tr><td>95.5</td><td>8.8</td></tr><tr><td>96.3</td><td>8.5</td></tr><tr><td>88.6</td><td>13.5</td></tr><tr><td>93.0</td><td>11.8</td></tr><tr><td>94.8</td><td>18.5</td></tr><tr><td>94.3</td><td>8.8</td></tr><tr><td>98.3</td><td>22.2</td></tr><tr><td>99.3</td><td>21.5</td></tr><tr><td>100.2</td><td>18.8</td></tr><tr><td>97.1</td><td>31.4</td></tr><tr><td>96.9</td><td>26.8</td></tr><tr><td>99.6</td><td>18.4</td></tr><tr><td>95.0</td><td>27.0</td></tr><tr><td>96.2</td><td>27.0</td></tr><tr><td>96.2</td><td>26.6</td></tr><tr><td>96.9</td><td>14.9</td></tr><tr><td>93.8</td><td>23.1</td></tr><tr><td>99.3</td><td>8.3</td></tr><tr><td>98.3</td><td>14.1</td></tr><tr><td>102.2</td><td>20.5</td></tr><tr><td>100.6</td><td>18.2</td></tr><tr><td>95.4</td><td>8.5</td></tr><tr><td>100.6</td><td>24.9</td></tr><tr><td>101.4</td><td>9.0</td></tr><tr><td>107.5</td><td>17.4</td></tr><tr><td>102.4</td><td>9.6</td></tr><tr><td>96.2</td><td>11.3</td></tr><tr><td>92.8</td><td>17.8</td></tr><tr><td>103.7</td><td>22.2</td></tr><tr><td>101.7</td><td>21.2</td></tr><tr><td>98.3</td><td>20.4</td></tr><tr><td>98.3</td><td>20.1</td></tr><tr><td>101.6</td><td>22.3</td></tr><tr><td>99.5</td><td>25.4</td></tr><tr><td>96.6</td><td>18.0</td></tr><tr><td>103.6</td><td>19.3</td></tr><tr><td>100.7</td><td>18.3</td></tr><tr><td>102.1</td><td>17.3</td></tr><tr><td>100.6</td><td>21.4</td></tr><tr><td>99.3</td><td>19.7</td></tr><tr><td>103.0</td><td>28.0</td></tr><tr><td>98.6</td><td>22.1</td></tr><tr><td>99.8</td><td>21.3</td></tr><tr><td>97.5</td><td>26.7</td></tr><tr><td>92.6</td><td>16.7</td></tr><tr><td>99.7</td><td>20.1</td></tr><tr><td>96.4</td><td>13.9</td></tr><tr><td>104.3</td><td>25.8</td></tr><tr><td>101.0</td><td>18.1</td></tr><tr><td>104.1</td><td>27.9</td></tr><tr><td>101.4</td><td>25.3</td></tr><tr><td>97.5</td><td>14.7</td></tr><tr><td>95.2</td><td>16.0</td></tr><tr><td>94.0</td><td>13.8</td></tr><tr><td>100.0</td><td>17.5</td></tr><tr><td>98.5</td><td>27.2</td></tr><tr><td>93.2</td><td>17.4</td></tr><tr><td>99.5</td><td>20.8</td></tr><tr><td>97.8</td><td>14.9</td></tr><tr><td>95.8</td><td>18.1</td></tr><tr><td>99.1</td><td>22.7</td></tr><tr><td>103.2</td><td>23.6</td></tr><tr><td>92.3</td><td>26.1</td></tr><tr><td>98.4</td><td>24.4</td></tr><tr><td>102.1</td><td>27.1</td></tr><tr><td>95.6</td><td>21.8</td></tr><tr><td>100.6</td><td>29.4</td></tr><tr><td>98.3</td><td>22.4</td></tr><tr><td>107.7</td><td>20.4</td></tr><tr><td>101.6</td><td>24.9</td></tr><tr><td>99.3</td><td>18.3</td></tr><tr><td>98.9</td><td>23.3</td></tr><tr><td>93.9</td><td>9.4</td></tr><tr><td>102.5</td><td>10.3</td></tr><tr><td>95.3</td><td>14.2</td></tr><tr><td>110.1</td><td>19.2</td></tr><tr><td>106.2</td><td>29.6</td></tr><tr><td>92.1</td><td>5.3</td></tr><tr><td>113.9</td><td>25.2</td></tr><tr><td>93.5</td><td>9.4</td></tr><tr><td>114.4</td><td>19.6</td></tr><tr><td>91.1</td><td>10.1</td></tr><tr><td>95.2</td><td>16.5</td></tr><tr><td>105.0</td><td>21.0</td></tr><tr><td>98.3</td><td>17.3</td></tr><tr><td>106.4</td><td>31.2</td></tr><tr><td>102.5</td><td>10.0</td></tr><tr><td>89.8</td><td>12.5</td></tr><tr><td>99.3</td><td>22.5</td></tr><tr><td>91.5</td><td>9.4</td></tr><tr><td>105.1</td><td>14.6</td></tr><tr><td>103.5</td><td>13.0</td></tr><tr><td>89.6</td><td>15.1</td></tr><tr><td>108.8</td><td>27.3</td></tr><tr><td>104.5</td><td>19.2</td></tr><tr><td>95.6</td><td>21.8</td></tr><tr><td>106.8</td><td>20.3</td></tr><tr><td>111.9</td><td>34.3</td></tr><tr><td>98.1</td><td>16.5</td></tr><tr><td>92.8</td><td>3.0</td></tr><tr><td>89.2</td><td>0.7</td></tr><tr><td>98.7</td><td>20.5</td></tr><tr><td>99.9</td><td>16.9</td></tr><tr><td>114.4</td><td>25.3</td></tr><tr><td>89.2</td><td>9.9</td></tr><tr><td>96.4</td><td>13.1</td></tr><tr><td>113.9</td><td>29.9</td></tr><tr><td>101.9</td><td>22.5</td></tr><tr><td>109.9</td><td>16.9</td></tr><tr><td>109.8</td><td>26.6</td></tr><tr><td>85.0</td><td>0.0</td></tr><tr><td>91.6</td><td>11.5</td></tr><tr><td>96.1</td><td>12.1</td></tr><tr><td>95.5</td><td>17.5</td></tr><tr><td>98.1</td><td>8.6</td></tr><tr><td>108.8</td><td>23.6</td></tr><tr><td>104.1</td><td>20.4</td></tr><tr><td>101.8</td><td>20.5</td></tr><tr><td>103.1</td><td>24.4</td></tr><tr><td>92.3</td><td>11.4</td></tr><tr><td>112.4</td><td>38.1</td></tr><tr><td>102.7</td><td>15.9</td></tr><tr><td>111.8</td><td>24.7</td></tr><tr><td>96.5</td><td>22.8</td></tr><tr><td>100.6</td><td>25.5</td></tr><tr><td>96.2</td><td>22.0</td></tr><tr><td>101.0</td><td>17.7</td></tr><tr><td>99.0</td><td>6.6</td></tr><tr><td>94.2</td><td>23.6</td></tr><tr><td>99.2</td><td>12.2</td></tr><tr><td>96.9</td><td>22.1</td></tr><tr><td>105.5</td><td>28.7</td></tr><tr><td>102.6</td><td>6.0</td></tr><tr><td>107.1</td><td>34.8</td></tr><tr><td>102.0</td><td>16.6</td></tr><tr><td>100.1</td><td>32.9</td></tr><tr><td>101.7</td><td>32.8</td></tr><tr><td>91.8</td><td>9.6</td></tr><tr><td>94.0</td><td>10.8</td></tr><tr><td>89.0</td><td>7.1</td></tr><tr><td>109.0</td><td>27.2</td></tr><tr><td>99.1</td><td>19.5</td></tr><tr><td>104.2</td><td>18.7</td></tr><tr><td>96.1</td><td>19.5</td></tr><tr><td>112.8</td><td>47.5</td></tr><tr><td>96.3</td><td>13.6</td></tr><tr><td>94.4</td><td>7.5</td></tr><tr><td>105.0</td><td>24.5</td></tr><tr><td>94.5</td><td>15.0</td></tr><tr><td>96.2</td><td>12.4</td></tr><tr><td>105.5</td><td>26.0</td></tr><tr><td>95.6</td><td>11.5</td></tr><tr><td>91.9</td><td>5.2</td></tr><tr><td>98.2</td><td>10.9</td></tr><tr><td>87.5</td><td>12.5</td></tr><tr><td>97.1</td><td>14.8</td></tr><tr><td>106.6</td><td>25.2</td></tr><tr><td>98.4</td><td>14.9</td></tr><tr><td>97.0</td><td>17.0</td></tr><tr><td>90.1</td><td>10.6</td></tr><tr><td>100.7</td><td>16.1</td></tr><tr><td>97.8</td><td>15.4</td></tr><tr><td>100.2</td><td>26.7</td></tr><tr><td>94.0</td><td>25.8</td></tr><tr><td>93.7</td><td>18.6</td></tr><tr><td>101.1</td><td>24.8</td></tr><tr><td>111.8</td><td>27.3</td></tr><tr><td>94.5</td><td>12.4</td></tr><tr><td>100.5</td><td>29.9</td></tr><tr><td>87.6</td><td>17.0</td></tr><tr><td>114.3</td><td>35.0</td></tr><tr><td>109.1</td><td>30.4</td></tr><tr><td>109.8</td><td>32.6</td></tr><tr><td>101.6</td><td>29.0</td></tr><tr><td>94.4</td><td>15.2</td></tr><tr><td>110.0</td><td>30.2</td></tr><tr><td>88.8</td><td>11.0</td></tr><tr><td>104.5</td><td>33.6</td></tr><tr><td>101.7</td><td>29.3</td></tr><tr><td>97.8</td><td>26.0</td></tr><tr><td>107.1</td><td>31.9</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"37e20f6c-91c9-409e-bd81-28b3c5de89cd\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT Thigh, BodyFat FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Thigh</th><th>BodyFat</th></tr></thead><tbody><tr><td>59.0</td><td>12.3</td></tr><tr><td>58.7</td><td>6.1</td></tr><tr><td>59.6</td><td>25.3</td></tr><tr><td>60.1</td><td>10.4</td></tr><tr><td>63.2</td><td>28.7</td></tr><tr><td>66.0</td><td>20.9</td></tr><tr><td>58.4</td><td>19.2</td></tr><tr><td>60.0</td><td>12.4</td></tr><tr><td>62.9</td><td>4.1</td></tr><tr><td>63.1</td><td>11.7</td></tr><tr><td>59.7</td><td>7.1</td></tr><tr><td>66.2</td><td>7.8</td></tr><tr><td>63.4</td><td>20.8</td></tr><tr><td>66.0</td><td>21.2</td></tr><tr><td>69.0</td><td>22.1</td></tr><tr><td>63.1</td><td>20.9</td></tr><tr><td>64.8</td><td>29.0</td></tr><tr><td>66.9</td><td>22.9</td></tr><tr><td>64.2</td><td>16.0</td></tr><tr><td>65.8</td><td>16.5</td></tr><tr><td>63.5</td><td>19.1</td></tr><tr><td>63.4</td><td>15.2</td></tr><tr><td>57.4</td><td>15.6</td></tr><tr><td>54.9</td><td>17.7</td></tr><tr><td>58.4</td><td>14.0</td></tr><tr><td>55.0</td><td>3.7</td></tr><tr><td>51.7</td><td>7.9</td></tr><tr><td>57.5</td><td>22.9</td></tr><tr><td>50.1</td><td>3.7</td></tr><tr><td>58.9</td><td>8.8</td></tr><tr><td>57.5</td><td>11.9</td></tr><tr><td>58.5</td><td>5.7</td></tr><tr><td>57.3</td><td>11.8</td></tr><tr><td>67.1</td><td>21.3</td></tr><tr><td>71.2</td><td>32.3</td></tr><tr><td>61.9</td><td>40.1</td></tr><tr><td>63.5</td><td>24.2</td></tr><tr><td>66.0</td><td>28.4</td></tr><tr><td>87.3</td><td>35.2</td></tr><tr><td>61.3</td><td>32.6</td></tr><tr><td>72.5</td><td>34.5</td></tr><tr><td>70.6</td><td>32.9</td></tr><tr><td>67.7</td><td>31.6</td></tr><tr><td>65.0</td><td>32.0</td></tr><tr><td>50.0</td><td>7.7</td></tr><tr><td>58.4</td><td>13.9</td></tr><tr><td>53.3</td><td>10.8</td></tr><tr><td>52.7</td><td>5.6</td></tr><tr><td>52.0</td><td>13.6</td></tr><tr><td>50.6</td><td>4.0</td></tr><tr><td>52.6</td><td>10.2</td></tr><tr><td>51.4</td><td>6.6</td></tr><tr><td>49.3</td><td>8.0</td></tr><tr><td>52.6</td><td>6.3</td></tr><tr><td>51.9</td><td>3.9</td></tr><tr><td>57.2</td><td>22.6</td></tr><tr><td>62.1</td><td>20.4</td></tr><tr><td>61.8</td><td>28.0</td></tr><tr><td>59.1</td><td>31.5</td></tr><tr><td>60.6</td><td>24.6</td></tr><tr><td>61.6</td><td>26.1</td></tr><tr><td>60.9</td><td>29.8</td></tr><tr><td>61.0</td><td>30.7</td></tr><tr><td>60.8</td><td>25.8</td></tr><tr><td>65.0</td><td>32.3</td></tr><tr><td>64.8</td><td>30.0</td></tr><tr><td>55.0</td><td>21.5</td></tr><tr><td>54.3</td><td>13.8</td></tr><tr><td>56.0</td><td>6.3</td></tr><tr><td>51.2</td><td>12.9</td></tr><tr><td>56.6</td><td>24.3</td></tr><tr><td>58.9</td><td>8.8</td></tr><tr><td>52.9</td><td>8.5</td></tr><tr><td>50.9</td><td>13.5</td></tr><tr><td>55.5</td><td>11.8</td></tr><tr><td>54.5</td><td>18.5</td></tr><tr><td>56.7</td><td>8.8</td></tr><tr><td>55.0</td><td>22.2</td></tr><tr><td>53.5</td><td>21.5</td></tr><tr><td>56.5</td><td>18.8</td></tr><tr><td>54.8</td><td>31.4</td></tr><tr><td>54.8</td><td>26.8</td></tr><tr><td>58.9</td><td>18.4</td></tr><tr><td>56.0</td><td>27.0</td></tr><tr><td>56.3</td><td>27.0</td></tr><tr><td>54.7</td><td>26.6</td></tr><tr><td>57.2</td><td>14.9</td></tr><tr><td>57.8</td><td>23.1</td></tr><tr><td>61.0</td><td>8.3</td></tr><tr><td>60.4</td><td>14.1</td></tr><tr><td>58.3</td><td>20.5</td></tr><tr><td>58.9</td><td>18.2</td></tr><tr><td>56.9</td><td>8.5</td></tr><tr><td>58.9</td><td>24.9</td></tr><tr><td>57.4</td><td>9.0</td></tr><tr><td>61.7</td><td>17.4</td></tr><tr><td>60.6</td><td>9.6</td></tr><tr><td>62.1</td><td>11.3</td></tr><tr><td>54.7</td><td>17.8</td></tr><tr><td>62.7</td><td>22.2</td></tr><tr><td>59.0</td><td>21.2</td></tr><tr><td>59.3</td><td>20.4</td></tr><tr><td>60.0</td><td>20.1</td></tr><tr><td>59.1</td><td>22.3</td></tr><tr><td>59.5</td><td>25.4</td></tr><tr><td>54.7</td><td>18.0</td></tr><tr><td>61.2</td><td>19.3</td></tr><tr><td>55.8</td><td>18.3</td></tr><tr><td>57.5</td><td>17.3</td></tr><tr><td>57.5</td><td>21.4</td></tr><tr><td>61.9</td><td>19.7</td></tr><tr><td>63.7</td><td>28.0</td></tr><tr><td>62.3</td><td>22.1</td></tr><tr><td>61.5</td><td>21.3</td></tr><tr><td>59.3</td><td>26.7</td></tr><tr><td>55.9</td><td>16.7</td></tr><tr><td>58.8</td><td>20.1</td></tr><tr><td>56.8</td><td>13.9</td></tr><tr><td>64.6</td><td>25.8</td></tr><tr><td>58.5</td><td>18.1</td></tr><tr><td>58.5</td><td>27.9</td></tr><tr><td>57.1</td><td>25.3</td></tr><tr><td>60.5</td><td>14.7</td></tr><tr><td>58.1</td><td>16.0</td></tr><tr><td>58.5</td><td>13.8</td></tr><tr><td>60.7</td><td>17.5</td></tr><tr><td>60.7</td><td>27.2</td></tr><tr><td>53.5</td><td>17.4</td></tr><tr><td>61.7</td><td>20.8</td></tr><tr><td>57.4</td><td>14.9</td></tr><tr><td>57.0</td><td>18.1</td></tr><tr><td>60.3</td><td>22.7</td></tr><tr><td>61.2</td><td>23.6</td></tr><tr><td>56.1</td><td>26.1</td></tr><tr><td>56.0</td><td>24.4</td></tr><tr><td>58.9</td><td>27.1</td></tr><tr><td>58.8</td><td>21.8</td></tr><tr><td>63.6</td><td>29.4</td></tr><tr><td>58.1</td><td>22.4</td></tr><tr><td>66.5</td><td>20.4</td></tr><tr><td>59.1</td><td>24.9</td></tr><tr><td>60.4</td><td>18.3</td></tr><tr><td>57.1</td><td>23.3</td></tr><tr><td>56.1</td><td>9.4</td></tr><tr><td>59.1</td><td>10.3</td></tr><tr><td>56.4</td><td>14.2</td></tr><tr><td>71.2</td><td>19.2</td></tr><tr><td>68.4</td><td>29.6</td></tr><tr><td>51.9</td><td>5.3</td></tr><tr><td>67.6</td><td>25.2</td></tr><tr><td>56.9</td><td>9.4</td></tr><tr><td>72.9</td><td>19.6</td></tr><tr><td>53.6</td><td>10.1</td></tr><tr><td>56.8</td><td>16.5</td></tr><tr><td>62.1</td><td>21.0</td></tr><tr><td>57.3</td><td>17.3</td></tr><tr><td>68.6</td><td>31.2</td></tr><tr><td>60.8</td><td>10.0</td></tr><tr><td>50.1</td><td>12.5</td></tr><tr><td>59.4</td><td>22.5</td></tr><tr><td>52.5</td><td>9.4</td></tr><tr><td>61.4</td><td>14.6</td></tr><tr><td>64.0</td><td>13.0</td></tr><tr><td>52.4</td><td>15.1</td></tr><tr><td>63.8</td><td>27.3</td></tr><tr><td>64.8</td><td>19.2</td></tr><tr><td>55.5</td><td>21.8</td></tr><tr><td>63.3</td><td>20.3</td></tr><tr><td>74.4</td><td>34.3</td></tr><tr><td>60.1</td><td>16.5</td></tr><tr><td>54.7</td><td>3.0</td></tr><tr><td>50.0</td><td>0.7</td></tr><tr><td>57.8</td><td>20.5</td></tr><tr><td>59.2</td><td>16.9</td></tr><tr><td>69.2</td><td>25.3</td></tr><tr><td>50.3</td><td>9.9</td></tr><tr><td>60.0</td><td>13.1</td></tr><tr><td>69.8</td><td>29.9</td></tr><tr><td>64.7</td><td>22.5</td></tr><tr><td>69.5</td><td>16.9</td></tr><tr><td>68.1</td><td>26.6</td></tr><tr><td>47.2</td><td>0.0</td></tr><tr><td>54.1</td><td>11.5</td></tr><tr><td>58.0</td><td>12.1</td></tr><tr><td>55.4</td><td>17.5</td></tr><tr><td>57.3</td><td>8.6</td></tr><tr><td>67.7</td><td>23.6</td></tr><tr><td>63.5</td><td>20.4</td></tr><tr><td>62.8</td><td>20.5</td></tr><tr><td>61.5</td><td>24.4</td></tr><tr><td>54.3</td><td>11.4</td></tr><tr><td>68.5</td><td>38.1</td></tr><tr><td>60.6</td><td>15.9</td></tr><tr><td>65.3</td><td>24.7</td></tr><tr><td>60.2</td><td>22.8</td></tr><tr><td>61.1</td><td>25.5</td></tr><tr><td>57.7</td><td>22.0</td></tr><tr><td>62.3</td><td>17.7</td></tr><tr><td>57.5</td><td>6.6</td></tr><tr><td>58.5</td><td>23.6</td></tr><tr><td>60.2</td><td>12.2</td></tr><tr><td>55.5</td><td>22.1</td></tr><tr><td>68.8</td><td>28.7</td></tr><tr><td>60.6</td><td>6.0</td></tr><tr><td>63.5</td><td>34.8</td></tr><tr><td>63.3</td><td>16.6</td></tr><tr><td>58.9</td><td>32.9</td></tr><tr><td>60.7</td><td>32.8</td></tr><tr><td>53.0</td><td>9.6</td></tr><tr><td>56.0</td><td>10.8</td></tr><tr><td>51.1</td><td>7.1</td></tr><tr><td>63.7</td><td>27.2</td></tr><tr><td>56.3</td><td>19.5</td></tr><tr><td>60.0</td><td>18.7</td></tr><tr><td>57.1</td><td>19.5</td></tr><tr><td>62.5</td><td>47.5</td></tr><tr><td>53.8</td><td>13.6</td></tr><tr><td>54.7</td><td>7.5</td></tr><tr><td>63.9</td><td>24.5</td></tr><tr><td>53.7</td><td>15.0</td></tr><tr><td>57.7</td><td>12.4</td></tr><tr><td>64.2</td><td>26.0</td></tr><tr><td>59.7</td><td>11.5</td></tr><tr><td>54.4</td><td>5.2</td></tr><tr><td>57.4</td><td>10.9</td></tr><tr><td>50.8</td><td>12.5</td></tr><tr><td>56.6</td><td>14.8</td></tr><tr><td>64.0</td><td>25.2</td></tr><tr><td>58.4</td><td>14.9</td></tr><tr><td>55.4</td><td>17.0</td></tr><tr><td>53.0</td><td>10.6</td></tr><tr><td>59.3</td><td>16.1</td></tr><tr><td>57.1</td><td>15.4</td></tr><tr><td>56.8</td><td>26.7</td></tr><tr><td>54.3</td><td>25.8</td></tr><tr><td>54.4</td><td>18.6</td></tr><tr><td>59.3</td><td>24.8</td></tr><tr><td>63.4</td><td>27.3</td></tr><tr><td>61.2</td><td>12.4</td></tr><tr><td>59.2</td><td>29.9</td></tr><tr><td>50.7</td><td>17.0</td></tr><tr><td>61.3</td><td>35.0</td></tr><tr><td>63.7</td><td>30.4</td></tr><tr><td>65.6</td><td>32.6</td></tr><tr><td>58.2</td><td>29.0</td></tr><tr><td>54.3</td><td>15.2</td></tr><tr><td>63.3</td><td>30.2</td></tr><tr><td>49.6</td><td>11.0</td></tr><tr><td>59.6</td><td>33.6</td></tr><tr><td>60.3</td><td>29.3</td></tr><tr><td>56.0</td><td>26.0</td></tr><tr><td>59.3</td><td>31.9</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"8c7e8283-758f-40be-83f1-2ca63b2dcf01\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT Knee, BodyFat FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Knee</th><th>BodyFat</th></tr></thead><tbody><tr><td>37.3</td><td>12.3</td></tr><tr><td>37.3</td><td>6.1</td></tr><tr><td>38.9</td><td>25.3</td></tr><tr><td>37.3</td><td>10.4</td></tr><tr><td>42.2</td><td>28.7</td></tr><tr><td>42.0</td><td>20.9</td></tr><tr><td>38.3</td><td>19.2</td></tr><tr><td>39.4</td><td>12.4</td></tr><tr><td>38.3</td><td>4.1</td></tr><tr><td>41.7</td><td>11.7</td></tr><tr><td>39.7</td><td>7.1</td></tr><tr><td>39.2</td><td>7.8</td></tr><tr><td>38.3</td><td>20.8</td></tr><tr><td>41.5</td><td>21.2</td></tr><tr><td>39.0</td><td>22.1</td></tr><tr><td>38.7</td><td>20.9</td></tr><tr><td>40.8</td><td>29.0</td></tr><tr><td>40.0</td><td>22.9</td></tr><tr><td>38.7</td><td>16.0</td></tr><tr><td>40.6</td><td>16.5</td></tr><tr><td>38.0</td><td>19.1</td></tr><tr><td>40.6</td><td>15.2</td></tr><tr><td>35.3</td><td>15.6</td></tr><tr><td>36.2</td><td>17.7</td></tr><tr><td>35.5</td><td>14.0</td></tr><tr><td>36.7</td><td>3.7</td></tr><tr><td>34.7</td><td>7.9</td></tr><tr><td>36.0</td><td>22.9</td></tr><tr><td>34.5</td><td>3.7</td></tr><tr><td>35.3</td><td>8.8</td></tr><tr><td>38.7</td><td>11.9</td></tr><tr><td>38.8</td><td>5.7</td></tr><tr><td>36.2</td><td>11.8</td></tr><tr><td>44.2</td><td>21.3</td></tr><tr><td>43.3</td><td>32.3</td></tr><tr><td>38.3</td><td>40.1</td></tr><tr><td>39.9</td><td>24.2</td></tr><tr><td>41.5</td><td>28.4</td></tr><tr><td>49.1</td><td>35.2</td></tr><tr><td>41.1</td><td>32.6</td></tr><tr><td>39.6</td><td>34.5</td></tr><tr><td>42.5</td><td>32.9</td></tr><tr><td>40.9</td><td>31.6</td></tr><tr><td>40.2</td><td>32.0</td></tr><tr><td>34.7</td><td>7.7</td></tr><tr><td>38.2</td><td>13.9</td></tr><tr><td>34.5</td><td>10.8</td></tr><tr><td>37.5</td><td>5.6</td></tr><tr><td>35.8</td><td>13.6</td></tr><tr><td>34.4</td><td>4.0</td></tr><tr><td>37.2</td><td>10.2</td></tr><tr><td>34.9</td><td>6.6</td></tr><tr><td>33.7</td><td>8.0</td></tr><tr><td>37.6</td><td>6.3</td></tr><tr><td>34.9</td><td>3.9</td></tr><tr><td>38.0</td><td>22.6</td></tr><tr><td>39.6</td><td>20.4</td></tr><tr><td>39.8</td><td>28.0</td></tr><tr><td>38.0</td><td>31.5</td></tr><tr><td>37.7</td><td>24.6</td></tr><tr><td>40.9</td><td>26.1</td></tr><tr><td>38.0</td><td>29.8</td></tr><tr><td>39.4</td><td>30.7</td></tr><tr><td>40.1</td><td>25.8</td></tr><tr><td>41.2</td><td>32.3</td></tr><tr><td>40.2</td><td>30.0</td></tr><tr><td>36.1</td><td>21.5</td></tr><tr><td>35.4</td><td>13.8</td></tr><tr><td>37.4</td><td>6.3</td></tr><tr><td>37.4</td><td>12.9</td></tr><tr><td>38.6</td><td>24.3</td></tr><tr><td>37.6</td><td>8.8</td></tr><tr><td>37.5</td><td>8.5</td></tr><tr><td>35.4</td><td>13.5</td></tr><tr><td>35.2</td><td>11.8</td></tr><tr><td>37.0</td><td>18.5</td></tr><tr><td>39.7</td><td>8.8</td></tr><tr><td>38.3</td><td>22.2</td></tr><tr><td>37.5</td><td>21.5</td></tr><tr><td>39.3</td><td>18.8</td></tr><tr><td>38.2</td><td>31.4</td></tr><tr><td>38.0</td><td>26.8</td></tr><tr><td>39.0</td><td>18.4</td></tr><tr><td>36.5</td><td>27.0</td></tr><tr><td>36.6</td><td>27.0</td></tr><tr><td>37.8</td><td>26.6</td></tr><tr><td>37.7</td><td>14.9</td></tr><tr><td>39.5</td><td>23.1</td></tr><tr><td>38.4</td><td>8.3</td></tr><tr><td>39.9</td><td>14.1</td></tr><tr><td>38.2</td><td>20.5</td></tr><tr><td>39.7</td><td>18.2</td></tr><tr><td>38.3</td><td>8.5</td></tr><tr><td>40.5</td><td>24.9</td></tr><tr><td>39.6</td><td>9.0</td></tr><tr><td>42.3</td><td>17.4</td></tr><tr><td>39.4</td><td>9.6</td></tr><tr><td>39.3</td><td>11.3</td></tr><tr><td>37.3</td><td>17.8</td></tr><tr><td>39.0</td><td>22.2</td></tr><tr><td>39.4</td><td>21.2</td></tr><tr><td>38.4</td><td>20.4</td></tr><tr><td>38.4</td><td>20.1</td></tr><tr><td>39.8</td><td>22.3</td></tr><tr><td>36.1</td><td>25.4</td></tr><tr><td>39.0</td><td>18.0</td></tr><tr><td>39.3</td><td>19.3</td></tr><tr><td>38.7</td><td>18.3</td></tr><tr><td>40.0</td><td>17.3</td></tr><tr><td>36.8</td><td>21.4</td></tr><tr><td>38.0</td><td>19.7</td></tr><tr><td>40.0</td><td>28.0</td></tr><tr><td>38.1</td><td>22.1</td></tr><tr><td>37.8</td><td>21.3</td></tr><tr><td>38.1</td><td>26.7</td></tr><tr><td>36.3</td><td>16.7</td></tr><tr><td>38.4</td><td>20.1</td></tr><tr><td>38.8</td><td>13.9</td></tr><tr><td>41.1</td><td>25.8</td></tr><tr><td>39.2</td><td>18.1</td></tr><tr><td>39.3</td><td>27.9</td></tr><tr><td>40.5</td><td>25.3</td></tr><tr><td>38.7</td><td>14.7</td></tr><tr><td>36.5</td><td>16.0</td></tr><tr><td>36.6</td><td>13.8</td></tr><tr><td>36.0</td><td>17.5</td></tr><tr><td>36.8</td><td>27.2</td></tr><tr><td>35.8</td><td>17.4</td></tr><tr><td>39.0</td><td>20.8</td></tr><tr><td>36.9</td><td>14.9</td></tr><tr><td>38.7</td><td>18.1</td></tr><tr><td>38.5</td><td>22.7</td></tr><tr><td>38.1</td><td>23.6</td></tr><tr><td>35.6</td><td>26.1</td></tr><tr><td>36.9</td><td>24.4</td></tr><tr><td>37.9</td><td>27.1</td></tr><tr><td>36.1</td><td>21.8</td></tr><tr><td>39.2</td><td>29.4</td></tr><tr><td>38.4</td><td>22.4</td></tr><tr><td>42.5</td><td>20.4</td></tr><tr><td>39.6</td><td>24.9</td></tr><tr><td>38.2</td><td>18.3</td></tr><tr><td>36.7</td><td>23.3</td></tr><tr><td>36.1</td><td>9.4</td></tr><tr><td>37.6</td><td>10.3</td></tr><tr><td>36.5</td><td>14.2</td></tr><tr><td>43.5</td><td>19.2</td></tr><tr><td>40.8</td><td>29.6</td></tr><tr><td>35.7</td><td>5.3</td></tr><tr><td>42.7</td><td>25.2</td></tr><tr><td>35.9</td><td>9.4</td></tr><tr><td>43.5</td><td>19.6</td></tr><tr><td>36.8</td><td>10.1</td></tr><tr><td>37.4</td><td>16.5</td></tr><tr><td>40.0</td><td>21.0</td></tr><tr><td>37.8</td><td>17.3</td></tr><tr><td>40.0</td><td>31.2</td></tr><tr><td>38.5</td><td>10.0</td></tr><tr><td>34.8</td><td>12.5</td></tr><tr><td>39.0</td><td>22.5</td></tr><tr><td>36.6</td><td>9.4</td></tr><tr><td>40.6</td><td>14.6</td></tr><tr><td>37.3</td><td>13.0</td></tr><tr><td>35.6</td><td>15.1</td></tr><tr><td>42.0</td><td>27.3</td></tr><tr><td>41.3</td><td>19.2</td></tr><tr><td>34.2</td><td>21.8</td></tr><tr><td>41.7</td><td>20.3</td></tr><tr><td>40.6</td><td>34.3</td></tr><tr><td>39.1</td><td>16.5</td></tr><tr><td>36.2</td><td>3.0</td></tr><tr><td>34.8</td><td>0.7</td></tr><tr><td>37.3</td><td>20.5</td></tr><tr><td>37.7</td><td>16.9</td></tr><tr><td>42.4</td><td>25.3</td></tr><tr><td>34.8</td><td>9.9</td></tr><tr><td>38.1</td><td>13.1</td></tr><tr><td>42.6</td><td>29.9</td></tr><tr><td>39.5</td><td>22.5</td></tr><tr><td>43.1</td><td>16.9</td></tr><tr><td>42.8</td><td>26.6</td></tr><tr><td>33.5</td><td>0.0</td></tr><tr><td>36.2</td><td>11.5</td></tr><tr><td>39.4</td><td>12.1</td></tr><tr><td>38.9</td><td>17.5</td></tr><tr><td>39.7</td><td>8.6</td></tr><tr><td>41.3</td><td>23.6</td></tr><tr><td>39.8</td><td>20.4</td></tr><tr><td>41.3</td><td>20.5</td></tr><tr><td>40.4</td><td>24.4</td></tr><tr><td>36.3</td><td>11.4</td></tr><tr><td>45.0</td><td>38.1</td></tr><tr><td>38.6</td><td>15.9</td></tr><tr><td>43.3</td><td>24.7</td></tr><tr><td>38.9</td><td>22.8</td></tr><tr><td>38.4</td><td>25.5</td></tr><tr><td>38.6</td><td>22.0</td></tr><tr><td>38.0</td><td>17.7</td></tr><tr><td>40.0</td><td>6.6</td></tr><tr><td>39.0</td><td>23.6</td></tr><tr><td>39.2</td><td>12.2</td></tr><tr><td>35.7</td><td>22.1</td></tr><tr><td>38.3</td><td>28.7</td></tr><tr><td>39.0</td><td>6.0</td></tr><tr><td>40.3</td><td>34.8</td></tr><tr><td>39.8</td><td>16.6</td></tr><tr><td>37.6</td><td>32.9</td></tr><tr><td>39.4</td><td>32.8</td></tr><tr><td>36.2</td><td>9.6</td></tr><tr><td>38.2</td><td>10.8</td></tr><tr><td>35.0</td><td>7.1</td></tr><tr><td>40.3</td><td>27.2</td></tr><tr><td>38.8</td><td>19.5</td></tr><tr><td>40.9</td><td>18.7</td></tr><tr><td>38.1</td><td>19.5</td></tr><tr><td>36.9</td><td>47.5</td></tr><tr><td>36.5</td><td>13.6</td></tr><tr><td>39.0</td><td>7.5</td></tr><tr><td>39.2</td><td>24.5</td></tr><tr><td>36.2</td><td>15.0</td></tr><tr><td>38.1</td><td>12.4</td></tr><tr><td>42.7</td><td>26.0</td></tr><tr><td>40.2</td><td>11.5</td></tr><tr><td>35.2</td><td>5.2</td></tr><tr><td>37.1</td><td>10.9</td></tr><tr><td>33.0</td><td>12.5</td></tr><tr><td>38.5</td><td>14.8</td></tr><tr><td>42.6</td><td>25.2</td></tr><tr><td>37.4</td><td>14.9</td></tr><tr><td>38.8</td><td>17.0</td></tr><tr><td>35.0</td><td>10.6</td></tr><tr><td>38.6</td><td>16.1</td></tr><tr><td>38.9</td><td>15.4</td></tr><tr><td>35.9</td><td>26.7</td></tr><tr><td>35.7</td><td>25.8</td></tr><tr><td>37.1</td><td>18.6</td></tr><tr><td>40.3</td><td>24.8</td></tr><tr><td>41.1</td><td>27.3</td></tr><tr><td>39.1</td><td>12.4</td></tr><tr><td>38.1</td><td>29.9</td></tr><tr><td>33.4</td><td>17.0</td></tr><tr><td>42.1</td><td>35.0</td></tr><tr><td>42.4</td><td>30.4</td></tr><tr><td>46.0</td><td>32.6</td></tr><tr><td>38.8</td><td>29.0</td></tr><tr><td>37.5</td><td>15.2</td></tr><tr><td>44.0</td><td>30.2</td></tr><tr><td>34.8</td><td>11.0</td></tr><tr><td>40.8</td><td>33.6</td></tr><tr><td>37.3</td><td>29.3</td></tr><tr><td>41.6</td><td>26.0</td></tr><tr><td>42.2</td><td>31.9</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"17e873b7-7c8b-4f1c-832d-2b8cd1f29f7b\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT Ankle, BodyFat FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Ankle</th><th>BodyFat</th></tr></thead><tbody><tr><td>21.9</td><td>12.3</td></tr><tr><td>23.4</td><td>6.1</td></tr><tr><td>24.0</td><td>25.3</td></tr><tr><td>22.8</td><td>10.4</td></tr><tr><td>24.0</td><td>28.7</td></tr><tr><td>25.6</td><td>20.9</td></tr><tr><td>22.9</td><td>19.2</td></tr><tr><td>23.2</td><td>12.4</td></tr><tr><td>23.8</td><td>4.1</td></tr><tr><td>25.0</td><td>11.7</td></tr><tr><td>25.2</td><td>7.1</td></tr><tr><td>25.9</td><td>7.8</td></tr><tr><td>21.5</td><td>20.8</td></tr><tr><td>23.7</td><td>21.2</td></tr><tr><td>23.1</td><td>22.1</td></tr><tr><td>21.7</td><td>20.9</td></tr><tr><td>23.1</td><td>29.0</td></tr><tr><td>24.4</td><td>22.9</td></tr><tr><td>22.9</td><td>16.0</td></tr><tr><td>24.0</td><td>16.5</td></tr><tr><td>22.1</td><td>19.1</td></tr><tr><td>24.6</td><td>15.2</td></tr><tr><td>22.2</td><td>15.6</td></tr><tr><td>22.1</td><td>17.7</td></tr><tr><td>22.9</td><td>14.0</td></tr><tr><td>22.5</td><td>3.7</td></tr><tr><td>21.4</td><td>7.9</td></tr><tr><td>21.0</td><td>22.9</td></tr><tr><td>21.3</td><td>3.7</td></tr><tr><td>22.6</td><td>8.8</td></tr><tr><td>33.9</td><td>11.9</td></tr><tr><td>21.5</td><td>5.7</td></tr><tr><td>24.5</td><td>11.8</td></tr><tr><td>25.2</td><td>21.3</td></tr><tr><td>26.3</td><td>32.3</td></tr><tr><td>21.9</td><td>40.1</td></tr><tr><td>22.6</td><td>24.2</td></tr><tr><td>24.7</td><td>28.4</td></tr><tr><td>29.6</td><td>35.2</td></tr><tr><td>24.7</td><td>32.6</td></tr><tr><td>26.6</td><td>34.5</td></tr><tr><td>23.7</td><td>32.9</td></tr><tr><td>25.0</td><td>31.6</td></tr><tr><td>23.0</td><td>32.0</td></tr><tr><td>21.0</td><td>7.7</td></tr><tr><td>23.4</td><td>13.9</td></tr><tr><td>22.5</td><td>10.8</td></tr><tr><td>21.9</td><td>5.6</td></tr><tr><td>20.6</td><td>13.6</td></tr><tr><td>21.9</td><td>4.0</td></tr><tr><td>22.4</td><td>10.2</td></tr><tr><td>21.0</td><td>6.6</td></tr><tr><td>21.4</td><td>8.0</td></tr><tr><td>22.6</td><td>6.3</td></tr><tr><td>22.5</td><td>3.9</td></tr><tr><td>22.0</td><td>22.6</td></tr><tr><td>22.5</td><td>20.4</td></tr><tr><td>22.7</td><td>28.0</td></tr><tr><td>22.5</td><td>31.5</td></tr><tr><td>22.9</td><td>24.6</td></tr><tr><td>23.1</td><td>26.1</td></tr><tr><td>22.1</td><td>29.8</td></tr><tr><td>23.6</td><td>30.7</td></tr><tr><td>22.7</td><td>25.8</td></tr><tr><td>24.7</td><td>32.3</td></tr><tr><td>22.7</td><td>30.0</td></tr><tr><td>21.7</td><td>21.5</td></tr><tr><td>21.5</td><td>13.8</td></tr><tr><td>22.4</td><td>6.3</td></tr><tr><td>21.6</td><td>12.9</td></tr><tr><td>22.4</td><td>24.3</td></tr><tr><td>21.6</td><td>8.8</td></tr><tr><td>23.1</td><td>8.5</td></tr><tr><td>19.1</td><td>13.5</td></tr><tr><td>20.9</td><td>11.8</td></tr><tr><td>21.4</td><td>18.5</td></tr><tr><td>24.2</td><td>8.8</td></tr><tr><td>21.8</td><td>22.2</td></tr><tr><td>21.5</td><td>21.5</td></tr><tr><td>22.7</td><td>18.8</td></tr><tr><td>23.7</td><td>31.4</td></tr><tr><td>22.0</td><td>26.8</td></tr><tr><td>23.0</td><td>18.4</td></tr><tr><td>24.1</td><td>27.0</td></tr><tr><td>22.0</td><td>27.0</td></tr><tr><td>33.7</td><td>26.6</td></tr><tr><td>21.8</td><td>14.9</td></tr><tr><td>23.3</td><td>23.1</td></tr><tr><td>23.8</td><td>8.3</td></tr><tr><td>24.4</td><td>14.1</td></tr><tr><td>22.5</td><td>20.5</td></tr><tr><td>23.1</td><td>18.2</td></tr><tr><td>22.1</td><td>8.5</td></tr><tr><td>24.5</td><td>24.9</td></tr><tr><td>24.6</td><td>9.0</td></tr><tr><td>23.2</td><td>17.4</td></tr><tr><td>22.9</td><td>9.6</td></tr><tr><td>23.3</td><td>11.3</td></tr><tr><td>21.9</td><td>17.8</td></tr><tr><td>22.3</td><td>22.2</td></tr><tr><td>22.3</td><td>21.2</td></tr><tr><td>22.4</td><td>20.4</td></tr><tr><td>23.2</td><td>20.1</td></tr><tr><td>25.4</td><td>22.3</td></tr><tr><td>22.0</td><td>25.4</td></tr><tr><td>24.8</td><td>18.0</td></tr><tr><td>23.5</td><td>19.3</td></tr><tr><td>23.4</td><td>18.3</td></tr><tr><td>24.8</td><td>17.3</td></tr><tr><td>22.8</td><td>21.4</td></tr><tr><td>22.3</td><td>19.7</td></tr><tr><td>23.6</td><td>28.0</td></tr><tr><td>23.9</td><td>22.1</td></tr><tr><td>21.9</td><td>21.3</td></tr><tr><td>21.8</td><td>26.7</td></tr><tr><td>22.1</td><td>16.7</td></tr><tr><td>22.8</td><td>20.1</td></tr><tr><td>23.3</td><td>13.9</td></tr><tr><td>24.8</td><td>25.8</td></tr><tr><td>24.5</td><td>18.1</td></tr><tr><td>24.6</td><td>27.9</td></tr><tr><td>23.2</td><td>25.3</td></tr><tr><td>22.6</td><td>14.7</td></tr><tr><td>22.1</td><td>16.0</td></tr><tr><td>23.5</td><td>13.8</td></tr><tr><td>21.9</td><td>17.5</td></tr><tr><td>22.2</td><td>27.2</td></tr><tr><td>20.8</td><td>17.4</td></tr><tr><td>21.8</td><td>20.8</td></tr><tr><td>22.2</td><td>14.9</td></tr><tr><td>23.2</td><td>18.1</td></tr><tr><td>23.0</td><td>22.7</td></tr><tr><td>22.6</td><td>23.6</td></tr><tr><td>20.5</td><td>26.1</td></tr><tr><td>23.0</td><td>24.4</td></tr><tr><td>22.7</td><td>27.1</td></tr><tr><td>22.4</td><td>21.8</td></tr><tr><td>23.8</td><td>29.4</td></tr><tr><td>22.5</td><td>22.4</td></tr><tr><td>24.5</td><td>20.4</td></tr><tr><td>21.6</td><td>24.9</td></tr><tr><td>22.0</td><td>18.3</td></tr><tr><td>22.3</td><td>23.3</td></tr><tr><td>22.7</td><td>9.4</td></tr><tr><td>23.2</td><td>10.3</td></tr><tr><td>22.0</td><td>14.2</td></tr><tr><td>25.2</td><td>19.2</td></tr><tr><td>24.6</td><td>29.6</td></tr><tr><td>22.0</td><td>5.3</td></tr><tr><td>24.7</td><td>25.2</td></tr><tr><td>20.4</td><td>9.4</td></tr><tr><td>25.1</td><td>19.6</td></tr><tr><td>23.8</td><td>10.1</td></tr><tr><td>22.8</td><td>16.5</td></tr><tr><td>24.9</td><td>21.0</td></tr><tr><td>21.7</td><td>17.3</td></tr><tr><td>25.2</td><td>31.2</td></tr><tr><td>25.0</td><td>10.0</td></tr><tr><td>21.8</td><td>12.5</td></tr><tr><td>24.6</td><td>22.5</td></tr><tr><td>21.0</td><td>9.4</td></tr><tr><td>25.0</td><td>14.6</td></tr><tr><td>23.5</td><td>13.0</td></tr><tr><td>20.4</td><td>15.1</td></tr><tr><td>23.4</td><td>27.3</td></tr><tr><td>25.6</td><td>19.2</td></tr><tr><td>21.9</td><td>21.8</td></tr><tr><td>24.6</td><td>20.3</td></tr><tr><td>24.0</td><td>34.3</td></tr><tr><td>23.4</td><td>16.5</td></tr><tr><td>22.1</td><td>3.0</td></tr><tr><td>22.0</td><td>0.7</td></tr><tr><td>22.4</td><td>20.5</td></tr><tr><td>21.5</td><td>16.9</td></tr><tr><td>24.0</td><td>25.3</td></tr><tr><td>22.2</td><td>9.9</td></tr><tr><td>22.0</td><td>13.1</td></tr><tr><td>24.8</td><td>29.9</td></tr><tr><td>24.7</td><td>22.5</td></tr><tr><td>25.8</td><td>16.9</td></tr><tr><td>24.1</td><td>26.6</td></tr><tr><td>20.2</td><td>0.0</td></tr><tr><td>21.8</td><td>11.5</td></tr><tr><td>22.7</td><td>12.1</td></tr><tr><td>22.4</td><td>17.5</td></tr><tr><td>22.6</td><td>8.6</td></tr><tr><td>24.7</td><td>23.6</td></tr><tr><td>23.5</td><td>20.4</td></tr><tr><td>24.8</td><td>20.5</td></tr><tr><td>22.9</td><td>24.4</td></tr><tr><td>21.8</td><td>11.4</td></tr><tr><td>25.5</td><td>38.1</td></tr><tr><td>24.7</td><td>15.9</td></tr><tr><td>26.0</td><td>24.7</td></tr><tr><td>22.4</td><td>22.8</td></tr><tr><td>24.1</td><td>25.5</td></tr><tr><td>24.0</td><td>22.0</td></tr><tr><td>22.3</td><td>17.7</td></tr><tr><td>22.5</td><td>6.6</td></tr><tr><td>24.1</td><td>23.6</td></tr><tr><td>23.8</td><td>12.2</td></tr><tr><td>22.0</td><td>22.1</td></tr><tr><td>23.7</td><td>28.7</td></tr><tr><td>24.0</td><td>6.0</td></tr><tr><td>21.8</td><td>34.8</td></tr><tr><td>24.1</td><td>16.6</td></tr><tr><td>21.4</td><td>32.9</td></tr><tr><td>23.3</td><td>32.8</td></tr><tr><td>22.5</td><td>9.6</td></tr><tr><td>22.6</td><td>10.8</td></tr><tr><td>21.7</td><td>7.1</td></tr><tr><td>23.2</td><td>27.2</td></tr><tr><td>23.0</td><td>19.5</td></tr><tr><td>25.5</td><td>18.7</td></tr><tr><td>21.8</td><td>19.5</td></tr><tr><td>23.6</td><td>47.5</td></tr><tr><td>21.5</td><td>13.6</td></tr><tr><td>22.6</td><td>7.5</td></tr><tr><td>22.9</td><td>24.5</td></tr><tr><td>22.0</td><td>15.0</td></tr><tr><td>23.9</td><td>12.4</td></tr><tr><td>27.0</td><td>26.0</td></tr><tr><td>23.4</td><td>11.5</td></tr><tr><td>22.5</td><td>5.2</td></tr><tr><td>21.8</td><td>10.9</td></tr><tr><td>19.7</td><td>12.5</td></tr><tr><td>22.6</td><td>14.8</td></tr><tr><td>23.4</td><td>25.2</td></tr><tr><td>22.5</td><td>14.9</td></tr><tr><td>23.2</td><td>17.0</td></tr><tr><td>21.3</td><td>10.6</td></tr><tr><td>22.8</td><td>16.1</td></tr><tr><td>23.6</td><td>15.4</td></tr><tr><td>21.0</td><td>26.7</td></tr><tr><td>21.0</td><td>25.8</td></tr><tr><td>22.7</td><td>18.6</td></tr><tr><td>23.0</td><td>24.8</td></tr><tr><td>22.3</td><td>27.3</td></tr><tr><td>22.3</td><td>12.4</td></tr><tr><td>24.0</td><td>29.9</td></tr><tr><td>20.1</td><td>17.0</td></tr><tr><td>23.4</td><td>35.0</td></tr><tr><td>24.6</td><td>30.4</td></tr><tr><td>25.4</td><td>32.6</td></tr><tr><td>24.1</td><td>29.0</td></tr><tr><td>22.6</td><td>15.2</td></tr><tr><td>22.6</td><td>30.2</td></tr><tr><td>21.5</td><td>11.0</td></tr><tr><td>23.2</td><td>33.6</td></tr><tr><td>21.5</td><td>29.3</td></tr><tr><td>22.7</td><td>26.0</td></tr><tr><td>24.6</td><td>31.9</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"e8aed46d-82a6-46db-84d3-db4c7311819a\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT Biceps, BodyFat FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Biceps</th><th>BodyFat</th></tr></thead><tbody><tr><td>32.0</td><td>12.3</td></tr><tr><td>30.5</td><td>6.1</td></tr><tr><td>28.8</td><td>25.3</td></tr><tr><td>32.4</td><td>10.4</td></tr><tr><td>32.2</td><td>28.7</td></tr><tr><td>35.7</td><td>20.9</td></tr><tr><td>31.9</td><td>19.2</td></tr><tr><td>30.5</td><td>12.4</td></tr><tr><td>35.9</td><td>4.1</td></tr><tr><td>35.6</td><td>11.7</td></tr><tr><td>32.8</td><td>7.1</td></tr><tr><td>37.2</td><td>7.8</td></tr><tr><td>32.5</td><td>20.8</td></tr><tr><td>36.9</td><td>21.2</td></tr><tr><td>36.1</td><td>22.1</td></tr><tr><td>31.1</td><td>20.9</td></tr><tr><td>36.2</td><td>29.0</td></tr><tr><td>38.2</td><td>22.9</td></tr><tr><td>37.2</td><td>16.0</td></tr><tr><td>37.1</td><td>16.5</td></tr><tr><td>32.5</td><td>19.1</td></tr><tr><td>33.0</td><td>15.2</td></tr><tr><td>27.9</td><td>15.6</td></tr><tr><td>29.8</td><td>17.7</td></tr><tr><td>31.1</td><td>14.0</td></tr><tr><td>29.9</td><td>3.7</td></tr><tr><td>28.7</td><td>7.9</td></tr><tr><td>29.2</td><td>22.9</td></tr><tr><td>30.5</td><td>3.7</td></tr><tr><td>30.1</td><td>8.8</td></tr><tr><td>32.5</td><td>11.9</td></tr><tr><td>30.1</td><td>5.7</td></tr><tr><td>29.0</td><td>11.8</td></tr><tr><td>37.5</td><td>21.3</td></tr><tr><td>37.3</td><td>32.3</td></tr><tr><td>32.0</td><td>40.1</td></tr><tr><td>35.1</td><td>24.2</td></tr><tr><td>33.2</td><td>28.4</td></tr><tr><td>45.0</td><td>35.2</td></tr><tr><td>34.1</td><td>32.6</td></tr><tr><td>36.4</td><td>34.5</td></tr><tr><td>33.6</td><td>32.9</td></tr><tr><td>36.7</td><td>31.6</td></tr><tr><td>35.8</td><td>32.0</td></tr><tr><td>26.1</td><td>7.7</td></tr><tr><td>29.7</td><td>13.9</td></tr><tr><td>27.9</td><td>10.8</td></tr><tr><td>28.8</td><td>5.6</td></tr><tr><td>28.8</td><td>13.6</td></tr><tr><td>26.8</td><td>4.0</td></tr><tr><td>26.0</td><td>10.2</td></tr><tr><td>26.7</td><td>6.6</td></tr><tr><td>29.6</td><td>8.0</td></tr><tr><td>38.5</td><td>6.3</td></tr><tr><td>27.7</td><td>3.9</td></tr><tr><td>35.9</td><td>22.6</td></tr><tr><td>33.1</td><td>20.4</td></tr><tr><td>37.7</td><td>28.0</td></tr><tr><td>31.6</td><td>31.5</td></tr><tr><td>34.5</td><td>24.6</td></tr><tr><td>36.2</td><td>26.1</td></tr><tr><td>32.5</td><td>29.8</td></tr><tr><td>32.7</td><td>30.7</td></tr><tr><td>33.6</td><td>25.8</td></tr><tr><td>35.3</td><td>32.3</td></tr><tr><td>34.8</td><td>30.0</td></tr><tr><td>29.6</td><td>21.5</td></tr><tr><td>32.8</td><td>13.8</td></tr><tr><td>32.6</td><td>6.3</td></tr><tr><td>27.3</td><td>12.9</td></tr><tr><td>31.5</td><td>24.3</td></tr><tr><td>30.3</td><td>8.8</td></tr><tr><td>29.7</td><td>8.5</td></tr><tr><td>29.3</td><td>13.5</td></tr><tr><td>29.4</td><td>11.8</td></tr><tr><td>29.3</td><td>18.5</td></tr><tr><td>30.2</td><td>8.8</td></tr><tr><td>30.8</td><td>22.2</td></tr><tr><td>31.4</td><td>21.5</td></tr><tr><td>30.3</td><td>18.8</td></tr><tr><td>29.4</td><td>31.4</td></tr><tr><td>29.9</td><td>26.8</td></tr><tr><td>34.3</td><td>18.4</td></tr><tr><td>31.2</td><td>27.0</td></tr><tr><td>29.7</td><td>27.0</td></tr><tr><td>32.4</td><td>26.6</td></tr><tr><td>32.6</td><td>14.9</td></tr><tr><td>29.2</td><td>23.1</td></tr><tr><td>30.2</td><td>8.3</td></tr><tr><td>28.8</td><td>14.1</td></tr><tr><td>29.1</td><td>20.5</td></tr><tr><td>31.4</td><td>18.2</td></tr><tr><td>30.1</td><td>8.5</td></tr><tr><td>33.3</td><td>24.9</td></tr><tr><td>30.3</td><td>9.0</td></tr><tr><td>32.9</td><td>17.4</td></tr><tr><td>31.6</td><td>9.6</td></tr><tr><td>30.6</td><td>11.3</td></tr><tr><td>31.6</td><td>17.8</td></tr><tr><td>35.3</td><td>22.2</td></tr><tr><td>32.2</td><td>21.2</td></tr><tr><td>27.9</td><td>20.4</td></tr><tr><td>31.0</td><td>20.1</td></tr><tr><td>31.0</td><td>22.3</td></tr><tr><td>30.1</td><td>25.4</td></tr><tr><td>31.0</td><td>18.0</td></tr><tr><td>30.5</td><td>19.3</td></tr><tr><td>35.1</td><td>18.3</td></tr><tr><td>35.1</td><td>17.3</td></tr><tr><td>32.1</td><td>21.4</td></tr><tr><td>33.3</td><td>19.7</td></tr><tr><td>33.5</td><td>28.0</td></tr><tr><td>35.3</td><td>22.1</td></tr><tr><td>30.7</td><td>21.3</td></tr><tr><td>31.8</td><td>26.7</td></tr><tr><td>29.8</td><td>16.7</td></tr><tr><td>29.9</td><td>20.1</td></tr><tr><td>33.4</td><td>13.9</td></tr><tr><td>33.6</td><td>25.8</td></tr><tr><td>32.1</td><td>18.1</td></tr><tr><td>33.9</td><td>27.9</td></tr><tr><td>33.0</td><td>25.3</td></tr><tr><td>34.4</td><td>14.7</td></tr><tr><td>30.6</td><td>16.0</td></tr><tr><td>34.4</td><td>13.8</td></tr><tr><td>35.6</td><td>17.5</td></tr><tr><td>33.8</td><td>27.2</td></tr><tr><td>33.9</td><td>17.4</td></tr><tr><td>33.3</td><td>20.8</td></tr><tr><td>31.6</td><td>14.9</td></tr><tr><td>27.5</td><td>18.1</td></tr><tr><td>31.2</td><td>22.7</td></tr><tr><td>33.5</td><td>23.6</td></tr><tr><td>33.6</td><td>26.1</td></tr><tr><td>34.0</td><td>24.4</td></tr><tr><td>30.9</td><td>27.1</td></tr><tr><td>32.7</td><td>21.8</td></tr><tr><td>34.3</td><td>29.4</td></tr><tr><td>31.7</td><td>22.4</td></tr><tr><td>35.5</td><td>20.4</td></tr><tr><td>30.8</td><td>24.9</td></tr><tr><td>32.0</td><td>18.3</td></tr><tr><td>31.6</td><td>23.3</td></tr><tr><td>30.5</td><td>9.4</td></tr><tr><td>31.8</td><td>10.3</td></tr><tr><td>33.5</td><td>14.2</td></tr><tr><td>36.1</td><td>19.2</td></tr><tr><td>33.3</td><td>29.6</td></tr><tr><td>25.8</td><td>5.3</td></tr><tr><td>36.0</td><td>25.2</td></tr><tr><td>31.6</td><td>9.4</td></tr><tr><td>38.5</td><td>19.6</td></tr><tr><td>27.8</td><td>10.1</td></tr><tr><td>30.6</td><td>16.5</td></tr><tr><td>33.7</td><td>21.0</td></tr><tr><td>32.2</td><td>17.3</td></tr><tr><td>35.2</td><td>31.2</td></tr><tr><td>31.6</td><td>10.0</td></tr><tr><td>27.0</td><td>12.5</td></tr><tr><td>30.1</td><td>22.5</td></tr><tr><td>27.0</td><td>9.4</td></tr><tr><td>31.3</td><td>14.6</td></tr><tr><td>33.5</td><td>13.0</td></tr><tr><td>28.3</td><td>15.1</td></tr><tr><td>34.0</td><td>27.3</td></tr><tr><td>36.4</td><td>19.2</td></tr><tr><td>30.2</td><td>21.8</td></tr><tr><td>37.2</td><td>20.3</td></tr><tr><td>36.1</td><td>34.3</td></tr><tr><td>32.5</td><td>16.5</td></tr><tr><td>30.4</td><td>3.0</td></tr><tr><td>24.8</td><td>0.7</td></tr><tr><td>31.0</td><td>20.5</td></tr><tr><td>32.4</td><td>16.9</td></tr><tr><td>35.4</td><td>25.3</td></tr><tr><td>31.0</td><td>9.9</td></tr><tr><td>31.5</td><td>13.1</td></tr><tr><td>34.4</td><td>29.9</td></tr><tr><td>34.8</td><td>22.5</td></tr><tr><td>39.1</td><td>16.9</td></tr><tr><td>35.6</td><td>26.6</td></tr><tr><td>27.7</td><td>0.0</td></tr><tr><td>31.4</td><td>11.5</td></tr><tr><td>30.0</td><td>12.1</td></tr><tr><td>30.5</td><td>17.5</td></tr><tr><td>32.9</td><td>8.6</td></tr><tr><td>37.2</td><td>23.6</td></tr><tr><td>36.4</td><td>20.4</td></tr><tr><td>36.6</td><td>20.5</td></tr><tr><td>33.4</td><td>24.4</td></tr><tr><td>29.6</td><td>11.4</td></tr><tr><td>37.1</td><td>38.1</td></tr><tr><td>34.0</td><td>15.9</td></tr><tr><td>33.7</td><td>24.7</td></tr><tr><td>31.7</td><td>22.8</td></tr><tr><td>32.9</td><td>25.5</td></tr><tr><td>31.2</td><td>22.0</td></tr><tr><td>30.8</td><td>17.7</td></tr><tr><td>30.6</td><td>6.6</td></tr><tr><td>33.8</td><td>23.6</td></tr><tr><td>31.7</td><td>12.2</td></tr><tr><td>29.4</td><td>22.1</td></tr><tr><td>32.1</td><td>28.7</td></tr><tr><td>32.9</td><td>6.0</td></tr><tr><td>34.8</td><td>34.8</td></tr><tr><td>37.3</td><td>16.6</td></tr><tr><td>33.1</td><td>32.9</td></tr><tr><td>36.7</td><td>32.8</td></tr><tr><td>31.4</td><td>9.6</td></tr><tr><td>29.0</td><td>10.8</td></tr><tr><td>30.9</td><td>7.1</td></tr><tr><td>36.8</td><td>27.2</td></tr><tr><td>29.5</td><td>19.5</td></tr><tr><td>32.7</td><td>18.7</td></tr><tr><td>28.6</td><td>19.5</td></tr><tr><td>34.7</td><td>47.5</td></tr><tr><td>31.3</td><td>13.6</td></tr><tr><td>27.5</td><td>7.5</td></tr><tr><td>35.7</td><td>24.5</td></tr><tr><td>28.5</td><td>15.0</td></tr><tr><td>31.4</td><td>12.4</td></tr><tr><td>38.4</td><td>26.0</td></tr><tr><td>27.9</td><td>11.5</td></tr><tr><td>29.4</td><td>5.2</td></tr><tr><td>34.1</td><td>10.9</td></tr><tr><td>25.3</td><td>12.5</td></tr><tr><td>33.4</td><td>14.8</td></tr><tr><td>33.2</td><td>25.2</td></tr><tr><td>34.6</td><td>14.9</td></tr><tr><td>32.4</td><td>17.0</td></tr><tr><td>31.7</td><td>10.6</td></tr><tr><td>31.8</td><td>16.1</td></tr><tr><td>30.9</td><td>15.4</td></tr><tr><td>27.8</td><td>26.7</td></tr><tr><td>31.3</td><td>25.8</td></tr><tr><td>30.3</td><td>18.6</td></tr><tr><td>32.6</td><td>24.8</td></tr><tr><td>35.1</td><td>27.3</td></tr><tr><td>29.8</td><td>12.4</td></tr><tr><td>35.9</td><td>29.9</td></tr><tr><td>28.5</td><td>17.0</td></tr><tr><td>34.9</td><td>35.0</td></tr><tr><td>35.6</td><td>30.4</td></tr><tr><td>35.3</td><td>32.6</td></tr><tr><td>32.1</td><td>29.0</td></tr><tr><td>29.2</td><td>15.2</td></tr><tr><td>37.5</td><td>30.2</td></tr><tr><td>25.6</td><td>11.0</td></tr><tr><td>35.2</td><td>33.6</td></tr><tr><td>31.3</td><td>29.3</td></tr><tr><td>30.5</td><td>26.0</td></tr><tr><td>33.7</td><td>31.9</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"236bacbf-5b06-441e-a14f-f7b5fdad93c6\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT Forearm, BodyFat FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Forearm</th><th>BodyFat</th></tr></thead><tbody><tr><td>27.4</td><td>12.3</td></tr><tr><td>28.9</td><td>6.1</td></tr><tr><td>25.2</td><td>25.3</td></tr><tr><td>29.4</td><td>10.4</td></tr><tr><td>27.7</td><td>28.7</td></tr><tr><td>30.6</td><td>20.9</td></tr><tr><td>27.8</td><td>19.2</td></tr><tr><td>29.0</td><td>12.4</td></tr><tr><td>31.1</td><td>4.1</td></tr><tr><td>30.0</td><td>11.7</td></tr><tr><td>29.4</td><td>7.1</td></tr><tr><td>30.2</td><td>7.8</td></tr><tr><td>28.6</td><td>20.8</td></tr><tr><td>31.6</td><td>21.2</td></tr><tr><td>30.5</td><td>22.1</td></tr><tr><td>26.4</td><td>20.9</td></tr><tr><td>30.8</td><td>29.0</td></tr><tr><td>31.6</td><td>22.9</td></tr><tr><td>30.5</td><td>16.0</td></tr><tr><td>30.1</td><td>16.5</td></tr><tr><td>30.3</td><td>19.1</td></tr><tr><td>32.8</td><td>15.2</td></tr><tr><td>25.9</td><td>15.6</td></tr><tr><td>26.7</td><td>17.7</td></tr><tr><td>28.0</td><td>14.0</td></tr><tr><td>28.2</td><td>3.7</td></tr><tr><td>27.0</td><td>7.9</td></tr><tr><td>26.6</td><td>22.9</td></tr><tr><td>27.9</td><td>3.7</td></tr><tr><td>26.7</td><td>8.8</td></tr><tr><td>27.7</td><td>11.9</td></tr><tr><td>26.4</td><td>5.7</td></tr><tr><td>30.0</td><td>11.8</td></tr><tr><td>31.5</td><td>21.3</td></tr><tr><td>31.7</td><td>32.3</td></tr><tr><td>29.8</td><td>40.1</td></tr><tr><td>30.6</td><td>24.2</td></tr><tr><td>30.5</td><td>28.4</td></tr><tr><td>29.0</td><td>35.2</td></tr><tr><td>31.0</td><td>32.6</td></tr><tr><td>32.7</td><td>34.5</td></tr><tr><td>28.7</td><td>32.9</td></tr><tr><td>29.8</td><td>31.6</td></tr><tr><td>31.5</td><td>32.0</td></tr><tr><td>23.1</td><td>7.7</td></tr><tr><td>27.4</td><td>13.9</td></tr><tr><td>26.2</td><td>10.8</td></tr><tr><td>26.8</td><td>5.6</td></tr><tr><td>25.5</td><td>13.6</td></tr><tr><td>25.8</td><td>4.0</td></tr><tr><td>25.8</td><td>10.2</td></tr><tr><td>26.1</td><td>6.6</td></tr><tr><td>26.0</td><td>8.0</td></tr><tr><td>27.4</td><td>6.3</td></tr><tr><td>27.5</td><td>3.9</td></tr><tr><td>30.2</td><td>22.6</td></tr><tr><td>28.3</td><td>20.4</td></tr><tr><td>30.9</td><td>28.0</td></tr><tr><td>28.8</td><td>31.5</td></tr><tr><td>29.6</td><td>24.6</td></tr><tr><td>31.8</td><td>26.1</td></tr><tr><td>29.8</td><td>29.8</td></tr><tr><td>29.9</td><td>30.7</td></tr><tr><td>29.0</td><td>25.8</td></tr><tr><td>31.1</td><td>32.3</td></tr><tr><td>30.1</td><td>30.0</td></tr><tr><td>27.4</td><td>21.5</td></tr><tr><td>27.4</td><td>13.8</td></tr><tr><td>28.1</td><td>6.3</td></tr><tr><td>27.1</td><td>12.9</td></tr><tr><td>27.3</td><td>24.3</td></tr><tr><td>27.3</td><td>8.8</td></tr><tr><td>27.3</td><td>8.5</td></tr><tr><td>25.7</td><td>13.5</td></tr><tr><td>27.0</td><td>11.8</td></tr><tr><td>27.0</td><td>18.5</td></tr><tr><td>29.2</td><td>8.8</td></tr><tr><td>25.7</td><td>22.2</td></tr><tr><td>26.8</td><td>21.5</td></tr><tr><td>28.7</td><td>18.8</td></tr><tr><td>27.2</td><td>31.4</td></tr><tr><td>25.2</td><td>26.8</td></tr><tr><td>29.6</td><td>18.4</td></tr><tr><td>27.3</td><td>27.0</td></tr><tr><td>26.3</td><td>27.0</td></tr><tr><td>27.7</td><td>26.6</td></tr><tr><td>28.0</td><td>14.9</td></tr><tr><td>28.4</td><td>23.1</td></tr><tr><td>29.3</td><td>8.3</td></tr><tr><td>29.6</td><td>14.1</td></tr><tr><td>27.7</td><td>20.5</td></tr><tr><td>28.4</td><td>18.2</td></tr><tr><td>28.2</td><td>8.5</td></tr><tr><td>29.6</td><td>24.9</td></tr><tr><td>27.9</td><td>9.0</td></tr><tr><td>30.8</td><td>17.4</td></tr><tr><td>30.1</td><td>9.6</td></tr><tr><td>27.8</td><td>11.3</td></tr><tr><td>27.5</td><td>17.8</td></tr><tr><td>30.9</td><td>22.2</td></tr><tr><td>31.0</td><td>21.2</td></tr><tr><td>26.2</td><td>20.4</td></tr><tr><td>29.2</td><td>20.1</td></tr><tr><td>30.3</td><td>22.3</td></tr><tr><td>27.2</td><td>25.4</td></tr><tr><td>29.4</td><td>18.0</td></tr><tr><td>28.5</td><td>19.3</td></tr><tr><td>29.6</td><td>18.3</td></tr><tr><td>30.7</td><td>17.3</td></tr><tr><td>26.0</td><td>21.4</td></tr><tr><td>28.2</td><td>19.7</td></tr><tr><td>27.8</td><td>28.0</td></tr><tr><td>31.1</td><td>22.1</td></tr><tr><td>27.6</td><td>21.3</td></tr><tr><td>27.3</td><td>26.7</td></tr><tr><td>26.3</td><td>16.7</td></tr><tr><td>28.0</td><td>20.1</td></tr><tr><td>29.8</td><td>13.9</td></tr><tr><td>29.5</td><td>25.8</td></tr><tr><td>28.6</td><td>18.1</td></tr><tr><td>31.2</td><td>27.9</td></tr><tr><td>29.6</td><td>25.3</td></tr><tr><td>28.0</td><td>14.7</td></tr><tr><td>27.5</td><td>16.0</td></tr><tr><td>29.2</td><td>13.8</td></tr><tr><td>30.2</td><td>17.5</td></tr><tr><td>30.3</td><td>27.2</td></tr><tr><td>28.2</td><td>17.4</td></tr><tr><td>29.6</td><td>20.8</td></tr><tr><td>27.8</td><td>14.9</td></tr><tr><td>26.5</td><td>18.1</td></tr><tr><td>28.4</td><td>22.7</td></tr><tr><td>28.6</td><td>23.6</td></tr><tr><td>29.3</td><td>26.1</td></tr><tr><td>29.8</td><td>24.4</td></tr><tr><td>28.8</td><td>27.1</td></tr><tr><td>28.3</td><td>21.8</td></tr><tr><td>28.4</td><td>29.4</td></tr><tr><td>27.4</td><td>22.4</td></tr><tr><td>29.8</td><td>20.4</td></tr><tr><td>27.9</td><td>24.9</td></tr><tr><td>28.5</td><td>18.3</td></tr><tr><td>27.5</td><td>23.3</td></tr><tr><td>27.2</td><td>9.4</td></tr><tr><td>29.7</td><td>10.3</td></tr><tr><td>28.3</td><td>14.2</td></tr><tr><td>30.3</td><td>19.2</td></tr><tr><td>29.7</td><td>29.6</td></tr><tr><td>25.2</td><td>5.3</td></tr><tr><td>30.4</td><td>25.2</td></tr><tr><td>29.0</td><td>9.4</td></tr><tr><td>33.8</td><td>19.6</td></tr><tr><td>26.3</td><td>10.1</td></tr><tr><td>28.3</td><td>16.5</td></tr><tr><td>29.2</td><td>21.0</td></tr><tr><td>27.7</td><td>17.3</td></tr><tr><td>30.7</td><td>31.2</td></tr><tr><td>28.0</td><td>10.0</td></tr><tr><td>34.9</td><td>12.5</td></tr><tr><td>28.2</td><td>22.5</td></tr><tr><td>26.3</td><td>9.4</td></tr><tr><td>29.2</td><td>14.6</td></tr><tr><td>30.6</td><td>13.0</td></tr><tr><td>26.2</td><td>15.1</td></tr><tr><td>31.2</td><td>27.3</td></tr><tr><td>33.7</td><td>19.2</td></tr><tr><td>28.7</td><td>21.8</td></tr><tr><td>33.1</td><td>20.3</td></tr><tr><td>31.8</td><td>34.3</td></tr><tr><td>29.8</td><td>16.5</td></tr><tr><td>27.4</td><td>3.0</td></tr><tr><td>25.9</td><td>0.7</td></tr><tr><td>28.7</td><td>20.5</td></tr><tr><td>28.4</td><td>16.9</td></tr><tr><td>21.0</td><td>25.3</td></tr><tr><td>26.9</td><td>9.9</td></tr><tr><td>26.6</td><td>13.1</td></tr><tr><td>29.5</td><td>29.9</td></tr><tr><td>30.3</td><td>22.5</td></tr><tr><td>32.5</td><td>16.9</td></tr><tr><td>29.0</td><td>26.6</td></tr><tr><td>24.6</td><td>0.0</td></tr><tr><td>28.3</td><td>11.5</td></tr><tr><td>26.4</td><td>12.1</td></tr><tr><td>28.9</td><td>17.5</td></tr><tr><td>29.3</td><td>8.6</td></tr><tr><td>31.8</td><td>23.6</td></tr><tr><td>30.4</td><td>20.4</td></tr><tr><td>32.4</td><td>20.5</td></tr><tr><td>29.2</td><td>24.4</td></tr><tr><td>27.3</td><td>11.4</td></tr><tr><td>31.2</td><td>38.1</td></tr><tr><td>30.1</td><td>15.9</td></tr><tr><td>29.9</td><td>24.7</td></tr><tr><td>27.1</td><td>22.8</td></tr><tr><td>29.8</td><td>25.5</td></tr><tr><td>27.3</td><td>22.0</td></tr><tr><td>27.8</td><td>17.7</td></tr><tr><td>30.0</td><td>6.6</td></tr><tr><td>28.8</td><td>23.6</td></tr><tr><td>28.4</td><td>12.2</td></tr><tr><td>26.6</td><td>22.1</td></tr><tr><td>28.9</td><td>28.7</td></tr><tr><td>29.2</td><td>6.0</td></tr><tr><td>30.7</td><td>34.8</td></tr><tr><td>23.1</td><td>16.6</td></tr><tr><td>29.5</td><td>32.9</td></tr><tr><td>31.6</td><td>32.8</td></tr><tr><td>27.5</td><td>9.6</td></tr><tr><td>26.2</td><td>10.8</td></tr><tr><td>28.8</td><td>7.1</td></tr><tr><td>31.0</td><td>27.2</td></tr><tr><td>27.9</td><td>19.5</td></tr><tr><td>30.0</td><td>18.7</td></tr><tr><td>26.7</td><td>19.5</td></tr><tr><td>29.1</td><td>47.5</td></tr><tr><td>26.3</td><td>13.6</td></tr><tr><td>25.9</td><td>7.5</td></tr><tr><td>30.4</td><td>24.5</td></tr><tr><td>25.7</td><td>15.0</td></tr><tr><td>29.9</td><td>12.4</td></tr><tr><td>32.0</td><td>26.0</td></tr><tr><td>27.0</td><td>11.5</td></tr><tr><td>26.8</td><td>5.2</td></tr><tr><td>31.1</td><td>10.9</td></tr><tr><td>22.0</td><td>12.5</td></tr><tr><td>29.3</td><td>14.8</td></tr><tr><td>30.0</td><td>25.2</td></tr><tr><td>30.1</td><td>14.9</td></tr><tr><td>29.7</td><td>17.0</td></tr><tr><td>27.3</td><td>10.6</td></tr><tr><td>29.1</td><td>16.1</td></tr><tr><td>29.6</td><td>15.4</td></tr><tr><td>26.1</td><td>26.7</td></tr><tr><td>28.7</td><td>25.8</td></tr><tr><td>26.3</td><td>18.6</td></tr><tr><td>28.5</td><td>24.8</td></tr><tr><td>29.6</td><td>27.3</td></tr><tr><td>28.9</td><td>12.4</td></tr><tr><td>30.5</td><td>29.9</td></tr><tr><td>24.8</td><td>17.0</td></tr><tr><td>30.1</td><td>35.0</td></tr><tr><td>30.7</td><td>30.4</td></tr><tr><td>29.8</td><td>32.6</td></tr><tr><td>29.3</td><td>29.0</td></tr><tr><td>27.3</td><td>15.2</td></tr><tr><td>32.6</td><td>30.2</td></tr><tr><td>25.7</td><td>11.0</td></tr><tr><td>28.6</td><td>33.6</td></tr><tr><td>27.2</td><td>29.3</td></tr><tr><td>29.4</td><td>26.0</td></tr><tr><td>30.0</td><td>31.9</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"implicitDf\":true,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"dee20da6-adcc-4838-8dab-5eb03db1adbf\",\"showTitle\":false,\"title\":\"\"}"}
``` python
%sql

SELECT Wrist, BodyFat FROM BodyFat;
```

::: {.output .display_data}
```{=html}
<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>Wrist</th><th>BodyFat</th></tr></thead><tbody><tr><td>17.1</td><td>12.3</td></tr><tr><td>18.2</td><td>6.1</td></tr><tr><td>16.6</td><td>25.3</td></tr><tr><td>18.2</td><td>10.4</td></tr><tr><td>17.7</td><td>28.7</td></tr><tr><td>18.8</td><td>20.9</td></tr><tr><td>17.7</td><td>19.2</td></tr><tr><td>18.8</td><td>12.4</td></tr><tr><td>18.2</td><td>4.1</td></tr><tr><td>19.2</td><td>11.7</td></tr><tr><td>18.5</td><td>7.1</td></tr><tr><td>19.0</td><td>7.8</td></tr><tr><td>17.7</td><td>20.8</td></tr><tr><td>18.8</td><td>21.2</td></tr><tr><td>18.2</td><td>22.1</td></tr><tr><td>16.9</td><td>20.9</td></tr><tr><td>17.3</td><td>29.0</td></tr><tr><td>19.3</td><td>22.9</td></tr><tr><td>18.5</td><td>16.0</td></tr><tr><td>18.2</td><td>16.5</td></tr><tr><td>18.4</td><td>19.1</td></tr><tr><td>19.9</td><td>15.2</td></tr><tr><td>16.7</td><td>15.6</td></tr><tr><td>17.1</td><td>17.7</td></tr><tr><td>17.6</td><td>14.0</td></tr><tr><td>17.7</td><td>3.7</td></tr><tr><td>16.5</td><td>7.9</td></tr><tr><td>17.0</td><td>22.9</td></tr><tr><td>17.2</td><td>3.7</td></tr><tr><td>17.6</td><td>8.8</td></tr><tr><td>18.4</td><td>11.9</td></tr><tr><td>17.9</td><td>5.7</td></tr><tr><td>18.8</td><td>11.8</td></tr><tr><td>18.7</td><td>21.3</td></tr><tr><td>19.7</td><td>32.3</td></tr><tr><td>17.0</td><td>40.1</td></tr><tr><td>19.0</td><td>24.2</td></tr><tr><td>19.4</td><td>28.4</td></tr><tr><td>21.4</td><td>35.2</td></tr><tr><td>18.3</td><td>32.6</td></tr><tr><td>21.4</td><td>34.5</td></tr><tr><td>17.4</td><td>32.9</td></tr><tr><td>18.4</td><td>31.6</td></tr><tr><td>18.8</td><td>32.0</td></tr><tr><td>16.1</td><td>7.7</td></tr><tr><td>18.3</td><td>13.9</td></tr><tr><td>17.3</td><td>10.8</td></tr><tr><td>17.9</td><td>5.6</td></tr><tr><td>16.3</td><td>13.6</td></tr><tr><td>16.8</td><td>4.0</td></tr><tr><td>17.3</td><td>10.2</td></tr><tr><td>17.2</td><td>6.6</td></tr><tr><td>16.9</td><td>8.0</td></tr><tr><td>18.5</td><td>6.3</td></tr><tr><td>18.5</td><td>3.9</td></tr><tr><td>18.9</td><td>22.6</td></tr><tr><td>18.5</td><td>20.4</td></tr><tr><td>19.2</td><td>28.0</td></tr><tr><td>18.2</td><td>31.5</td></tr><tr><td>18.5</td><td>24.6</td></tr><tr><td>20.2</td><td>26.1</td></tr><tr><td>18.3</td><td>29.8</td></tr><tr><td>19.1</td><td>30.7</td></tr><tr><td>18.8</td><td>25.8</td></tr><tr><td>18.4</td><td>32.3</td></tr><tr><td>18.7</td><td>30.0</td></tr><tr><td>17.4</td><td>21.5</td></tr><tr><td>18.7</td><td>13.8</td></tr><tr><td>18.1</td><td>6.3</td></tr><tr><td>17.3</td><td>12.9</td></tr><tr><td>18.6</td><td>24.3</td></tr><tr><td>18.3</td><td>8.8</td></tr><tr><td>18.2</td><td>8.5</td></tr><tr><td>16.9</td><td>13.5</td></tr><tr><td>16.8</td><td>11.8</td></tr><tr><td>18.3</td><td>18.5</td></tr><tr><td>18.1</td><td>8.8</td></tr><tr><td>18.8</td><td>22.2</td></tr><tr><td>18.3</td><td>21.5</td></tr><tr><td>19.0</td><td>18.8</td></tr><tr><td>19.0</td><td>31.4</td></tr><tr><td>17.7</td><td>26.8</td></tr><tr><td>19.0</td><td>18.4</td></tr><tr><td>19.2</td><td>27.0</td></tr><tr><td>18.0</td><td>27.0</td></tr><tr><td>18.2</td><td>26.6</td></tr><tr><td>18.8</td><td>14.9</td></tr><tr><td>18.1</td><td>23.1</td></tr><tr><td>18.8</td><td>8.3</td></tr><tr><td>18.7</td><td>14.1</td></tr><tr><td>17.7</td><td>20.5</td></tr><tr><td>18.8</td><td>18.2</td></tr><tr><td>18.4</td><td>8.5</td></tr><tr><td>19.1</td><td>24.9</td></tr><tr><td>17.8</td><td>9.0</td></tr><tr><td>20.4</td><td>17.4</td></tr><tr><td>18.5</td><td>9.6</td></tr><tr><td>18.2</td><td>11.3</td></tr><tr><td>18.2</td><td>17.8</td></tr><tr><td>18.3</td><td>22.2</td></tr><tr><td>18.6</td><td>21.2</td></tr><tr><td>17.0</td><td>20.4</td></tr><tr><td>18.4</td><td>20.1</td></tr><tr><td>19.7</td><td>22.3</td></tr><tr><td>17.7</td><td>25.4</td></tr><tr><td>18.8</td><td>18.0</td></tr><tr><td>18.1</td><td>19.3</td></tr><tr><td>19.1</td><td>18.3</td></tr><tr><td>19.2</td><td>17.3</td></tr><tr><td>17.3</td><td>21.4</td></tr><tr><td>18.1</td><td>19.7</td></tr><tr><td>17.4</td><td>28.0</td></tr><tr><td>19.8</td><td>22.1</td></tr><tr><td>17.4</td><td>21.3</td></tr><tr><td>17.5</td><td>26.7</td></tr><tr><td>17.3</td><td>16.7</td></tr><tr><td>18.1</td><td>20.1</td></tr><tr><td>19.5</td><td>13.9</td></tr><tr><td>18.5</td><td>25.8</td></tr><tr><td>18.0</td><td>18.1</td></tr><tr><td>19.5</td><td>27.9</td></tr><tr><td>18.4</td><td>25.3</td></tr><tr><td>17.6</td><td>14.7</td></tr><tr><td>17.6</td><td>16.0</td></tr><tr><td>18.0</td><td>13.8</td></tr><tr><td>17.6</td><td>17.5</td></tr><tr><td>17.2</td><td>27.2</td></tr><tr><td>17.4</td><td>17.4</td></tr><tr><td>18.1</td><td>20.8</td></tr><tr><td>17.7</td><td>14.9</td></tr><tr><td>17.6</td><td>18.1</td></tr><tr><td>17.1</td><td>22.7</td></tr><tr><td>17.9</td><td>23.6</td></tr><tr><td>17.3</td><td>26.1</td></tr><tr><td>18.1</td><td>24.4</td></tr><tr><td>17.6</td><td>27.1</td></tr><tr><td>17.1</td><td>21.8</td></tr><tr><td>17.7</td><td>29.4</td></tr><tr><td>17.6</td><td>22.4</td></tr><tr><td>18.7</td><td>20.4</td></tr><tr><td>16.6</td><td>24.9</td></tr><tr><td>17.8</td><td>18.3</td></tr><tr><td>17.9</td><td>23.3</td></tr><tr><td>18.2</td><td>9.4</td></tr><tr><td>18.3</td><td>10.3</td></tr><tr><td>17.3</td><td>14.2</td></tr><tr><td>18.7</td><td>19.2</td></tr><tr><td>18.4</td><td>29.6</td></tr><tr><td>16.9</td><td>5.3</td></tr><tr><td>18.4</td><td>25.2</td></tr><tr><td>17.8</td><td>9.4</td></tr><tr><td>19.6</td><td>19.6</td></tr><tr><td>17.4</td><td>10.1</td></tr><tr><td>17.9</td><td>16.5</td></tr><tr><td>19.4</td><td>21.0</td></tr><tr><td>17.7</td><td>17.3</td></tr><tr><td>19.1</td><td>31.2</td></tr><tr><td>18.6</td><td>10.0</td></tr><tr><td>16.9</td><td>12.5</td></tr><tr><td>18.2</td><td>22.5</td></tr><tr><td>16.5</td><td>9.4</td></tr><tr><td>19.1</td><td>14.6</td></tr><tr><td>19.7</td><td>13.0</td></tr><tr><td>16.5</td><td>15.1</td></tr><tr><td>18.5</td><td>27.3</td></tr><tr><td>19.4</td><td>19.2</td></tr><tr><td>17.7</td><td>21.8</td></tr><tr><td>19.8</td><td>20.3</td></tr><tr><td>18.8</td><td>34.3</td></tr><tr><td>17.4</td><td>16.5</td></tr><tr><td>17.7</td><td>3.0</td></tr><tr><td>16.9</td><td>0.7</td></tr><tr><td>17.7</td><td>20.5</td></tr><tr><td>17.8</td><td>16.9</td></tr><tr><td>20.1</td><td>25.3</td></tr><tr><td>16.9</td><td>9.9</td></tr><tr><td>16.7</td><td>13.1</td></tr><tr><td>18.4</td><td>29.9</td></tr><tr><td>18.1</td><td>22.5</td></tr><tr><td>19.9</td><td>16.9</td></tr><tr><td>19.0</td><td>26.6</td></tr><tr><td>16.5</td><td>0.0</td></tr><tr><td>17.2</td><td>11.5</td></tr><tr><td>17.4</td><td>12.1</td></tr><tr><td>17.7</td><td>17.5</td></tr><tr><td>18.2</td><td>8.6</td></tr><tr><td>20.0</td><td>23.6</td></tr><tr><td>19.1</td><td>20.4</td></tr><tr><td>18.8</td><td>20.5</td></tr><tr><td>18.5</td><td>24.4</td></tr><tr><td>17.9</td><td>11.4</td></tr><tr><td>19.9</td><td>38.1</td></tr><tr><td>18.7</td><td>15.9</td></tr><tr><td>18.5</td><td>24.7</td></tr><tr><td>17.1</td><td>22.8</td></tr><tr><td>18.8</td><td>25.5</td></tr><tr><td>17.4</td><td>22.0</td></tr><tr><td>16.9</td><td>17.7</td></tr><tr><td>18.5</td><td>6.6</td></tr><tr><td>18.8</td><td>23.6</td></tr><tr><td>18.6</td><td>12.2</td></tr><tr><td>17.4</td><td>22.1</td></tr><tr><td>18.7</td><td>28.7</td></tr><tr><td>18.4</td><td>6.0</td></tr><tr><td>17.4</td><td>34.8</td></tr><tr><td>19.4</td><td>16.6</td></tr><tr><td>17.3</td><td>32.9</td></tr><tr><td>18.4</td><td>32.8</td></tr><tr><td>17.7</td><td>9.6</td></tr><tr><td>17.6</td><td>10.8</td></tr><tr><td>17.4</td><td>7.1</td></tr><tr><td>18.9</td><td>27.2</td></tr><tr><td>18.6</td><td>19.5</td></tr><tr><td>19.0</td><td>18.7</td></tr><tr><td>18.0</td><td>19.5</td></tr><tr><td>18.4</td><td>47.5</td></tr><tr><td>17.8</td><td>13.6</td></tr><tr><td>18.6</td><td>7.5</td></tr><tr><td>19.2</td><td>24.5</td></tr><tr><td>17.1</td><td>15.0</td></tr><tr><td>18.9</td><td>12.4</td></tr><tr><td>19.6</td><td>26.0</td></tr><tr><td>17.8</td><td>11.5</td></tr><tr><td>17.0</td><td>5.2</td></tr><tr><td>19.2</td><td>10.9</td></tr><tr><td>15.8</td><td>12.5</td></tr><tr><td>18.8</td><td>14.8</td></tr><tr><td>18.4</td><td>25.2</td></tr><tr><td>18.8</td><td>14.9</td></tr><tr><td>19.0</td><td>17.0</td></tr><tr><td>16.9</td><td>10.6</td></tr><tr><td>19.0</td><td>16.1</td></tr><tr><td>18.0</td><td>15.4</td></tr><tr><td>17.6</td><td>26.7</td></tr><tr><td>18.3</td><td>25.8</td></tr><tr><td>18.3</td><td>18.6</td></tr><tr><td>19.0</td><td>24.8</td></tr><tr><td>18.5</td><td>27.3</td></tr><tr><td>18.3</td><td>12.4</td></tr><tr><td>19.1</td><td>29.9</td></tr><tr><td>16.5</td><td>17.0</td></tr><tr><td>19.4</td><td>35.0</td></tr><tr><td>19.5</td><td>30.4</td></tr><tr><td>19.5</td><td>32.6</td></tr><tr><td>18.5</td><td>29.0</td></tr><tr><td>18.5</td><td>15.2</td></tr><tr><td>18.8</td><td>30.2</td></tr><tr><td>18.5</td><td>11.0</td></tr><tr><td>20.1</td><td>33.6</td></tr><tr><td>18.0</td><td>29.3</td></tr><tr><td>19.8</td><td>26.0</td></tr><tr><td>20.9</td><td>31.9</td></tr></tbody></table></div>
```
:::

::: {.output .display_data}
    Databricks visualization. Run in Databricks to view.
:::
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"129f3027-368f-4b6a-bc1f-bf7f74afadbc\",\"showTitle\":false,\"title\":\"\"}"}
## Prediction of Body Fat
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"68f80524-3f88-4064-a3dd-364508b409fc\",\"showTitle\":false,\"title\":\"\"}"}
### Preparation of data for Machine Learning
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"1a1c293c-d7b2-4c2a-9f1e-6558b066fd8a\",\"showTitle\":false,\"title\":\"\"}"}
I will use VectorAssembler, which is a transformer that combines a given
list of columns into a single vector column. It is useful for combining
raw features and features generated by different feature transformers
into a single feature vector, in order to train ML models like logistic
regression and decision trees.

In each row, the values of the input columns will be concatenated into a
vector in the specified order.
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"79fae5f7-f01d-46a4-ba96-b9b4534c87cc\",\"showTitle\":false,\"title\":\"\"}"}
``` python
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = df.columns, outputCol = 'features')
vdf = vectorAssembler.transform(df)
vdf = vdf.select(['features', 'BodyFat'])
vdf.show(3)
```

::: {.output .stream .stdout}
    +--------------------+-------+
    |            features|BodyFat|
    +--------------------+-------+
    |[1.0708,12.3,23.0...|   12.3|
    |[1.0853,6.1,22.0,...|    6.1|
    |[1.0414,25.3,22.0...|   25.3|
    +--------------------+-------+
    only showing top 3 rows
:::
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"e564f43f-fe88-4355-a503-fbcf7b089794\",\"showTitle\":false,\"title\":\"\"}" jupyter="{\"outputs_hidden\":true}"}
Next, I will split the data into training and testing sets for machine
learning model development. The training set is used to train the model
on patterns in the data, while the testing set is used to assess how
well the model generalizes to new, unseen data. This helps to evaluate
the model\'s performance and check for overfitting.
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"dc59c894-f719-4c1c-9fa8-873401b62f00\",\"showTitle\":false,\"title\":\"\"}"}
``` python
splits = vdf.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]
```
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"a67d9fed-2a32-4440-9ffb-7df6932063ff\",\"showTitle\":false,\"title\":\"\"}"}
### Linear Regression
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"06fb2214-0325-4bd1-8ddc-d1bf07406e34\",\"showTitle\":false,\"title\":\"\"}"}
``` python
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol = 'features', labelCol='BodyFat', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))
```

::: {.output .stream .stdout}
    Coefficients: [-194.29179300477628,0.458974143362629,0.0,0.0,0.0,0.0,0.0,0.0780924418429046,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    Intercept: 208.23935236512824
:::
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"edb4931a-2d62-45ad-a371-83cc58b05236\",\"showTitle\":false,\"title\":\"\"}"}
#### Summary of the model over the training set
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"9b2379fb-95a4-4b79-8e91-3af8cad5eb71\",\"showTitle\":false,\"title\":\"\"}"}
``` python
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
```

::: {.output .stream .stdout}
    RMSE: 0.810689
    r2: 0.990579
:::
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"2bd53b87-f361-4559-b1be-06207f82277f\",\"showTitle\":false,\"title\":\"\"}"}
When we say that the R-squared value is 0.99, it means that around 99%
of the differences or variations we see in the \"BodyFat\" values can be
accounted for or explained by the model we\'ve built. In other words,
the model is doing a really good job of capturing and explaining the
patterns and variability in the \"BodyFat\" data. The closer the
R-squared value is to 1, the better the model fits the data.

RMSE tells us how well our model\'s predictions match up with the real
values. But to understand if the RMSE is good or bad, we need to compare
it to some basic numbers like the average, smallest, and largest values
in our data. This comparison helps us see if our model is doing a decent
job at predicting, considering the range of values we have in our
dataset.
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"7b2235bf-efc7-4a59-9063-1f719ae2a3a0\",\"showTitle\":false,\"title\":\"\"}"}
``` python
train_df.describe().show()
```

::: {.output .stream .stdout}
    +-------+-----------------+
    |summary|          BodyFat|
    +-------+-----------------+
    |  count|              178|
    |   mean|18.79662921348315|
    | stddev|8.375798461820827|
    |    min|              3.0|
    |    max|             47.5|
    +-------+-----------------+
:::
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"19c648c0-1d69-4178-b543-35931c5c4eb3\",\"showTitle\":false,\"title\":\"\"}"}
We can compare the RMSE to the variability in the BodyFat data:

The RMSE of 0.810689 is relatively small compared to the range of
BodyFat values, which go from 3.0 to 47.5.

The mean BodyFat value is 18.80, and the RMSE is smaller than this mean.
This suggests that, on average, our model\'s predictions are quite close
to the actual BodyFat values.

The standard deviation of BodyFat is 8.38, and our RMSE is smaller than
this as well. This indicates that the model\'s predictions are generally
closer to the true values than the average distance between the actual
BodyFat values.
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"d1564981-a103-46e6-8b0c-f0587ea61dc1\",\"showTitle\":false,\"title\":\"\"}"}
#### Summary of the model over the test set
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"0a00e969-487a-4b27-ab77-85effd6ac979\",\"showTitle\":false,\"title\":\"\"}"}
``` python
lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","BodyFat","features").show(10)

from pyspark.ml.evaluation import RegressionEvaluator

lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="BodyFat",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)
```

::: {.output .stream .stdout}
    +------------------+-------+--------------------+
    |        prediction|BodyFat|            features|
    +------------------+-------+--------------------+
    | 35.24403421319681|   34.3|[1.018,34.3,35.0,...|
    |  37.7442456249542|   35.2|[1.0202,35.2,46.0...|
    | 34.75389020147844|   34.8|[1.0209,34.8,44.0...|
    |35.421301558733546|   34.5|[1.0217,34.5,45.0...|
    | 31.52339414091179|   32.9|[1.025,32.9,44.0,...|
    |31.553324400342575|   32.0|[1.0269,32.0,41.0...|
    |31.714280798708785|   31.6|[1.0279,31.6,48.0...|
    |31.188208798198872|   31.5|[1.028,31.5,54.0,...|
    |28.861293236829937|   29.8|[1.0317,29.8,56.0...|
    | 28.82683066826843|   29.4|[1.0325,29.4,43.0...|
    +------------------+-------+--------------------+
    only showing top 10 rows

    R Squared (R2) on test data = 0.994369
    Root Mean Squared Error (RMSE) on test data = 0.622103
:::
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"d4e251d1-3579-4c49-b7e6-3f2955754d6e\",\"showTitle\":false,\"title\":\"\"}"}
A lower RMSE on the test data compared to the train data suggests that
the model is generalizing well to new, unseen data.
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"2ea2f869-bb06-4886-9af2-2877919e68db\",\"showTitle\":false,\"title\":\"\"}"}
### Decision tree regression
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"e6461a76-27b9-4596-b3f4-e2d31f06c862\",\"showTitle\":false,\"title\":\"\"}"}
``` python
from pyspark.ml.regression import DecisionTreeRegressor

dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'BodyFat')
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)
dt_predictions.select('prediction', 'BodyFat', 'features').show(10)

dt_evaluator = RegressionEvaluator(
    labelCol="BodyFat", predictionCol="prediction", metricName="rmse")
rmse = dt_evaluator.evaluate(dt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
```

::: {.output .stream .stdout}
    +------------------+-------+--------------------+
    |        prediction|BodyFat|            features|
    +------------------+-------+--------------------+
    | 37.73333333333333|   34.3|[1.018,34.3,35.0,...|
    | 37.73333333333333|   35.2|[1.0202,35.2,46.0...|
    | 37.73333333333333|   34.8|[1.0209,34.8,44.0...|
    | 37.73333333333333|   34.5|[1.0217,34.5,45.0...|
    |             33.25|   32.9|[1.025,32.9,44.0,...|
    |30.966666666666658|   32.0|[1.0269,32.0,41.0...|
    |30.966666666666658|   31.6|[1.0279,31.6,48.0...|
    |30.966666666666658|   31.5|[1.028,31.5,54.0,...|
    |             29.68|   29.8|[1.0317,29.8,56.0...|
    |             29.68|   29.4|[1.0325,29.4,43.0...|
    +------------------+-------+--------------------+
    only showing top 10 rows

    Root Mean Squared Error (RMSE) on test data = 0.96897
:::
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"9292748b-571f-407f-9e00-daad95e1f83e\",\"showTitle\":false,\"title\":\"\"}"}
### Gradient-boosted tree regression
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"1aebfc87-4b07-4dfb-9c89-7f8488a5ba7e\",\"showTitle\":false,\"title\":\"\"}"}
``` python
from pyspark.ml.regression import GBTRegressor

gbt = GBTRegressor(featuresCol = 'features', labelCol = 'BodyFat', maxIter=10)
gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(test_df)
gbt_predictions.select('prediction', 'BodyFat', 'features').show(10)

gbt_evaluator = RegressionEvaluator(
    labelCol="BodyFat", predictionCol="prediction", metricName="rmse")
rmse = gbt_evaluator.evaluate(gbt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
```

::: {.output .stream .stdout}
    +------------------+-------+--------------------+
    |        prediction|BodyFat|            features|
    +------------------+-------+--------------------+
    |37.671498584197494|   34.3|[1.018,34.3,35.0,...|
    | 36.41591704332286|   35.2|[1.0202,35.2,46.0...|
    |  37.6031676789961|   34.8|[1.0209,34.8,44.0...|
    | 36.41591704332286|   34.5|[1.0217,34.5,45.0...|
    | 33.04101163010007|   32.9|[1.025,32.9,44.0,...|
    | 31.06352318234042|   32.0|[1.0269,32.0,41.0...|
    |30.671988504874715|   31.6|[1.0279,31.6,48.0...|
    |30.839086395037228|   31.5|[1.028,31.5,54.0,...|
    | 29.61138601494879|   29.8|[1.0317,29.8,56.0...|
    |29.811665601100543|   29.4|[1.0325,29.4,43.0...|
    +------------------+-------+--------------------+
    only showing top 10 rows

    Root Mean Squared Error (RMSE) on test data = 0.891016
:::
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"4d88f1b7-fa90-41a7-beba-2619b1499bdb\",\"showTitle\":false,\"title\":\"\"}"}
Linear regression performed the best on this data.
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"e3617758-02fd-41e8-8a45-924346faa553\",\"showTitle\":false,\"title\":\"\"}"}
## Summary

The project involved:

1.  Loading and preprocessing of the dataset
2.  Statistical analysis of the data
3.  Exploratory Data Analysis to uncover patterns and insights
4.  Correlation Analysis to understand relationships between variables
5.  Utilizing tree models to predict Body Fat percentage
6.  The Root Mean Squared Error (RMSE) for each model on the test data
    was:

-   Linear Regression: 0.622103
-   Decision Tree Regression: 0.96897
-   Gradient-Boosted Tree Regression: 0.891016 These results highlight
    the effectiveness of the Linear Regression model in predicting Body
    Fat percentage, outperforming both Linear Decision Tree Regression
    and Gradient-Boosted Tree Regression models.
:::
