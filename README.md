# SMM: 353 Project 

Description:
The primary objective is to understand and compare the sentiments of SFU and UBC subreddits throughout the school year

### Getting Started

Ensure that you have the following libraries installed before running the project:

***pandas:*** For data manipulation and analysis.

***numpy:*** For numerical operations on data.

***scipy:*** For scientific computing and statistical tests.

***matplotlib:*** For creating visualizations.

***statsmodels:*** For advanced statistical models and tests.

***spark:*** ONLY needed if running 1-clean1.py, for cleaning data from cluster.

### Installation

To install the required dependencies, you can use the following command:

```
pip install pandas numpy scipy matplotlib statsmodels
```
To install spark, follow instructions at (https://spark.apache.org/downloads.html) and make sure your terminal environment is properly set up. Something along the lines of

```
export PYSPARK_PYTHON=python3
export PATH=${PATH}:/home/you/spark-3.5.0-bin-hadoop3/bin
```

### Executing Program 
If you would like to run 1-clean1.py to see how stats_data-L was produced, unzip reddit-subset.zip and make sure reddit-subset is in the same directory as the script.

```
spark-submit 1-clean1.py reddit-subset/submissions output
```

To execute the statistical analysis, run the following command in your terminal and ensure that you have the necessary data file (stats_data-L) in the same directory as the script.

```
python3 stat_tests.py stats_data-L
```


### Repository Information

File Information:

***stat_tests.py:*** contains all the statistical tests conducted on the reddit dataset

***collect_data.py*** was used to get the data from the cluster (see References)

***1-clean1.py:*** was used to clean the data and calculate the sentiments and outputs stats_data-L to use for stat_tests.py

***2-sa.py:**** contains feature engineering using the nltk library

***stats_data-L:*** contains cleaned data compressed into a single folder for testing

***reddit-subset.zip:*** contains all the data collected from the cluster


### Authors
Sevy, Mrinal, Monishka

### References

Professor Greg Baker's code for getting data from cluster was utilized (https://coursys.sfu.ca/2023fa-cmpt-353-d1/pages/RedditData)

NLTK resource: (https://realpython.com/python-nltk-sentiment-analysis/#using-nltks-pre-trained-sentiment-analyzer)
