# Power Outage Data Exploration & Prediction

**By Jesse Huang and Danny Lee**

<br>

## Project Overview

This project aims to investigate the factors that lead to power outage events with the hopes of gaining valuable insights into why these events occur and the ways in which outages can be mitigated. If you are interested in learning more about the dataset we based our project on, more information can be found [here](https://www.sciencedirect.com/science/article/pii/S2352340918307182). The power outage data exploration and prediction on website is for DSC 80 at UCSD.

## Introduction

Our lives are becoming increasingly exposed to the far reaching effects of climate change. One of these effects is power outages, and the increasing risk of power outages resulting from natural disasters exacerbated by climate change. In California, climate change has undoubtedly contributed to longer wildfire seasons, with larger and more intense wildfires, causing damage to power infrastructure and disruptions to communities and ecosystems.

Our goal for the first section of this project is to investigate the underlying climate trends over time and geographical patterns to develop a better understanding about power outage risks. To be more specific, <b>how do variables like the number of customers affected by power outage events, changing climate, and geographical position of states contribute to the risk of potential power outages and their duration?</b> We believe that this knowledge is critical for enhancing public safety, supporting vulnerable populations, and mitigating environmental impacts.

The data set we are using has 1534 rows, each representing a major outage in the continental United States from January 2000 to July 2016, and 54 feature columns. Here are a few important ones:

| Column                  | Description  |
|:------------------------|:-------|
| YEAR                    | Indicates the year when the outage event occurred |
| MONTH                   | Indicates the month when the outage event occurred |
| U.S._STATE              | Represents all the states in the continental U.S. |
| CLIMATE.REGION          | Nine Climatically Consistent Regions in Continental U.S. |
| ANOMALY.LEVEL           | Represents the Oceanic El Niño/La Niña Index |
| CLIMATE.CATEGORY        | Represents the climate episodes corresponding to the years |
| OUTAGE.DURATION         | Duration of outage events (in minutes) |
| DEMAND.LOSS.MW          | Amount of peak demand lost during an outage event (in Megawatt) |
| CUSTOMERS.AFFECTED      | Number of customers affected by the power outage event |
| TOTAL.CUSTOMERS         | Annual number of total customers served in the U.S. state |
| POPULATION              | Population in the U.S. state per year |


In our study, we primarily focused on 5 columns: CLIMATE.REGION , ANOMALY.LEVEL, CLIMATE.CATEGORY, DEMAND.LOSS.MW, and POPULATION. We found these columns to be particularly helpful because they represent many of the direct risk characteristics that may contribute to the power outage event itself and the amount of time it takes to restore back the power. Since we are also considering climate variables, many of our exploratory visualizations were conditioned on years from the YEAR column to illustrate trends and patterns over time.


---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

In order to increase the readability and accuracy of our data, we followed the following steps to clean our DataFrame:
1. <b>Dropping unnecessary rows and columns:</b> When we first retrieved the dataset, the columns and indices were mispositioned in the dataframe. We addressed this issue by dropping three rows and one column in order to relocate the column names top of the dataframe and the indices to the left most side of the dataframe.
2. <b>Converting columns to proper dtypes:</b> This step is to make the time column more accurate and accessible. We converted OUTAGE.START.DATE, OUTAGE.START.TIME, OUTAGE.RESTORATION.DATE and OUTAGE.RESTORATION.TIME columns into datetime64[ns] dtype.
3. <b>Converting percentages to proportions:</b> This step allows for arithmetic operations and enhanced interpretability during data analysis. Proportions rid the need of percentage symbols which are difficult to work with when manipulating data and performing conversions.
4. <b>Combining two columns into one:</b> This step combines OUTAGE.START.DATE and OUTAGE.START.TIME into OUTAGE.START, and OUTAGE.END.DATE and OUTAGE.END.TIME into OUTAGE.RESTORATION. This is important because it reduces the size of the dataframe for efficiency, simplicity, and easy access to both the date and time.

<br>

**After cleaning the data, the columns now look like this**

Previously all columns were dtype objects, the latter half of the dataframe now looks like this

| Column                | Dtype          |
|:----------------------|:---------------|
| RES.CUST.PCT          | float64        |
| COM.CUST.PCT          | float64        |
| IND.CUST.PCT          | float64        |
| PC.REALGSP.STATE      | object         |
| PC.REALGSP.USA        | object         |
| PC.REALGSP.REL        | object         |
| PC.REALGSP.CHANGE     | float64        |
| UTIL.REALGSP          | object         |
| TOTAL.REALGSP         | object         |
| UTIL.CONTRI           | float64        |
| PI.UTIL.OFUSA         | float64        |
| POPULATION            | object         |
| POPPCT_URBAN          | float64        |
| POPPCT_UC             | float64        |
| POPDEN_URBAN          | object         |
| POPDEN_UC             | object         |
| POPDEN_RURAL          | object         |
| AREAPCT_URBAN         | float64        |
| AREAPCT_UC            | float64        |
| PCT_LAND              | float64        |
| PCT_WATER_TOT         | float64        |
| PCT_WATER_INLAND      | float64        |
| OUTAGE.START          | datetime64[ns] |
| OUTAGE.RESTORATION    | datetime64[ns] |


### Univariate Analysis

**Distribution of Power Outages**

The histogram below shows the frequency of power outages for each U.S. state from 2000 to 2016 in our dataset.

In this histogram, we notice that California has the highest number of power outages, trailed by Texas, Michigan, and Washington. Conversely, Alaska, South Dekota and North Dekota rarely experience any outages. This may suggest that there is a correlation between the state’s population and the number of power outages experienced.

<iframe
  src="assets/univariate_1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

**Distribution of Power Outages**

The histogram below shows the frequency of power outages for each U.S. state from 2000 to 2016 in our dataset.

In this histogram, we are immediately drawn to the spike in outages during the year of 2011 at 209 power outages. However, the years prior and after experienced no more than 174 outages. While there is a decline in power outages after 2011, the distribution is skewed right. This may suggest that the number of outages increases over the years from 2000 to 2016.

<iframe
  src="assets/univariate_2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Bivariate Analysis
**Distribution of Oceanic Niño Index**

The scatter plot below shows the yearly magnitude and trend of the Oceanic Niño Index from 2000 to 2016 in our dataset.

In this scatter plot, we notice notable fluctuations in the Oceanic Niño Index, similar to a pattern of oscillation in the data. This suggest that the effects of El Niño and La Niña are stronger during certain years. Towards the latter part of the plot, there is a sudden increase in the index, reaching values of 2 and 2.5 not seen prior to 2015, indicating a stronger than expected El Niño or La Niña effect.

<iframe
  src="assets/bivariate_1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

**Distribution of Customers Affected**

The box plot below shows the relationship between the climate regions in U.S. and the number of customers affected by power outage events in those regions from 2000 to 2016 in our dataset.

This box plot reveals that the East North Central climate region experiences the highest impact in terms of customers affected by power outages. We know this from the high 25th and 75th quartiles, and mean when compared to all the other box plots. Following the East North Central climate region are Central, South, and Southwest. This suggests that East North Central power outages are more severe in magnitude and range than any other climate region.

<iframe
  src="assets/bivariate_2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Interesting Aggregates

A pivot table is show below. According to the pivot table, it is apparent that the Southwest and Northwest climate regions were the least affected by power outages, for many of their columns are filled with zeros representing zero hours of power outage during those years. On the other hand, when we look at the columns for the South climate region, all the columns are filled with large values. This indicates that the South climate region was one of the most heavily affected by power outages. These findings are consistent with the boxplot seen above.

|     2000 |   2001 |   2002 |   2003 |   2004 |     2005 |   2006 |   2007 |     2008 |     2009 |     2010 |       2011 |      2012 |     2013 |   2014 |     2015 |     2016 |
|---------:|-------:|-------:|-------:|-------:|---------:|-------:|-------:|---------:|---------:|---------:|-----------:|----------:|---------:|-------:|---------:|---------:|
| 119099   |      0 |  95000 | 318100 | 174000 | 154856   | 190591 | 214838 | 132136   | 120287   | 173870   |  95013.7   | 146644    |  98599.9 | 118609 |  20665.6 |      0   |
|      0   |      0 | 190000 | 357111 | 180536 | 152265   | 283544 | 111112 | 165294   |  93563.9 |  98946.8 |  77225.3   |  86796.2  | 140964   | 105751 | 104572   | 102070   |
|      0   | 130000 | 142181 | 500075 | 104730 |  84916.3 | 141155 | 119944 | 123926   |  79768.6 | 137199   |  98351.2   | 107194    |  35738.3 | 232832 |  99042.2 |  14367.7 |
|      0   |      0 |      0 | 100000 | 122333 |      0   | 176807 | 135216 |   4001.5 |  93300   | 107812   |    884.615 |  71000    | 105000   |      0 | 150033   |   5600   |
| 595250   |  58193 | 609261 | 122843 | 130679 | 307982   | 123860 | 155894 | 586450   | 170576   | 174506   |  73056.5   |  70287.2  | 195482   |  42600 |  78371   | 118680   |
|  91777.8 | 600000 |  77500 | 145583 | 305434 | 306466   |  85000 |  82750 | 152016   |  99250   |  62945.8 | 100822     | 208156    |  97048.6 | 290377 | 113548   |  67781.7 |
| 270456   |      0 |      0 |  68500 |  30000 |      0   |  65000 |      0 |  74031   | 113029   |  31000   |  32866.7   |   7594.75 |  11743.3 |      0 |   1450   |  12556   |
|  32000   | 175611 | 769750 | 115750 | 197855 | 613378   | 493523 | 179952 | 265810   | 246725   | 170847   |  62703.7   |  18166.1  |  44523.3 | 933475 |  20034.5 |  28575   |
|      0   |      0 |      0 |      0 | 120212 |      0   |  15000 |      0 | 126000   |  35500   |      0   |  11500     |      0    |      0   |      0 |      0   |      0   |

Additional research shows, "Tropical cyclones, storms with high winds that originate over tropical oceans, make a power outage 14 times more likely. And a tropical cyclone with heavy precipitation on a hot day—like the hurricanes that each fall hit the Gulf Coast? They make power outages 52 times more likely" [(source)](https://deohs.washington.edu/hsm-blog/these-four-regions-us-are-hardest-hit-power-outages). This article's findings are consistent with the pivot table and boxplot seen above.

## Assessment of Missingness

### NMAR Analysis

NMAR is the term that describes a situation where the chance that a value is missing depends on the actual missing value itself. In the power outages dataframe, the missingness of HURRICANE.NAMES is likely to be NMAR because the severity of the hurricane determines whether the names are recorded or not. Knowing that only notable hurricanes receive names, hurricanes with missing name values may not have been severe enough. Thus, the missingness of name values can be attributed to the hurricane itself. If we want to transform HURRICANE.NAMES into MAR, we could add a severity column that tells you the category of each hurricane.

### Missingness Dependency

** Power Outage Cause and Power Outage Duration **

Null Hypothesis: The missingness of values in CAUSE.CATEGORY.DETAIL does not depend on the values in OUTAGE.DURATION

Alternative Hypothesis: The missingness of values in CAUSE.CATEGORY.DETAIL does depend on the values in OUTAGE.DURATION

Test Statistic: The absolute difference in means

Below shows the empirical distribution of our test statistics with 1000 permutations, the vertical red line marks the observed test statistic and the vertical purple line marks the critical value.

<iframe
  src="assets/missingness_1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

From the permutation test, we fail to reject the null hypothesis because 0.089 is greater than the 0.05 pre-defined cutoff value. Thus, the missingness of CAUSE.CATEGORY.DETAIL is MCAR.

<br>

** Power Outage Cause and Power Outage Duration **

Null Hypothesis: The missingness of the values in CAUSE.CATEGORY.DETAIL does not depend on the values in TOTAL.CUSTOMERS

Alternative Hypothesis: The missingness of the values in CAUSE.CATEGORY.DETAIL does depend on the values in TOTAL.CUSTOMERS

Test Statistic: The absolute difference in means

Below shows the empirical distribution of our test statistics with 1000 permutations, the vertical red line marks the observed test statistic and the vertical purple line marks the critical value.

<iframe
  src="assets/missingness_2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

From the permutation test, we reject the null hypothesis because 0.0 is less than the 0.05 pre-defined cutoff value. Thus, the missingness of CAUSE.CATEGORY.DETAIL is MAR, dependent on TOTAL.CUSTOMERS.


**Data Cleaning**
```py
# Removes three unnecessary rows above column names and unnecessary columns
messy_outages.columns = messy_outages.iloc[4].to_list()
outages = messy_outages.iloc[6:].drop(columns = ['variables']).reset_index(drop=True)
```

```py
# Removes three unnecessary rows above column names and unnecessary columns
messy_outages.columns = messy_outages.iloc[4].to_list()
outages = messy_outages.iloc[6:].drop(columns = ['variables']).reset_index(drop=True)
```

```py
outages.dtypes
```

### OUTPUT

```py
# Combines 'OUTAGE.START.DATE' and 'OUTAGE.START.TIME' into a new pd.Timestamp column called 'OUTAGE.START'
start_date = pd.to_datetime(outages['OUTAGE.START.DATE'])
start_time = pd.to_datetime(outages['OUTAGE.START.TIME'], format = '%H:%M:%S')
combined_start = start_date + pd.to_timedelta(start_time.dt.strftime('%H:%M:%S'))
outages['OUTAGE.START'] = combined_start

# Combines 'OUTAGE.RESTORATION.DATE' and 'OUTAGE.RESTORATION.TIME' into a new pd.Timestamp column called 'OUTAGE.RESTORATION'
end_date = pd.to_datetime(outages['OUTAGE.RESTORATION.DATE'])
end_time = pd.to_datetime(outages['OUTAGE.RESTORATION.TIME'], format = '%H:%M:%S')
combined_end = end_date + pd.to_timedelta(end_time.dt.strftime('%H:%M:%S'))
outages['OUTAGE.RESTORATION'] = combined_end

outages.drop(['OUTAGE.START.DATE',
              'OUTAGE.START.TIME', 
              'OUTAGE.RESTORATION.DATE', 
              'OUTAGE.RESTORATION.TIME'], axis=1, inplace=True)
```

```py
# Converts 'OUTAGE.DURATION' from minutes to hours for clarity and simplification
outages['OUTAGE.DURATION'] = outages['OUTAGE.DURATION'].transform(lambda row: (row / 60).astype('float64'))
```

```py
# Turns percents into proportions
cols_with_percents = ['RES.PERCEN', 'COM.PERCEN', 'IND.PERCEN', 'RES.CUST.PCT', 'COM.CUST.PCT', 'IND.CUST.PCT', 
                      'PC.REALGSP.CHANGE', 'UTIL.CONTRI', 'PI.UTIL.OFUSA', 'POPPCT_URBAN', 'POPPCT_UC', 'AREAPCT_URBAN',
                      'AREAPCT_UC', 'PCT_LAND', 'PCT_WATER_TOT', 'PCT_WATER_INLAND']

for col in cols_with_percents:
    outages[col] = outages[col].transform(lambda row: (row / 100).astype('float64'))
```

```py
outages.dtypes
```

### OUTPUT

```py
outages
```

### TABLE

**Univariate Analysis**

```py
fig_1 = px.histogram(data_frame=outages, x="U.S._STATE")
fig_1.update_layout(title='Distribution of Power Outages by State')
fig_1.update_layout(xaxis_title='U.S. State')
fig_1.update_layout(yaxis_title='Power Outage Counts')
fig_1.update_layout(xaxis_tickangle=-45)
fig_1.show()
```

### PLOT

```py
fig_2 = px.histogram(data_frame=outages, x="YEAR", nbins=17) # CAUSE.CATEGORY might be interesting as well
fig_2.update_layout(title='Distribution of Power Outages by Year')
fig_2.update_layout(xaxis_title='Year')
fig_2.update_layout(yaxis_title='Power Outage Counts')
fig_2.update_traces(marker_color='purple')
fig_2.show()
```

### PLOT

**Bivariate Analysis**

```py
fig_3 = px.scatter(outages, x='YEAR', y='ANOMALY.LEVEL')
fig_3.update_layout(xaxis_title='Year')
fig_3.update_layout(yaxis_title='Oceanic Niño Index')
fig_3.update_layout(title='Distribution of Oceanic Niño Index from 2000 to 2016')
fig_3.update_traces(marker_color='purple')
fig_3.show()
```

### PLOT

```py
box_plot = outages[outages['CUSTOMERS.AFFECTED'] <= 400000]
fig_4 = px.box(box_plot, x="CLIMATE.REGION", y="CUSTOMERS.AFFECTED")
fig_4.update_layout(xaxis_title='Climate Region')
fig_4.update_layout(yaxis_title='Millions of Customers Affected')
fig_4.update_layout(title='Distribution of Customers Affected by Climate Region')
fig_4.show()
```

### PLOT

**Interesting Aggregates**

A pivot table is show below. According to the pivot table, it is apparent that the Southwest and Northwest climate regions were the least affected by power outages, for many of their columns are filled with zeros to represent 0 hours of power outage during those years. On the other hand, when look at the columns for the South climate region, all the columns are filled with large values. This indicates that the South climate region is one of the most heavily affected by power outages. These findings are consistent with the boxplot seen above.

```py
pd.pivot_table(outages, values='CUSTOMERS.AFFECTED', index='CLIMATE.REGION', 
               columns='YEAR', aggfunc='mean', fill_value=0)
```

### TABLE


---

## Assessment of Missingness
```py
def permutation(df, permute_col, no_permute_col):
    n_repetitions = 1000

    differences = []
    for _ in range(n_repetitions):
        
        with_shuffled = df.assign(Shuffled_Col=np.random.permutation(df[permute_col]))
        
        group_means = (
            with_shuffled
            .groupby('Shuffled_Col')
            [no_permute_col]
            .mean()
        )
        
        difference = abs(group_means.diff().iloc[-1])
        differences.append(difference)
        
    return differences
```

**Missingness Dependency**
Null Hypothesis: The missingness of 'CAUSE.CATEGORY.DETAIL' does not depend on 'OUTAGE.DURATION'

Alternative Hypothesis: The missingness of 'CAUSE.CATEGORY.DETAIL' does depend on 'OUTAGE.DURATION'

```py
missing_ind = outages[['CAUSE.CATEGORY.DETAIL', 'OUTAGE.DURATION']].copy()
missing_ind['CAUSE.CATEGORY.DETAIL'] = pd.isna(missing_ind['CAUSE.CATEGORY.DETAIL'])

grouped_means_ind = missing_ind.groupby('CAUSE.CATEGORY.DETAIL')['OUTAGE.DURATION'].mean()
obs_stat_ind = abs(grouped_means_ind.diff().iloc[-1])

differences_ind = permutation(missing_ind, 'CAUSE.CATEGORY.DETAIL', 'OUTAGE.DURATION')
```

```py
fig_ind = px.histogram(
    pd.DataFrame(differences_ind), x=0, nbins=50, histnorm='probability', 
    title='Empirical Distribution of Test Statistic')
fig_ind.update_layout(xaxis_title='Difference in Means')
fig_ind.add_vline(x=obs_stat_ind, line_color='red')
fig_ind.show()
```

### PLOT

```py
p_value_ind = (obs_stat_ind <= np.array(differences_ind)).mean()
p_value_ind
```
0.071

From the permutation test above, we fail to reject null hypothesis because 0.71 is greater than the 0.05 pre-defined cutoff value. Thus, the missingness of 'CAUSE.CATEGORY.DETAIL' is MCAR.


Null Hypothesis: The missingness of 'CAUSE.CATEGORY.DETAIL' does not depend on 'TOTAL.CUSTOMERS'

Alternative Hypothesis: The missingness of 'CAUSE.CATEGORY.DETAIL' does depend on 'TOTAL.CUSTOMERS'

```py
missing_dep = outages[['CAUSE.CATEGORY.DETAIL', 'TOTAL.CUSTOMERS']].copy()
missing_dep['CAUSE.CATEGORY.DETAIL'] = pd.isna(missing_dep['CAUSE.CATEGORY.DETAIL'])

grouped_means_dep = missing_dep.groupby('CAUSE.CATEGORY.DETAIL')['TOTAL.CUSTOMERS'].mean()
obs_stat_dep = abs(grouped_means_dep.diff().iloc[-1])

differences_dep = permutation(missing_dep, 'CAUSE.CATEGORY.DETAIL', 'TOTAL.CUSTOMERS')
```

```py
fig_dep = px.histogram(
    pd.DataFrame(differences_dep), x=0, nbins=50, histnorm='probability', 
    title='Empirical Distribution of Test Statistic')
fig_dep.update_layout(xaxis_title='Difference in Means')
fig_dep.add_vline(x=obs_stat_dep, line_color='red')
fig_dep.show()
```

### PLOT

```py
p_value_dep = (obs_stat_dep <= np.array(differences_dep)).mean()
p_value_dep
```
0.0

From the permutation test above, we reject null hypothesis because 0.0 is less than the 0.05 pre-defined cutoff value. Thus, the missingness of 'CAUSE.CATEGORY.DETAIL' is MAR, dependent on 'TOTAL.CUSTOMERS'.
---

## Hypothesis Testing
Null Hypothesis: The distribution of outage duration with cause of 'severe weather' and 'intentional attack' are drawn from the same distribution and any observed difference is due to random chance

Alternative Hypothesis: The distributions of outage duration with cause of 'severe weather' and 'intentional attack' are drawn from different population distributions

```py
temp = outages[['CAUSE.CATEGORY', 'OUTAGE.DURATION']].copy()
temp['OUTAGE.DURATION'] = pd.to_numeric(temp['OUTAGE.DURATION']).fillna(0).sort_values()
temp = temp[temp['OUTAGE.DURATION'] < 10]
temp = temp[temp['CAUSE.CATEGORY'].isin(['severe weather', 'intentional attack'])]
temp
```

### TABLE

```py
observed_difference = temp.groupby('CAUSE.CATEGORY')['OUTAGE.DURATION'].mean().diff().iloc[-1]
observed_difference
```
1.9093327843618473

```py
n_repetitions = 500

differences = []
for _ in range(n_repetitions):
    
    with_shuffled = temp.assign(Shuffled_Duration=np.random.permutation(temp['OUTAGE.DURATION']))

    group_means = (
        with_shuffled
        .groupby('CAUSE.CATEGORY')
        .mean()
        .loc[:, 'Shuffled_Duration']
    )
    difference = group_means.diff().iloc[-1]
    
    differences.append(difference)
```

```py
fig = px.histogram(pd.DataFrame(differences), x=0, nbins=50, histnorm='probability', 
                   title='Empirical Distribution of Test Statistic')
fig.add_vline(x=observed_difference, line_color='red')
fig.update_layout(xaxis_title='Difference in Means')
fig.show()
```

### PLOT

```py
p_value_hyp = (observed_difference <= np.array(differences)).mean()
p_value_dep
```
0.0

We can see that our observed statistic falls far from distribution of test statistics, meaning that we reject the null hypothesis in favor of the alternative hypothesis. In other words, the permutation test suggests that the cause of 'severe weather' and 'intentional attack' are not drawn from the same underlying population distribution.
---

## Framing our Prediction Problem

Our goal for the second part of this project is to build off the knowledge we gained from our exploratory data analysis earlier to predict the duration of power outage events in the future. We plan to do this by using present features such as population, type of outage, geographical information, and information about the state to base our linear regression model off of. We believe this is critical for public safety, supporting vulnerable populations, and mitigating economic impact.
---

## Baseline Model

Recall basic science experiments, most science teachers in the US teach the concept of independent and dependent variables, where the independent variable has some kind of effect on the dependent variable. In this prediction problem, our dependent variable is the duration of power outages, and our independent variable is now one of many features, any of which could be a factor that helps us predict outage durations better. These features were split between numerical and categorical variables. For example, CUSTOMERS.AFFECTED is a numerical variable because it is a number that we can perform orders of operations on. By that idea, features like CLIMATE.REGION are categorical variables. There are more complex situations like MONTH, which can be represented in a numerical form, but we characterized those as categorical variables, because we intend to see if different months of the year have an effect on outage durations. A full list of features can be found below.

```py
num_features = ['ANOMALY.LEVEL', 'TOTAL.CUSTOMERS', 'TOTAL.PRICE', 'TOTAL.SALES', 'CUSTOMERS.AFFECTED', 'PCT_WATER_TOT']
cat_features = ['YEAR', 'MONTH', 'CAUSE.CATEGORY', 'CLIMATE.REGION']
predictor = ['OUTAGE.DURATION']
```

By aggregating these features into one large dataset, we obtain our training and testing sets to build our linear regression model. There are a couple concepts here that may need to be explained. First, we separate the dataset into training and testing sets to ensure that our model’s performance can be generalized to data beyond what it was trained on. This split between the training and testing set is arbitrary, but we chose a 80:20 ratio. Linear regression is the core idea of the prediction problem, where we intend to find the line of best fit based on the features we give it. Think of a scatter plot on the XY-plane, but now there’s multiple x’s. This results in a n-dimensional coordinate plane, where n is the number of features, which we cannot physically visualize, but becomes more intuitive as you imagine how the regression changes from 2D to 3D.

Using scikit-learn, a Python-based machine learning library, we transform our data in preparation for creating our basic model for linear regression, which includes only 1 numerical feature - TOTAL.CUSTOMERS - and 2 categorical features - CAUSE.CATEGORY and YEAR. The numerical feature was standardized (see z-scores), and the categorical features one-hot-encoded. That is, each unique value within those features becomes a new feature, where its values are 1 if the value at that original position was the same as the unique value that are these new features, and 0 if it is not. Let’s say the CAUSE.CATEGORY feature had 3 unique values - ‘severe weather’, ‘intentional attack’, and ‘system failure’, once we one-hot-encode the CAUSE.CATEGORY feature, we would have three new features in the format of or similar to ‘is_severe_weather’, ‘is_intentional_attack’, ‘is_system_failure’.

After fitting the model, we obtain our regression results. The R-squared, as known as the coefficient of determination, is a number from 0-1 that explains how much of the variance of the y is explained by our features X. We obtain a R-squared of 0.25 for our training set, and 0.12 for our testing set. The R-squared of the training set being larger than the testing set is not necessarily a bad thing, as that tells us we are not ‘overfitting’ to the training set, that our model can be generalized to data beyond our training data. The root mean square error (RMSE) measures the average deviation between predicted values and actual values, ranging from 0 to infinity, in the units of the y, in this case, OUTAGE.DURATION is in hours. The RMSE of the training set is 61.28, and the RMSE of the testing set is 81.68, which tells us that our model is off by 61 hours in the training and 82 hours in the testing set. Evidently, more work needs to be done to improve the efficacy of the model.

---

## Final Model

In the basic model, we chose the 3 features - TOTAL.CUSTOMERS, CAUSE.CATEGORY, and YEAR - by judgment, but those may have not been the most effective, as shown in the low R-squared of the basic model’s results. To find better combinations of features to predict duration of outages better, we use cross validation, a process by which we divide the training set even more to essentially test the training set before the best training set is compared with the testing set. For more specifics, check out k-fold cross validation.

Using cross validation, we choose between 7 combinations of features
TOTAL.CUSTOMERS, CAUSE.CATEGORY, MONTH
ANOMALY.LEVEL, TOTAL.CUSTOMERS
ANOMALY.LEVEL, TOTAL.CUSTOMERS, TOTAL.PRICE
ANOMALY.LEVEL, TOTAL.CUSTOMERS, TOTAL.PRICE, TOTAL.SALES
ANOMALY.LEVEL, TOTAL.CUSTOMERS, TOTAL.PRICE, TOTAL.SALES, CAUSE.CATEGORY
ANOMALY.LEVEL, TOTAL.CUSTOMERS, TOTAL.PRICE, TOTAL.SALES, CUSTOMERS.AFFECTED, PCT_WATER_TOT, CAUSE.CATEGORY, CLIMATE.REGION
ANOMALY.LEVEL, TOTAL.CUSTOMERS, TOTAL.PRICE, TOTAL.SALES, YEAR, MONTH, CAUSE.CATEGORY, CLIMATE.REGION

After performing k-fold cross validation with 5 folds, we obtain the following table.
| Validation Fold | sample_1  | sample_2  | sample_3  | sample_4  | sample_5  | sample_6  | sample_7  |
|-----------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Fold 1      	| 63.231110 | 63.098073 | 63.134034 | 63.092402 | 71.345084 | 72.005555 | 63.192334 |
| Fold 2      	| 54.167529 | 54.608197 | 54.698204 | 54.874145 | 49.871620 | 45.008102 | 54.350286 |
| Fold 3      	| 80.416242 | 80.640858 | 81.418273 | 81.591553 | 72.733141 | 72.657506 | 80.893216 |
| Fold 4      	| 79.489611 | 79.683935 | 79.820178 | 79.871914 | 68.978334 | 72.351689 | 79.660000 |
| Fold 5      	| 73.959864 | 74.351174 | 74.225691 | 74.259999 | 70.401243 | 64.150105 | 73.927456 |



The average validation RMSE between the 5 folds is below. As you can see, sample_6 has the 
| Sample   | Value 	|
|----------|-----------|
| sample_1 | 70.252871 |
| sample_2 | 70.476447 |
| sample_3 | 70.659276 |
| sample_4 | 70.737376 |
| sample_5 | 66.665885 |
| sample_6 | 65.234591 |
| sample_7 | 70.422838 |

WRITE. As I previously mentioned, linear regression can be n-dimensional. Hyperparameter. Degree 1.

The final model performance reports a R-squared of 0.29 for the training set, and 0.14 for the testing set. The RMSE for the training set is 59.48, and the RMSE for the testing set is 80.49. Overall, using cross validation allowed us to improve our model’s performance by around 120%.

---

## Fairness Analysis

After building our model, it is essential to scrutinize it not only in efficacy but also fairness, that is, does our model perform equally well for individuals between two different groups. This process of rapid iteration and testing is so that we have confidence that our model is representative of the world from which it tries to understand. We decided to divide the groups between the West Climate Region of the United States, one-hot-encoded as True, including Northwest, Southwest, West, amd West North Central, and the East Climate Region, one-hot-encoded as False, including East North Central, Central, South, Southeast, and Northeast. In short, we’re comparing the West Coast and the rest of America.

Null Hypothesis: Our model is fair. The root mean squared error (RMSE) for the west climate region and east climate region are roughly the same, and any differences are due to random chance.  

Alternative Hypothesis: Our model is unfair. The root mean squared error (RMSE) for the east climate region is lower than its precision for the west climate region.

Test Statistics: We chose to compare the difference in root mean squared error between the two models to compare two numerical distributions which describe the effectiveness of our model between the two regions, keeping it as the signed difference in RMSE ensures our ability to discern whether the model is less effective for the east climate region than for the west.

Significance Level: Following permutation testing, we decided to use a significance level of 5%.

The figure above demonstrates the empirical distribution of test statistics over 500 permutations, with the red line representing the observed statistic of -23.89. We ultimately calculate a p–value of 0.16, greater than the p-value of 0.05, which leads us to conclude that we fail to reject the null hypothesis, and have no significant evidence that the root mean square error between the East and West regions are different. Thus, our fairness test suggests that our model is fair, with any differences being attributed to random chance.

---

## Conclusion

WRITE
