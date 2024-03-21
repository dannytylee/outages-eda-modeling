# Power Outage Data Exploration & Prediction

**By Jesse Huang and Danny Lee**

## Project Overview

This project aims to investigate the factors that lead to power outage events with the hopes of gaining valuable insights into why these events occur and the ways in which outages can be mitigated. If you are interested in learning more about the dataset we based our project on, more information can be found [here](https://www.sciencedirect.com/science/article/pii/S2352340918307182). The power outage data exploration and prediction on website is for DSC 80 at UCSD.

# Introduction

the so what?


Our goal for the first part of this project is to investigate underlying climate trends over time and geographical patterns to develop a better understanding about where power outages are more likely to occur and their magnitudes. To be more specific, how do variables about the number of customers affected by historical power outage events, changing climate, and geographical position of states contribute to the risk of potential power outages and their duration in the present?

---

## Data Cleaning and Exploratory Data Analysis
```py
messy_outages = pd.read_excel('outage.xlsx')
# print(messy_outages.head().to_markdown(index=False))
```

<iframe
  src="assets/univariate_1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>




<iframe
  src="assets/univariate_2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>




<iframe
  src="assets/bivariate_1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>




<iframe
  src="assets/bivariate_2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


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

A pivot table is show below. According to the pivot table, it is apparent that the southwest and northwest climate regions were the least affected by power outages, for many of their columns are filled with zeros to represent 0 hours of power outage during those years. On the other hand, when look at the columns for the south climate region, all the columns are filled with large values. This indicates that the south climate region was one of the most heavily affected by power outages. These findings are consistent with the boxplot seen above.

```py
pd.pivot_table(outages, values='CUSTOMERS.AFFECTED', index='CLIMATE.REGION', 
               columns='YEAR', aggfunc='mean', fill_value=0)
```

### TABLE

Additional research shows, "Tropical cyclones, storms with high winds that originate over tropical oceans, make a power outage 14 times more likely. And a tropical cyclone with heavy precipitation on a hot day—like the hurricanes that each fall hit the Gulf Coast? They make power outages 52 times more likely" (https://deohs.washington.edu/hsm-blog/these-four-regions-us-are-hardest-hit-power-outages).

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
```py
num_features = ['ANOMALY.LEVEL', 'TOTAL.CUSTOMERS', 'TOTAL.PRICE', 'TOTAL.SALES', 'CUSTOMERS.AFFECTED', 'PCT_WATER_TOT']
cat_features = ['YEAR', 'MONTH', 'CAUSE.CATEGORY', 'CLIMATE.REGION']
predictor = ['OUTAGE.DURATION']
```

```py
model_outages = outages[cat_features + num_features + predictor].copy()
model_outages = model_outages.dropna()

true_predictions = model_outages['OUTAGE.DURATION']
model_outages = model_outages.drop(columns = ['OUTAGE.DURATION'])
```

```py
X = model_outages
y = true_predictions

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
```

```py
preproc_base = ColumnTransformer(
    transformers=[
        ('standardize', StandardScaler(), ['TOTAL.CUSTOMERS']),
        ('categorical_cols', OneHotEncoder(drop='first'), ['CAUSE.CATEGORY', 'YEAR'])
    ],
    remainder='drop')
```

```py
pl_base = Pipeline([
    ('preprocessor', preproc_base), 
    ('lin-reg', LinearRegression())
])
```

```py
pl_base.fit(X_train, y_train)
```
Pipeline(steps=[('preprocessor',
                 ColumnTransformer(transformers=[('standardize',
                                                  StandardScaler(),
                                                  ['TOTAL.CUSTOMERS']),
                                                 ('categorical_cols',
                                                  OneHotEncoder(drop='first'),
                                                  ['CAUSE.CATEGORY',
                                                   'YEAR'])])),
                ('lin-reg', LinearRegression())])

```py
pl_base.score(X, y)
```
0.21480899432369982


```py
train_predicted = pl_base.predict(X_train)
rmse(y_train, train_predicted)
```
61.27683537688739

```py
pl_base.score(X_test, y_test)
```
0.1165678626471397


```py
test_predicted = pl_base.predict(X_test)
rmse(y_test, test_predicted)
```
81.68316868678828

---

## Final Model
```py
pipes = {
    'sample_1': make_pipeline(
        make_column_transformer(
            (FunctionTransformer(lambda x: x), ['TOTAL.CUSTOMERS']), # don't need to standardize because perf doesn't change
            (OneHotEncoder(drop='first', handle_unknown='ignore'), ['CAUSE.CATEGORY', 'MONTH']),
        ),
        LinearRegression(),
    ),
    'sample_2': make_pipeline(
        make_column_transformer(
            (FunctionTransformer(lambda x: x), ['ANOMALY.LEVEL', 'TOTAL.CUSTOMERS']),
        ),
        LinearRegression(),
    ),
    'sample_3': make_pipeline(
        make_column_transformer(
            (FunctionTransformer(lambda x: x), ['ANOMALY.LEVEL', 'TOTAL.CUSTOMERS', 'TOTAL.PRICE']),
        ),
        LinearRegression(),
    ),
    'sample_4': make_pipeline(
        make_column_transformer(
            (FunctionTransformer(lambda x: x), ['ANOMALY.LEVEL', 'TOTAL.CUSTOMERS', 'TOTAL.PRICE', 'TOTAL.SALES']),
        ),
        LinearRegression(),
    ),
    'sample_5': make_pipeline(
        make_column_transformer(
            (FunctionTransformer(lambda x: x), ['ANOMALY.LEVEL', 'TOTAL.CUSTOMERS', 'TOTAL.PRICE', 'TOTAL.SALES']),
            (OneHotEncoder(drop='first', handle_unknown='ignore'), ['CAUSE.CATEGORY']),
        ),
        LinearRegression(),
    ),
    'sample_6': make_pipeline(
        make_column_transformer(
            (FunctionTransformer(lambda x: x), ['ANOMALY.LEVEL', 'TOTAL.CUSTOMERS', 'TOTAL.PRICE', 'TOTAL.SALES', 'CUSTOMERS.AFFECTED', 'PCT_WATER_TOT']),
            (OneHotEncoder(drop='first', handle_unknown='ignore'), ['CAUSE.CATEGORY', 'CLIMATE.REGION']),
        ),
        LinearRegression(),
    ),
    'sample_7': make_pipeline(
        make_column_transformer(
            (FunctionTransformer(lambda x: x), ['ANOMALY.LEVEL', 'TOTAL.CUSTOMERS', 'TOTAL.PRICE', 'TOTAL.SALES']),
            (OneHotEncoder(drop='first', handle_unknown='ignore'), ['YEAR', 'MONTH', 'CAUSE.CATEGORY', 'CLIMATE.REGION']),
        ),
        LinearRegression(),
    ),
}
```

```py
pipe_df = pd.DataFrame()

for pipe in pipes:
    errs = cross_val_score(pipes[pipe], X_train, y_train,
                           cv=5, scoring='neg_root_mean_squared_error')
    pipe_df[pipe] = -errs
    
pipe_df.index = [f'Fold {i}' for i in range(1, 6)]
pipe_df.index.name = 'Validation Fold'
```

```py
pipe_df
```
### TABLE

```py
pipe_df.mean()
```
### OUTPUT

```py
'Lowest Average Validation RMSE: ' + str(pipe_df.mean().idxmin())
```
'Lowest Average Validation RMSE: sample_6'


```py
errs_df = pd.DataFrame()

for d in range(1, 11):
    pl = make_pipeline(PolynomialFeatures(d), LinearRegression())
    X = X_train[['ANOMALY.LEVEL', 'TOTAL.CUSTOMERS', 'TOTAL.PRICE', 
                 'TOTAL.SALES', 'CUSTOMERS.AFFECTED', 'PCT_WATER_TOT']]
    y = y_train
   
    errs = cross_val_score(pl, X, y, cv=5, scoring='neg_root_mean_squared_error') # 5-Fold Cross Validation
    errs_df[f'Deg {d}'] = -errs # errs = negative RMSE
    
errs_df.index = [f'Fold {i}' for i in range(1, 6)]
errs_df.index.name = 'Validation Fold'
```

```py
errs_df
```
### TABLE

```py
errs_df.mean(axis=0)
```
Deg 1         67.860227
Deg 2         89.791544
Deg 3        120.551792
Deg 4        822.983713
Deg 5       1861.478560
Deg 6       4099.196760
Deg 7      10326.678933
Deg 8     117226.136721
Deg 9     156230.284096
Deg 10     85205.598474
dtype: float64

```py
'Lowest Testing Error Degree: ' + str(errs_df.mean(axis=0).idxmin())
```
'Lowest Testing Error Degree: Deg 1'

```py
preproc = ColumnTransformer(
    transformers=[
        ('quantile', QuantileTransformer(n_quantiles = 100), ['TOTAL.SALES', 'CUSTOMERS.AFFECTED', 
                                                              'TOTAL.CUSTOMERS', 'TOTAL.PRICE']),
        
        ('standardize', StandardScaler(), ['ANOMALY.LEVEL', 'PCT_WATER_TOT']),
        
        ('categorical_cols', OneHotEncoder(drop='first', handle_unknown='ignore'), ['CAUSE.CATEGORY', 'CLIMATE.REGION']),
        
        ('poly', PolynomialFeatures(degree = 1), ['ANOMALY.LEVEL', 'TOTAL.CUSTOMERS', 
                                                  'TOTAL.PRICE', 'TOTAL.SALES', 
                                                  'CUSTOMERS.AFFECTED', 'PCT_WATER_TOT']),
    ],
    remainder='passthrough')
```
Why QuantileTransformer for 'TOTAL.SALES' and 'TOTAL.CUSTOMERS'?

Because the numbers are so large on both features, QuantileTransformer can be useful for reducing the impact of outliers, especially linear models. Additionally, we wanted all features on a similar scale so that the coefficients are more interpretable.


```py
pl = Pipeline([
    ('preprocessor', preproc), 
    ('lin-reg', LinearRegression())
])
```

```py
pl.fit(X_train, y_train)
```
Pipeline(steps=[('preprocessor',
                 ColumnTransformer(remainder='passthrough',
                                   transformers=[('quantile',
                                                  QuantileTransformer(n_quantiles=100),
                                                  ['TOTAL.SALES',
                                                   'CUSTOMERS.AFFECTED',
                                                   'TOTAL.CUSTOMERS',
                                                   'TOTAL.PRICE']),
                                                 ('standardize',
                                                  StandardScaler(),
                                                  ['ANOMALY.LEVEL',
                                                   'PCT_WATER_TOT']),
                                                 ('categorical_cols',
                                                  OneHotEncoder(drop='first',
                                                                handle_unknown='ignore'),
                                                  ['CAUSE.CATEGORY',
                                                   'CLIMATE.REGION']),
                                                 ('poly',
                                                  PolynomialFeatures(degree=1),
                                                  ['ANOMALY.LEVEL',
                                                   'TOTAL.CUSTOMERS',
                                                   'TOTAL.PRICE', 'TOTAL.SALES',
                                                   'CUSTOMERS.AFFECTED',
                                                   'PCT_WATER_TOT'])])),
                ('lin-reg', LinearRegression())])

**Baseline Model Performance**

```py
pl_base.score(X_train, y_train)
```
0.2511739088911349

```py
train_predicted = pl_base.predict(X_train)
rmse(y_train, train_predicted)
```
61.27683537688739

```py
pl_base.score(X_test, y_test)
```
0.1165678626471397

```py
test_predicted = pl_base.predict(X_test)
rmse(y_test, test_predicted)
```
81.68316868678828

**Final Model Performance**

```py
pl.score(X_train, y_train)
```
0.2945144443242381

```py
rmse(y_train, pl.predict(X_train))
```
59.47711676851063

```py
pl.score(X_test, y_test)
```
0.14209170863605658

```py
rmse(y_test, pl.predict(X_test))
```
80.49453804992676
---

## Fairness Analysis
Let our two climate region groups be

<b>West Climate Region of the United States<b> (True):
- Northwest
- Southwest
- West
- West North Central

<b>East Climate Region of the United States<b> (False):
- East North Central
- Central
- South
- Southeast
- Northeast

Null Hypothesis: Our model is fair. Its RMSE for the west climate region and east climate region are roughly the same, and any differences are due to random chance.

Alternative Hypothesis: Our model is unfair. Its RMSE for the east climate region is lower than its precision for the west climate region.

```py
def model_RMSE(df, pipeline):
    features = ['ANOMALY.LEVEL', 'TOTAL.CUSTOMERS', 'TOTAL.PRICE', 'OUTAGE.DURATION',
                'TOTAL.SALES', 'CUSTOMERS.AFFECTED', 'PCT_WATER_TOT', 'CAUSE.CATEGORY', 'CLIMATE.REGION']
    
    df = df[features].dropna()
    X = df.drop(columns=['OUTAGE.DURATION'])
    y = df['OUTAGE.DURATION'].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    rmse_value = rmse(y_pred, y_test)
    return rmse_value
```

```py
west = ['Northwest' ,'Southwest', 'West', 'West North Central']
east = ['East North Central', 'Central', 'South', 'Southeast', 'Northeast']
fairness_df = outages.copy()
fairness_df['CLIMATE.REGION'] = fairness_df['CLIMATE.REGION'].transform(lambda row: True if row in west else False)
```

```py
obser_stat = fairness_df.groupby('CLIMATE.REGION').agg(model_RMSE, pl)['OBS']
obser_stat = obser_stat.diff().iloc[-1]
obser_stat
```
-23.88935579176754

```py
n_repetitions = 500
differences = []

for _ in range(n_repetitions):
    
    with_shuffled = fairness_df.assign(Shuffled_Col=np.random.permutation(fairness_df['CLIMATE.REGION']))
    
    group_means = (
        with_shuffled
        .groupby('Shuffled_Col')
        .agg(model_RMSE, pl)
        ['OBS']
    )
    
    difference = group_means.diff().iloc[-1]
    differences.append(difference)
```

```py
fig = px.histogram(pd.DataFrame(differences), x=0, nbins=50, histnorm='probability', 
                   title='Empirical Distribution of Test Statistic')
fig.add_vline(x=obser_stat, line_color='red')
fig.show()
```
### PLOT

```py
p_value_fair = (obser_stat >= np.array(differences)).mean()
p_value_fair
```
0.16

```py
In conclusion, we fail to reject null hypothesis because 0.16 is greater than the 0.05 pre-defined cutoff value. Thus, the permutation test suggests that our model is fair, and any differences are due to random chance.
```
