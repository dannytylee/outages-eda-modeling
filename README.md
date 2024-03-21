# Power Outage Data Exploration & Prediction

**Name(s)**: Jesse Huang & Danny Lee

**Website Link**: (your website link)

### Imports
```
import pandas as pd
import numpy as np
from pathlib import Path
# newly added
from datetime import datetime, time, timedelta
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, FunctionTransformer, StandardScaler, QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import warnings

import plotly.express as px
pd.options.plotting.backend = 'plotly'

warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')
#from dsc80_utils import * # Feel free to uncomment and use this.
```

**Formula of RMSE**
```
def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred) ** 2))
```