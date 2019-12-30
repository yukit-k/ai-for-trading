#!/usr/bin/env python
# coding: utf-8

# # Sharpe ratio (Solution)

# ## Install packages

# In[ ]:


import sys


# In[ ]:


get_ipython().system('{sys.executable} -m pip install -r requirements.txt')


# In[ ]:


import cvxpy as cvx
import numpy as np
import pandas as pd
import time
import os
import quiz_helper
import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 8)


# ### data bundle

# In[ ]:


import os
import quiz_helper
from zipline.data import bundles


# In[ ]:


os.environ['ZIPLINE_ROOT'] = os.path.join(os.getcwd(), '..', '..','data','module_4_quizzes_eod')
ingest_func = bundles.csvdir.csvdir_equities(['daily'], quiz_helper.EOD_BUNDLE_NAME)
bundles.register(quiz_helper.EOD_BUNDLE_NAME, ingest_func)
print('Data Registered')


# ### Build pipeline engine

# In[ ]:


from zipline.pipeline import Pipeline
from zipline.pipeline.factors import AverageDollarVolume
from zipline.utils.calendars import get_calendar

universe = AverageDollarVolume(window_length=120).top(500) 
trading_calendar = get_calendar('NYSE') 
bundle_data = bundles.load(quiz_helper.EOD_BUNDLE_NAME)
engine = quiz_helper.build_pipeline_engine(bundle_data, trading_calendar)


# ### View Data¶
# With the pipeline engine built, let's get the stocks at the end of the period in the universe we're using. We'll use these tickers to generate the returns data for the our risk model.

# In[ ]:


universe_end_date = pd.Timestamp('2016-01-05', tz='UTC')

universe_tickers = engine    .run_pipeline(
        Pipeline(screen=universe),
        universe_end_date,
        universe_end_date)\
    .index.get_level_values(1)\
    .values.tolist()
    
universe_tickers


# # Get Returns data

# In[ ]:


from zipline.data.data_portal import DataPortal

data_portal = DataPortal(
    bundle_data.asset_finder,
    trading_calendar=trading_calendar,
    first_trading_day=bundle_data.equity_daily_bar_reader.first_trading_day,
    equity_minute_reader=None,
    equity_daily_reader=bundle_data.equity_daily_bar_reader,
    adjustment_reader=bundle_data.adjustment_reader)


# ## Get pricing data helper function

# In[ ]:


def get_pricing(data_portal, trading_calendar, assets, start_date, end_date, field='close'):
    end_dt = pd.Timestamp(end_date.strftime('%Y-%m-%d'), tz='UTC', offset='C')
    start_dt = pd.Timestamp(start_date.strftime('%Y-%m-%d'), tz='UTC', offset='C')

    end_loc = trading_calendar.closes.index.get_loc(end_dt)
    start_loc = trading_calendar.closes.index.get_loc(start_dt)

    return data_portal.get_history_window(
        assets=assets,
        end_dt=end_dt,
        bar_count=end_loc - start_loc,
        frequency='1d',
        field=field,
        data_frequency='daily')


# ## get pricing data into a dataframe

# In[ ]:


returns_df =     get_pricing(
        data_portal,
        trading_calendar,
        universe_tickers,
        universe_end_date - pd.DateOffset(years=5),
        universe_end_date)\
    .pct_change()[1:].fillna(0) #convert prices into returns

returns_df


# ## Sector data helper function
# We'll create an object for you, which defines a sector for each stock.  The sectors are represented by integers.  We inherit from the Classifier class.  [Documentation for Classifier](https://www.quantopian.com/posts/pipeline-classifiers-are-here), and the [source code for Classifier](https://github.com/quantopian/zipline/blob/master/zipline/pipeline/classifiers/classifier.py)

# In[ ]:


from zipline.pipeline.classifiers import Classifier
from zipline.utils.numpy_utils import int64_dtype
class Sector(Classifier):
    dtype = int64_dtype
    window_length = 0
    inputs = ()
    missing_value = -1

    def __init__(self):
        self.data = np.load('../../data/project_4_sector/data.npy')

    def _compute(self, arrays, dates, assets, mask):
        return np.where(
            mask,
            self.data[assets],
            self.missing_value,
        )


# In[ ]:


sector = Sector()


# ## We'll use 2 years of data to calculate the factor

# **Note:** Going back 2 years falls on a day when the market is closed. Pipeline package doesn't handle start or end dates that don't fall on days when the market is open. To fix this, we went back 2 extra days to fall on the next day when the market is open.

# In[ ]:


factor_start_date = universe_end_date - pd.DateOffset(years=2, days=2)
factor_start_date


# ## Create smoothed momentum factor

# In[ ]:


from zipline.pipeline.factors import Returns
from zipline.pipeline.factors import SimpleMovingAverage


# create a pipeline called p
p = Pipeline(screen=universe)
# create a factor of one year returns, deman by sector, then rank
factor = (
    Returns(window_length=252, mask=universe).
    demean(groupby=Sector()). #we use the custom Sector class that we reviewed earlier
    rank().
    zscore()
)


# Use this factor as input into SimpleMovingAverage, with a window length of 5
# Also rank and zscore (don't need to de-mean by sector, s)
factor_smoothed = (
    SimpleMovingAverage(inputs=[factor], window_length=5).
    rank().
    zscore()
)

# add the unsmoothed factor to the pipeline
p.add(factor, 'Momentum_Factor')
# add the smoothed factor to the pipeline too
p.add(factor_smoothed, 'Smoothed_Momentum_Factor')


# ## visualize the pipeline
# 
# Note that if the image is difficult to read in the notebook, right-click and view the image in a separate tab.

# In[ ]:


p.show_graph(format='png')


# ## run pipeline and view the factor data

# In[ ]:


df = engine.run_pipeline(p, factor_start_date, universe_end_date)


# In[ ]:


df.head()


# ## Evaluate Factors
# 
# We'll go over some tools that we can use to evaluate alpha factors.  To do so, we'll use the [alphalens library](https://github.com/quantopian/alphalens)
# 

# ## Import alphalens

# In[ ]:


import alphalens as al


# ## Get price data
# 
# Note, we already got the price data and converted it to returns, which we used to calculate a factor.  We'll retrieve the price data again, but won't convert these to returns.  This is because we'll use alphalens functions that take their input as prices and not returns.
# 
# ## Define the list of assets
# Just to make sure we get the prices for the stocks that have factor values, we'll get the list of assets, which may be a subset of the original universe

# In[ ]:


# get list of stocks in our portfolio (tickers that identify each stock)
assets = df.index.levels[1].values.tolist()
print(f"stock universe number of stocks {len(universe_tickers)}, and number of stocks for which we have factor values {len(assets)}")


# In[ ]:


factor_start_date


# In[ ]:


pricing = get_pricing(
        data_portal,
        trading_calendar,
        assets, #notice that we used assets instead of universe_tickers; in this example, they're the same
        factor_start_date, # notice we're using the same start and end dates for when we calculated the factor
        universe_end_date)


# ## Prepare data for use in alphalens
# 

# In[ ]:


factor_names = df.columns
print(f"The factor names are {factor_names}")
factor_data = {}
for factor_name in factor_names:
    print("Formatting factor data for: " + factor_name)
    # get clean factor and forward returns for each factor
    factor_data[factor_name] = al.utils.get_clean_factor_and_forward_returns(
        factor=df[factor_name],
        prices=pricing,
        periods=[1])


# ### factor returns

# In[ ]:


ls_factor_return = []

for i, factor_name in enumerate(factor_names):
    # use alphalens function "factor_returns" to calculate factor returns
    factor_return = al.performance.factor_returns(factor_data[factor_name])
    factor_return.columns = [factor_name]
    ls_factor_return.append(factor_return)


# # Quiz 1: Sharpe ratio
# 
# Generally, a sharpe ratio of 1 or higher indicates a better factor than one with a lower Sharpe ratio.  In other words, the returns that would have been accrued by a portfolio that was weighted according to the alpha factor would have had an annualized return that is greater or equal to its annualized volatility
# 
# Recall that the annualize the sharpe ratio (from daily to annual), multiply by $ \sqrt[2]{252} $

# In[ ]:


def sharpe_ratio(df, frequency="daily"):

    if frequency == "daily":
        # TODO: daily to annual conversion
        annualization_factor = np.sqrt(252)
    elif frequency == "monthly":
        #TODO: monthly to annual conversion
        annualization_factor = np.sqrt(12)
    else:
        # TODO: no conversion
        annualization_factor = 1
        
    #TODO: calculate the sharpe ratio and store it in a dataframe.
    # name the column 'Sharpe Ratio'.  
    # round the numbers to 2 decimal places
    df_sharpe = pd.DataFrame(data=annualization_factor*df.mean()/df.std(),
     columns=['Sharpe Ratio']).round(2)
    
    return df_sharpe


# ## Quiz 2
# 
# Compare the sharpe ratio of the unsmoothed versus smoothed version of the factors.

# ## Answer 2

# In[ ]:


# TODO: calculate sharpe ratio on the unsmooothed factor
sharpe_ratio(ls_factor_return[0])


# In[ ]:


# TODO: calculate sharpe ratio on the smooothed factor
sharpe_ratio(ls_factor_return[1])


# ## Answer 2 continued
# The smoothed factor has a slightly lower sharpe ratio in this example.

# In[ ]:




