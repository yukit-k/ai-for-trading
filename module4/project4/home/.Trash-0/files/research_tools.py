import pandas as pd
from zipline.data import bundles
from zipline.data.data_portal import DataPortal
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.utils.calendars import get_calendar
from zipline.assets._assets import Equity


# Set-Up Pricing Data Access
trading_calendar = get_calendar('NYSE')
bundle = 'quandl'
bundle_data = bundles.load(bundle)

pipeline_loader = USEquityPricingLoader(
    bundle_data.equity_daily_bar_reader,
    bundle_data.adjustment_reader,
)

# Set-Up Pipeline Engine
engine = SimplePipelineEngine(
    get_loader=lambda column: pipeline_loader,
    calendar=trading_calendar.all_sessions,
    asset_finder=bundle_data.asset_finder,
)

def run_pipeline(pipeline, start_date, end_date):
    return engine.run_pipeline(
        pipeline,
        pd.Timestamp(start_date, tz='utc'),
        pd.Timestamp(end_date, tz='utc')
    )

data = DataPortal(
    bundle_data.asset_finder,
    trading_calendar=trading_calendar,
    first_trading_day=bundle_data.equity_daily_bar_reader.first_trading_day,
    equity_minute_reader=None,
    equity_daily_reader=bundle_data.equity_daily_bar_reader,
    adjustment_reader=bundle_data.adjustment_reader,
)

def get_symbols(tickers, as_of_date=None):
    if (type(tickers)==str):
        return bundle_data.asset_finder.lookup_symbols(
            [tickers], as_of_date=as_of_date)
    else:
        if(type(tickers[0]==Equity)):
           return tickers
        else:
           return bundle_data.asset_finder.lookup_symbols(
               tickers, as_of_date=as_of_date)

def get_pricing(tickers, start_date, end_date, field='close'):

    end_dt = pd.Timestamp(end_date, tz='UTC', offset='C')
    start_dt = pd.Timestamp(start_date, tz='UTC', offset='C')
        
    symbols = get_symbols(tickers, as_of_date=end_dt)
    
    end_loc = trading_calendar.closes.index.get_loc(end_dt)
    start_loc = trading_calendar.closes.index.get_loc(start_dt)    
    
    dat = data.get_history_window(
        assets=symbols,
        end_dt=end_dt,
        bar_count=end_loc - start_loc,
        frequency='1d',
        field=field,
        data_frequency='daily'
    )

    return dat
