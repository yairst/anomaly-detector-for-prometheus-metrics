import pandas as pd
from prophet import Prophet

class Trainer():
    """Custom class to make forecasts on Prometheus metrics based on Facebook Prophet algorithm

    Parameters
    ----------
    metrics_list : list of full expressions defining the time series, like
        'prometheus_target_interval_length_seconds{quantile="0.99"}' or 
        'rate(prometheus_tsdb_head_chunks_created_total[1m])'. In case ``data_source`` is set to ``"local"``
        the metrics are assumed to be stored in csv format with the expression as the file name, for example:
        'rate(prometheus_tsdb_head_chunks_created_total[1m]).csv'
    
    data_source : {"local", "mysql", "prom"}, default="local"

    url : string, default='./'
        The path or url where the data is, depending on data_source value:
            - If "local", then absolute path.
            - If "mysql" or "prom", then their url.

    start/end : start and end timestamps in '%yyyy-%mm-%ddT%hh:%mm:%ssZ format. if not provided
        set to the last 24 hours if ``data_source`` is ``"prom"`` or ``"mysql"``. Else (``"local"``),
        use all the data.

    step : Any valid frequency for pd.date_range, such as 'D' or 'M', default='5min'.
        The desired resolution of the historic data to be fitted.

    interval_width : float in the range (0,1). The larger - the wider the interval will be, 
        such that prediction-based anomaly detection will have less false alarms (but may miss
        some anomalies). The default high value of 0.99 assumes focusing on precision (and
        less on recall).
        
    periods : int, default=1440
        The number of samples need to be forecasted. the default of 1440 stands for 
        forecasting 24 hours ahead, assuming 1 minute resolution of the historic data (usually
        this is the case with Prometheus metrics). Note that this parameters is depend only
        on the resolution of the historic data and not on the resolution determined in ``freq``.

    freq : Any valid frequency for pd.date_range, such as 'D' or 'M', default='5min'.
        The desired resolution of the forecasted data points.

"""
    def __init__(self, metrics_list, data_source="local", url="./", start=None, end=None, step='5min', 
                interval_width=0.99, periods=1440, freq='5min'):

        self.metrics_list = metrics_list
        self.data_source = data_source
        self.url = url      
        self.start = start
        self.end = end
        self.step = step
        self.interval_width = interval_width
        self.periods = periods
        self.freq = freq

    # TODO: should be prepare data. decide wether to do it on all the list or only
    # one by one. also: need to deal with missing values or a lot of outliers (more than normal)
    def get_data(self):
        df = pd.read_csv('example_wp_log_peyton_manning.csv', parse_dates=['ds'])
        return df

    def fit_predict(self, df, interval_width=0.99, periods=288):
        m = Prophet(interval_width=interval_width)
        m.fit(df)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


if __name__ == 'main':
    t = Trainer([])
    df = t.get_data()
    forecast = t.fit_predict(df,periods=365)
    forecast.to_csv('forecast')