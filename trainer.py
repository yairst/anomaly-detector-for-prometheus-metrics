import pandas as pd
from prophet import Prophet
from matplotlib import pyplot as plt
from prometheus_api_client import PrometheusConnect
from arguments import get_params
from utils import get_metrics_from_file, get_metric_list
from feature_engine.outliers import Winsorizer
import pickle
import platform


def fit_predict(df, interval_width=0.99, periods=288, freq='5min', season=None):
    """
    Returns a tuple of `(m, forecast)` where `m` is a Prophet model fitted to `df` and `forecast`
    is 4 columns data frame with forecasting data, including history. The columns are:
    - 'ds': timestamps
    - 'yhat': the predicted values for the time-series
    - 'yhat_lower' and 'yhat_upper': uncertainty intervals
    `periods`: Int number of periods to forecast forward.
    `freq`: Any valid frequency for pd.date_range, such as 'D' or 'M'. The time period of
    the forecasted data
    `season`: optional parameter for adding additional seasonalities to the default ones.
    A dict with keys: 'names', 'vals' and 'fourier' holding equal length lists with the names of
    the seasonalities, their periods (float with units of days) and the number of
    fourier coefficients  accordingly. 
    """
    m = Prophet(interval_width=interval_width)
    if season is not None:
        for name, val, f_order in zip(season['names'], season['vals'], season['fourier']):
            m.add_seasonality(name=name, period=val, fourier_order=f_order)

    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return m, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


if __name__ == '__main__':

    # get args
    args = get_params()

    # connect to prometheus
    prom = PrometheusConnect(url =args.url, disable_ssl=True)

    # get metrics list from file
    metrics = get_metrics_from_file('test_metrics.txt')

    # check for special seasonalities:
    season=None
    if args.seasonality_vals is not None:
        season = {}
        season['names'] = args.seasonality_names
        season['vals'] = args.seasonality_vals
        season['fourier'] = args.seasonality_fourier

    for metric in metrics:
        metrics_list = get_metric_list(prom, metric, args.start_time, args.end_time, args.step)
        forecasted_metrics_list = []
        for metric_obj in metrics_list:
            df = metric_obj.metric_values
            if args.winsorizing:
                wins = Winsorizer(capping_method='iqr',tail='both', fold=1.5)
                df['y'] = wins.fit_transform(pd.DataFrame(df['y']))
            m, forecast = fit_predict(df, periods=args.periods, freq=args.freq, season=season)
            forecast_only_future = forecast[forecast['ds'] > args.end_time]
            metric_obj.metric_values = forecast_only_future
            forecasted_metrics_list.append(metric_obj)
            if args.debug:
                m.plot(forecast)
                plt.title(metric_obj.metric_name + str(metric_obj.label_config))
                plt.show()
        if platform.system() == 'Windows':
            metric = metric.replace(":","$")
        with open('forecasts/' + metric + '.pkl', 'wb') as outp:
            pickle.dump(forecasted_metrics_list, outp)

    