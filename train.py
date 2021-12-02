import pandas as pd
from prophet import Prophet
from matplotlib import pyplot as plt
from prometheus_api_client import PrometheusConnect
from arguments import get_params
from utils import query_to_df
from feature_engine.outliers import Winsorizer


def fit_predict(df, interval_width=0.99, periods=1440, freq='5min', season=None):
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

    # get queries list from file
    with open('test_queries.txt') as f:
        queries = f.read().splitlines()

    # check for special seasonalities:
    season=None
    if args.seasonality_vals is not None:
        season = {}
        season['names'] = args.seasonality_names
        season['vals'] = args.seasonality_vals
        season['fourier'] = args.seasonality_fourier

    for q in queries:
        if q[0] != "#":
            df = query_to_df(prom, q, args.start_time, args.end_time, args.step)
            if args.winsorizing:
                wins = Winsorizer(capping_method='iqr',tail='both', fold=1.5)
                df['y'] = wins.fit_transform(pd.DataFrame(df['y']))
            m, forecast = fit_predict(df, periods=args.periods, freq=args.freq, season=season)
            forecast_only_future = forecast[forecast['ds'] > args.end_time]
            forecast_only_future.to_csv('forecasts/' + q + '.csv', index=False)
            if args.debug:
                m.plot(forecast)
                plt.show()
    