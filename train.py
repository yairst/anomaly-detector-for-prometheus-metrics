import pandas as pd
from prophet import Prophet
from matplotlib import pyplot as plt
from prometheus_api_client import PrometheusConnect
from arguments import get_params
from utils import query_to_df


def fit_predict(df, interval_width=0.99, periods=1440, freq='5min', season=None):
    m = Prophet(interval_width=interval_width)
    if season is not None:
        for name, val in season.items():
            m.add_seasonality(name=name, period=val[0], fourier_order=val[1])
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return m, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


if __name__ == '__main__':

    # get args
    args = get_params()

    # connect to prometheus
    prom = PrometheusConnect(url =args.url, disable_ssl=True)

    # get random 3 metrics
    # metrics = np.random.choice(prom.all_metrics(),size=3,replace=False) 

    # get queries list from file
    with open('test_queries.txt') as f:
        queries = f.read().splitlines()

    # check for special seasonalities:
    season=None
    if args.seasonality_vals is not None:
        season = {i:[j, k] for i, j, k in zip(args.seasonality_names, args.seasonality_vals,
                                                args.seasonality_fourier)}

    for q in queries:
        df = query_to_df(prom, q, args.start_time, args.end_time, args.step)
        m, forecast = fit_predict(df, periods=args.periods, freq=args.freq, season=season)
        forecast_only_future = forecast[forecast['ds'] > args.end_time]
        forecast_only_future.to_csv('forecasts/' + q + '.csv', index=False)
        if args.debug:
            m.plot(forecast)
            plt.show()

    # df = pd.read_csv('example_wp_log_peyton_manning.csv', parse_dates=['ds'])
    # m, forecast = fit_predict(df, periods=365, freq='D')
    # forecast.to_csv('forecast.csv')
    # m.plot(forecast)
    # plt.show()
    