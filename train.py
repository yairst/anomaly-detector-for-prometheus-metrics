import pandas as pd
from prophet import Prophet
from matplotlib import pyplot as plt
from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime
from arguments import get_params

def query_to_df(prom, query, start_time, end_time, step):
    start_time = parse_datetime(start_time)
    end_time = parse_datetime(end_time)
    query_range = prom.custom_query_range(query ,
                                start_time=start_time,
                                end_time=end_time,
                                step=step)
    df = pd.DataFrame(query_range[0]['values'],columns=['ds','y'])
    df['ds'] = pd.to_datetime(df['ds'],unit='s').astype('datetime64[ns, Asia/Jerusalem]').dt.tz_localize(None)
    df['y'] = df['y'].astype(float)
    return df

def fit_predict(df, interval_width=0.99, periods=1440, freq='5min'):
    m = Prophet(interval_width=interval_width)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return m, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


if __name__ == '__main__':

    # get args
    args = get_params()

    # connect to prometheus
    prom = PrometheusConnect(url ="http://localhost:9090", disable_ssl=True)

    # get random 3 metrics
    # metrics = np.random.choice(prom.all_metrics(),size=3,replace=False) 

    # get queries list from file
    with open('test_queries.txt') as f:
        queries = f.read().splitlines()

    for q in queries:
        df = query_to_df(prom, q, args.start_time, args.end_time, args.step)
        m, forecast = fit_predict(df, periods=300, freq='5s')
        forecast.to_csv('forecasts/' + q + '.csv')
        if args.debug:
            m.plot(forecast)
            plt.show()

    # df = pd.read_csv('example_wp_log_peyton_manning.csv', parse_dates=['ds'])
    # m, forecast = fit_predict(df, periods=365, freq='D')
    # forecast.to_csv('forecast.csv')
    # m.plot(forecast)
    # plt.show()
    