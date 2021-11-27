import pandas as pd
from utils import query_to_df
from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime

# start simple:
# 1. read one forecast csv file
# 2. read 10 samples of real metric values in the forecast horizon
# 3. compare those samples to yhats of the prediction
# 4. do majority vote to clssify if this is anomaly


if __name__ == '__main__':
    
    # start and end time for the last 10 samples
    start_time = parse_datetime("2021-11-21 12:15:00")
    end_time = parse_datetime("2021-11-21 12:17:15")
    step = '15s'

    # read forecast and cut only the relevant timestamps
    pred = pd.read_csv('forecasts/go_memstats_alloc_bytes.csv',parse_dates=['ds'])
    td = pred.iloc[-1,0] - pred.iloc[-2,0]
    pred = pred[(pred['ds'] >= start_time - td) & (pred['ds'] <= end_time + td)]
    pred = pred.set_index('ds').resample(step).bfill().reset_index()
    pred = pred[(pred['ds'] >= start_time) & (pred['ds'] <= end_time)].reset_index()

    # connect to prometheus
    prom = PrometheusConnect(url ='http://localhost:9090', disable_ssl=True)

    # read last 10 samples from prometheus
    actual = query_to_df(prom, 'go_memstats_alloc_bytes', start_time, end_time, step)

    # check for anomaly:
    is_anomaly = int(((actual.y > pred.yhat_upper) | (actual.y < pred.yhat_lower)).sum() >= 5)

