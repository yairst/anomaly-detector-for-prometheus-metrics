import pandas as pd
from utils import query_to_df, get_metrics_from_file
from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime
from arguments import get_params

# TODO: change get_forecast_slice so it can read from metric_object
def get_forecast_slice(forecast, start_time="10min", end_time="now", step='1min'):
    # convert start and end times to datetime objects:
    start_time = parse_datetime(start_time)
    end_time = parse_datetime(end_time)

    # read forecast
    pred = pd.read_csv('forecasts/' + forecast + '.csv',parse_dates=['ds'])
    
    # first slice: little wider in both sides to deal with the fact that the period time of the metric ("step")
    # is always not higher than the period time of the forecast ("freq"):
    freq = pred.iloc[-1,0] - pred.iloc[-2,0]
    pred = pred[(pred['ds'] >= start_time - freq) & (pred['ds'] <= end_time + freq)]

    # resample to the period time of the metric (because the use of bfill() - need to set "ds" as index):
    pred = pred.set_index('ds').resample(step).bfill()

    # second slice: to get identical timestamp to the metric (and reset_index to get identical index to metric):
    pred = pred[start_time:end_time].reset_index()

    return pred

def is_anomaly(actual, pred, anomaly_type="upper"):
    """ anomaly_type: string, default: "upper".
            the type of anomaly to be detected: can be one of "upper", "lower" or "both"
        return: -1 for lower anomaly, 1 for upper anomaly and 0 for no anomaly
    """
    n_samples = len(actual)
    if ((anomaly_type == "upper" or anomaly_type == "both") and
        (actual.y > pred.yhat_upper).sum() >= (n_samples / 2)):
        return 1
    if ((anomaly_type == "lower" or anomaly_type == "both") and
        (actual.y < pred.yhat_lower).sum() >= (n_samples / 2)):
        return -1
    return 0


if __name__ == '__main__':

    # get args
    args = get_params()
    
    # connect to prometheus
    prom = PrometheusConnect(url=args.url, disable_ssl=True)

    # get metrics list from file
    metrics = get_metrics_from_file('test_metrics.txt')

    for metric in metrics:
        # TODO: add support in multi-time-series metrics and looping on MetricList object
        # read forecast and cut only the relevant timestamps
        pred = get_forecast_slice(metric,
                                start_time=args.start_time,
                                end_time=args.end_time,
                                step=args.step)

        # TODO: change to get metric object
        # read last 10 samples from prometheus
        actual = query_to_df(prom, metric, args.start_time, args.end_time, args.step)

        # check for anomaly:
        anomaly = is_anomaly(actual, pred)
        print("metric: {} anomaly state is {}.".format(metric, anomaly))
