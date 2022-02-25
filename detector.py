import pandas as pd
from utils import get_metrics_from_file, get_metric_list, get_full_metric_name
from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime
from arguments import get_params
import pickle
import platform


def get_slice_from_ts(ts, start_time="10min", end_time="now", step='1min'):
    """
    Returns resampled slice from time-series data frame.

    Data frame should have column named 'ds' of type Datetime

    The slice is resampled by the period time defined in `step`, where `step` should be not higher
    than the period time of the time-series
    """
    # convert start and end times to datetime objects:
    start_time = parse_datetime(start_time)
    end_time = parse_datetime(end_time)
    
    # first slicing: little wider in both sides to deal with the fact that the desired period time ("step")
    # is always not higher than the period time of the ts (extracted as "freq"):
    freq = ts.iloc[-1,0] - ts.iloc[-2,0]
    slice = ts[(ts['ds'] >= start_time - freq) & (ts['ds'] <= end_time + freq)]

    # resample to the desired period time, step (because the use of bfill() - need to set "ds" as index):
    slice = slice.set_index('ds').resample(step).bfill()

    # second slicing: to get the desired range:
    slice = slice[start_time:end_time].reset_index()

    return slice
    # return pd.DataFrame([1,2,3])

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
        
        # get forecasts for metric (potentially multiple time-series)
        # forecasts are list of Metric objects
        if platform.system() == 'Windows':
            metric = metric.replace(":","$")
        with open('forecasts/' + metric + '.pkl', 'rb') as inp:
            metrics_list = pickle.load(inp)

        for metric_obj in metrics_list:
            # read forecast and cut only the relevant timestamps
            df = metric_obj.metric_values
            pred = get_slice_from_ts(df,
                                    start_time=args.start_time,
                                    end_time=args.end_time,
                                    step=args.step)

            # read actual metric from prometheus
            metirc_full_name = get_full_metric_name(metric_obj)
            # since we call get_metric_list on specific metric (only one time-series) we expect to get
            # one element list, hence the [0]
            actual = get_metric_list(prom, metirc_full_name,
                                    args.start_time, args.end_time, args.step)[0].metric_values

            # check for anomaly:
            anomaly = is_anomaly(actual, pred)
            print("\nmetric: {} anomaly state is {}.\n".format(metirc_full_name, anomaly))
