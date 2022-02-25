import pandas as pd
from prometheus_api_client import MetricsList
from prometheus_api_client.utils import parse_datetime
from prometheus_api_client.metric_range_df import MetricRangeDataFrame


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


def get_metrics_from_file(file_path):
    # get metrics list from file
    with open(file_path) as f:
        metrics = f.read().splitlines()
    # remove commented or empty lines from list:
    metrics = [q for q in metrics if (len(q) > 0) and (not q.startswith("#"))]
    return metrics


def query_to_df_all(prom, query, start_time, end_time, step):
    # the difference from query_to_df is in the case of query that results in multiple time-series
    # the function returns all the time-series
    start_time = parse_datetime(start_time)
    end_time = parse_datetime(end_time)
    query_range = prom.custom_query_range(query ,
                                start_time=start_time,
                                end_time=end_time,
                                step=step)
    df = (MetricRangeDataFrame(query_range).reset_index()
                                           .rename(columns={'timestamp':'ds', 'value':'y'})
                                           .astype({'y':'float'}))
    df['ds'] = pd.to_datetime(df['ds'],unit='s').astype('datetime64[ns, Asia/Jerusalem]').dt.tz_localize(None)
    return df


def get_metric_list(prom, metric, start_time, end_time, step):
    # take metric or recorded rule and returns metric list object
    start_time = parse_datetime(start_time)
    end_time = parse_datetime(end_time)
    # use custom_query_range and not get_metric_range_data to exploit the implicit resampling
    metric_data = prom.custom_query_range(metric,
                            start_time=start_time,
                            end_time=end_time,
                            step=step)
    metric_object_list = MetricsList(metric_data)
    for item in metric_object_list:
        item.metric_values.ds = item.metric_values.ds.astype('datetime64[ns, Asia/Jerusalem]').dt.tz_localize(None)
    return metric_object_list


def get_full_metric_name(metric):
    # this function transform the format of label_config attribute in Metric object to one that can be
    # passed to api call to prometheus to get specific query, with all the labels
    metric_labels = str(metric.label_config).replace(", '",", ").replace("': ","=").replace("{'","{")
    return metric.metric_name + metric_labels
