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

def get_queries(file_path):
    # get queries list from file
    with open(file_path) as f:
        queries = f.read().splitlines()
    # remove commented queries from list:
    queries = [q for q in queries if not q.startswith("#")]
    return queries

def query_to_df_new(prom, query, start_time, end_time, step):
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

def query_to_metrics_list(prom, query, start_time, end_time, step):
    # this function is good only in the case where the query is simple metric and not expression
    # like query_to_df_new() it supports query that yilds multiple time-series, like any summary metric
    # for example
    start_time = parse_datetime(start_time)
    end_time = parse_datetime(end_time)
    query_range = prom.custom_query_range(query ,
                                start_time=start_time,
                                end_time=end_time,
                                step=step)
    return MetricsList(query_range)

def get_full_metric_name(metric):
    # this function transform the format of label_config attribute in Metric object to one that can be
    # passed to api call to prometheus to get specific query, with all the labels
    metric_labels = str(metric.label_config).replace(", '",", ").replace("': ","=").replace("{'","{")
    return metric.metric_name + metric_labels
