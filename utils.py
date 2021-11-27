import pandas as pd
from prometheus_api_client.utils import parse_datetime

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