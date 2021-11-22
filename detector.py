import pandas as pd
from prophet import Prophet

class Detector():

    def __init__(self, forecast_file_name, start=None, n_samples=10, freq='min'):
        self.start = start
        self.n_samples = n_samples
        self.forecast_file_name = forecast_file_name
        self.freq = freq
        if self.start is not None:
            self.end = pd.Timestamp(self.start) + pd.Timedelta(self.n_samples - 1, self.freq)

    # get recent data
    def get_recent_data(self):
        df = pd.read_csv('example_wp_log_peyton_manning.csv', parse_dates=['ds'])
        if self.start is not None:
            df = df[(df['ds'] >= self.start) & (df['ds'] <= self.end)]
        else:
            df = df.tail(self.n_samples)
        return df

    # get forecast
    def get_forecast(self):
        df = pd.read_csv(self.forecast_file_name)
        if self.start is not None:
            df = df[(df['ds'] >= self.start) & (df['ds'] <= self.end)]
        else:
            df = df.tail(self.n_samples)
        return df

    # compare
    # def anomaly_detector(actual,forecast):
