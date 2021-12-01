import argparse


def get_params():
    parser = argparse.ArgumentParser(description="Anomaly Detecion Trainer")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--winsorizing", action="store_true")
    parser.add_argument("--start_time", default="30d",
                        help="Any valid datetime string for dateparser.parse(). default: 30d. \
                        The start time of the historic data. Can be absolute or relative. \
                        If absolute need to be in YMD order.")
    parser.add_argument("--end_time", default="now",
                        help="Any valid datetime string for dateparser.parse(). default: now. \
                        The end time of the historic data. Can be absolute or relative. \
                        If absolute need to be in YMD order.")
    parser.add_argument("--step", default="1min",
                        help="Any valid frequency for pd.date_range, such as 'D' or 'M', default='1m'. \
                        The desired resolution of the historic data to be fitted.")
    parser.add_argument("--url", default="http://localhost:9090",
                        help="Prometheus url. default: http://localhost:9090")
    parser.add_argument("--periods", default=1440, type=int,
                        help="The number of samples need to be forecasted. the default of 1440 stands for \
                        forecasting 24 hours ahead, assuming 1 minute resolution of the historic data. \
                        Note that this parameters is depend only on the resolution of the historic data \
                        and not on the resolution determined in --freq.")
    parser.add_argument("--freq",default='5min',
                        help="Any valid frequency for pd.date_range, such as 'D' or 'M', default='5min'. \
                        The desired resolution of the forecasted data points.")
    parser.add_argument("--seasonality_vals", nargs='+', type=float,
                        help="Seasonalities to add in case there are other seasonalities in addition or \
                        instead of daily, weekly or yearly. Need to be float number of days.")
    parser.add_argument("--seasonality_names", nargs='+',
                        help="seasonalities names in the order of seasonalities_vals")
    parser.add_argument("--seasonality_fourier", nargs='+', type=int,
                        help="seasonalities fourier order in the order of seasonalities_vals")

    args, _ = parser.parse_known_args()
    return args
