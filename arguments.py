import argparse


def get_params():
    parser = argparse.ArgumentParser(description="Anomaly Detecion Trainer")
    parser.add_argument("--start_time", default="30d",
                        help="Any valid datetime string for dateparser.parse(). default: 30d. \
                        The start time of the historic data. Can be absolute or relative. \
                        If absolute need to be in YMD order.")
    parser.add_argument("--end_time", default="now",
                        help="Any valid datetime string for dateparser.parse(). default: now. \
                        The end time of the historic data. Can be absolute or relative. \
                        If absolute need to be in YMD order.")
    parser.add_argument("--step", default="1m",
                        help="Any valid frequency for pd.date_range, such as 'D' or 'M', default='1m'. \
                        The desired resolution of the historic data to be fitted.")
    parser.add_argument("--debug", action="store_true")

    args, _ = parser.parse_known_args()
    return args