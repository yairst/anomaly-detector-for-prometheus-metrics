from detector import Detector
import pandas as pd


def test_get_recent_data():
    d = Detector('forcast.csv', start='2012-03-07', n_samples=10, freq='D')
    df = d.get_recent_data()
    assert len(df) == 10
    assert df.iloc[0,0] == pd.Timestamp('2012-03-07')

def test_get_forecast():
    pass


test_get_recent_data()
