import trainer
import pandas as pd

def test_get_data():
    df = trainer.get_data()
    assert isinstance(df.iloc[0,0], pd.Timestamp)
    assert isinstance(df.iloc[0,1],float)

def test_fit_predict(periods=300):
    df = pd.read_csv('example_wp_log_peyton_manning.csv', parse_dates=['ds'])
    forecast = trainer.fit_predict(df, periods=periods)
    assert len(forecast) == len(df) + periods
    assert len(forecast.columns) == 4
    assert forecast.isnull().sum().sum() == 0


test_get_data()
test_fit_predict()
