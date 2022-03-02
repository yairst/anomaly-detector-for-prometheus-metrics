from trainer import fit_predict
import pandas as pd


df = pd.read_csv('prophet_get_started/example_wp_log_peyton_manning.csv', parse_dates=['ds'])
df = df.iloc[-100:]

# use case 1: freq is higher than the time-period of the df
def test_fit_predict_freq_higher_than_step():
    _, forecast = fit_predict(df, periods=6, freq='2D')
    assert len(forecast) == len(df) + 6

# use case 2: freq is lower than the time-period of the df
def test_fit_predict_freq_lower_than_step():
    _, forecast = fit_predict(df, periods=5, freq='H')
    assert len(forecast) == len(df) + 5

# use case 3: additional seasonality
def test_fit_predict_additional_seasonality():
    m, _ = fit_predict(df, periods=5, freq='D',
                             season={'names':['2days'],'vals':[2],'fourier':[3]})
    assert '2days' in m.seasonalities

# use case 4: adding country holidays
def test_fit_predict_with_country_holidays():
    m, _ = fit_predict(df, periods=5, freq='D', country='US')
    assert m.train_holiday_names is not None