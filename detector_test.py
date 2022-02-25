from detector import get_slice_from_ts, is_anomaly
import pandas as pd
import pickle
import numpy as np

# load test data frame with period time of 15 seconds and define desired output period time to
# 5 seconds #
with open('test_data/prometheus_tsdb_head_chunks.pkl', 'rb') as inp:
    metric_obj = pickle.load(inp)
df_inp = metric_obj[0].metric_values

step ='5s'


### Test get_slice_from_ts ###

# load expected output and extract start and end time for 1st and 3rd use cases#
df_out_identical_expected = pd.read_csv('test_data/df_out_identical.csv', parse_dates=['ds'])
start_time_identical = df_out_identical_expected.ds.iloc[0]
end_time_identical = df_out_identical_expected.ds.iloc[-1]

# define earlier start time than the first time of the input range:
start_time_earlier = df_inp.ds.iloc[0] - pd.Timedelta(1,'h')
end_time_later = df_inp.ds.iloc[-1] + pd.Timedelta(1,'h')

### Use Cases ###

def test_get_slice_from_ts_identical_timestamps():
    """
    start time and end time are integer multiples of the original period time.

    For example, 11:30:30 or 23:00:15, if the period time is 15 seconds.
    """

    df_out_identical = get_slice_from_ts(df_inp, start_time_identical, end_time_identical,step=step)
    pd.testing.assert_frame_equal(df_out_identical, df_out_identical_expected)


def test_get_slice_from_ts_not_identical_timestamps():
    """
    start time and end time are not integer multiples of the original period time.

    For example, 11:30:25 or 23:00:05, if the period time is 15 seconds.
    """

    # load expected output and extract start and end time #
    df_out_not_identical_expected = pd.read_csv('test_data/df_out_not_identical.csv', parse_dates=['ds'])
    start_time_not_identical = df_out_not_identical_expected.ds.iloc[0]
    end_time_not_identical = df_out_not_identical_expected.ds.iloc[-1]

    # test #
    df_out_not_identical = get_slice_from_ts(df_inp, start_time_not_identical,
                                             end_time_not_identical,step=step)
    pd.testing.assert_frame_equal(df_out_not_identical, df_out_not_identical_expected)


def test_get_slice_from_ts_relative_timestamps():
    """
    simulate real case where start time and end time have milliseconds presicion (since we use now()).
    """
    start_time = start_time_identical - pd.Timedelta(5000 * np.random.rand(), 'ms')
    end_time = end_time_identical + pd.Timedelta(5000 * np.random.rand(), 'ms')

    # test #
    df_out = get_slice_from_ts(df_inp, start_time, end_time, step=step)
    pd.testing.assert_frame_equal(df_out, df_out_identical_expected)

### Edge Cases ###

def test_get_slice_from_ts_start_equals_end_identical():
    # load expected output and extract start and end time #
    df_expected = pd.read_csv('test_data/df_out_one_row_identical.csv', parse_dates=['ds'])
    start_time = df_expected.ds.iloc[0]
    end_time = df_expected.ds.iloc[-1]

    # test #
    df_out = get_slice_from_ts(df_inp, start_time, end_time, step=step)
    pd.testing.assert_frame_equal(df_out, df_expected)

def test_get_slice_from_ts_start_equals_end_non_identical():
    # load expected output and extract start and end time #
    df_expected = pd.read_csv('test_data/df_out_one_row_non_identical.csv', parse_dates=['ds'])
    start_time = df_expected.ds.iloc[0]
    end_time = df_expected.ds.iloc[-1]

    # test #
    df_out = get_slice_from_ts(df_inp, start_time, end_time, step=step)
    pd.testing.assert_frame_equal(df_out, df_expected)

def test_get_slice_from_ts_start_equals_end_ms_precision():
    """
    Since start time and end time are equal, but are not integer multiple of the output period time,
    no row of the resampled input df can be catched by them and we should get empty df
    """
    # define start and end time #
    start_time = start_time_identical + pd.Timedelta(5000 * np.random.rand(), 'ms')
    end_time = start_time

    # test #
    df_out = get_slice_from_ts(df_inp, start_time, end_time, step=step)
    assert df_out.empty

def test_get_slice_from_ts_start_and_end_out_of_range():
    end_time_earlier = start_time_earlier + pd.Timedelta(30,'m')

    # test #
    df_out = get_slice_from_ts(df_inp, start_time_earlier, end_time_earlier, step=step)
    assert df_out.empty

def test_get_slice_from_ts_start_in_range_end_out_of_range():
    start_time = df_inp.ds.iloc[-4]
    df_out_expected = pd.read_csv('test_data/df_out_tail.csv', parse_dates=['ds'])

    # test #
    df_out = get_slice_from_ts(df_inp, start_time, end_time_later, step=step)
    pd.testing.assert_frame_equal(df_out, df_out_expected)

def test_get_slice_from_ts_start_out_of_range_end_in_range():
    end_time = df_inp.ds.iloc[3]
    df_out_expected = pd.read_csv('test_data/df_out_head.csv', parse_dates=['ds'])

    # test #
    df_out = get_slice_from_ts(df_inp, start_time_earlier, end_time, step=step)
    pd.testing.assert_frame_equal(df_out, df_out_expected)

### Test is_anomaly ###

# we can use df_out_identical_expected as the input data for the test
pred = df_out_identical_expected
len_pred = len(pred)
len_majority = len_pred // 2 + 1
len_minority = len_pred - len_majority
pred_min = pred.yhat_lower.min()
pred_max = pred.yhat_upper.max()
pred_yhat = pred.yhat.values

# test data for different scenarios
no_anomaly1 = pd.DataFrame(pred_yhat,columns=['y'])
no_anomaly2 = pd.DataFrame(np.append(pred_yhat[:len_majority], [pred_min - 1] * len_minority),
                            columns=['y'])
no_anomaly3 = pd.DataFrame(np.append(pred_yhat[:len_majority], [pred_max + 1] * len_minority),
                            columns=['y'])

lower = pd.DataFrame(np.append(pred_yhat[:len_minority], [pred_min - 1] * len_majority),
                            columns=['y'])
upper = pd.DataFrame(np.append(pred_yhat[:len_minority], [pred_max + 1] * len_majority),
                            columns=['y'])


def test_is_anomaly_actual_equals_yhat():
    check_upper = is_anomaly(no_anomaly1, pred)
    check_lower = is_anomaly(no_anomaly1, pred, anomaly_type="lower")
    check_both = is_anomaly(no_anomaly1, pred, anomaly_type="both")
    assert (check_upper, check_lower, check_both) == (0, 0, 0)

def test_is_anomaly_minority_lower_than_min():
    check_upper = is_anomaly(no_anomaly2, pred)
    check_lower = is_anomaly(no_anomaly2, pred, anomaly_type="lower")
    check_both = is_anomaly(no_anomaly2, pred, anomaly_type="both")
    assert (check_upper, check_lower, check_both) == (0, 0, 0)

def test_is_anomaly_minority_higher_than_max():
    check_upper = is_anomaly(no_anomaly3, pred)
    check_lower = is_anomaly(no_anomaly3, pred, anomaly_type="lower")
    check_both = is_anomaly(no_anomaly3, pred, anomaly_type="both")
    assert (check_upper, check_lower, check_both) == (0, 0, 0)

def test_is_anomaly_actual_lower():
    check_upper = is_anomaly(lower, pred)
    check_lower = is_anomaly(lower, pred, anomaly_type="lower")
    check_both = is_anomaly(lower, pred, anomaly_type="both")
    assert (check_upper, check_lower, check_both) == (0, -1, -1)

def test_is_anomaly_actual_upper():
    check_upper = is_anomaly(upper, pred)
    check_lower = is_anomaly(upper, pred, anomaly_type="lower")
    check_both = is_anomaly(upper, pred, anomaly_type="both")
    assert (check_upper, check_lower, check_both) == (1, 0, 1)      