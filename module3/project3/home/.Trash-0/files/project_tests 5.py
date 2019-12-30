from collections import OrderedDict
import pandas as pd
import numpy as np
from datetime import date, timedelta


pd.options.display.float_format = '{:.8f}'.format


def _generate_random_tickers(n_tickers=None):
    min_ticker_len = 3
    max_ticker_len = 5
    tickers = []

    if not n_tickers:
        n_tickers = np.random.randint(8, 14)

    ticker_symbol_random = np.random.randint(ord('A'), ord('Z')+1, (n_tickers, max_ticker_len))
    ticker_symbol_lengths = np.random.randint(min_ticker_len, max_ticker_len, n_tickers)
    for ticker_symbol_rand, ticker_symbol_length in zip(ticker_symbol_random, ticker_symbol_lengths):
        ticker_symbol = ''.join([chr(c_id) for c_id in ticker_symbol_rand[:ticker_symbol_length]])
        tickers.append(ticker_symbol)

    return tickers


def _generate_random_dates(n_days=None):
    if not n_days:
        n_days = np.random.randint(14, 20)

    start_year = np.random.randint(1999, 2017)
    start_month = np.random.randint(1, 12)
    start_day = np.random.randint(1, 29)
    start_date = date(start_year, start_month, start_day)

    dates = []
    for i in range(n_days):
        dates.append(start_date + timedelta(days=i))

    return dates


def _generate_random_dfs(n_df, index, columns):
    all_df_data = np.random.random((n_df, len(index), len(columns)))

    return [pd.DataFrame(df_data, index, columns) for df_data in all_df_data]


def _generate_output_error_msg(fn_name, fn_inputs, fn_outputs, fn_expected_outputs):
    formatted_inputs = []
    formatted_outputs = []
    formatted_expected_outputs = []

    for input_name, input_value in fn_inputs.items():
        formatted_outputs.append('INPUT {}:\n{}\n'.format(
            input_name, str(input_value)))
    for output_name, output_value in fn_outputs.items():
        formatted_outputs.append('OUTPUT {}:\n{}\n'.format(
            output_name, str(output_value)))
    for expected_output_name, expected_output_value in fn_expected_outputs.items():
        formatted_expected_outputs.append('EXPECTED OUTPUT FOR {}:\n{}\n'.format(
            expected_output_name, str(expected_output_value)))

    return 'Wrong value for {}.\n' \
           '{}\n' \
           '{}\n' \
           '{}' \
        .format(
            fn_name,
            '\n'.join(formatted_inputs),
            '\n'.join(formatted_outputs),
            '\n'.join(formatted_expected_outputs))


def _assert_output(fn, fn_inputs, fn_expected_outputs):
    assert type(fn_expected_outputs) == OrderedDict

    fn_outputs = OrderedDict()
    fn_raw_out = fn(**fn_inputs)

    if len(fn_expected_outputs) == 1:
        fn_outputs[list(fn_expected_outputs)[0]] = fn_raw_out
    elif len(fn_expected_outputs) > 1:
        assert type(fn_raw_out) == tuple,\
            'Expecting function to return tuple, got type {}'.format(type(fn_raw_out))
        assert len(fn_raw_out) == len(fn_expected_outputs),\
            'Expected {} outputs in tuple, only found {} outputs'.format(len(fn_expected_outputs), len(fn_raw_out))
        for key_i, output_key in enumerate(fn_expected_outputs.keys()):
            fn_outputs[output_key] = fn_raw_out[key_i]

    err_message = _generate_output_error_msg(
        fn.__name__,
        fn_inputs,
        fn_outputs,
        fn_expected_outputs)

    for fn_out, (out_name, expected_out) in zip(fn_outputs.values(), fn_expected_outputs.items()):
        assert isinstance(fn_out, type(expected_out)),\
            'Wrong type for output {}. Got {}, expected {}'.format(out_name, type(fn_out), type(expected_out))

        if hasattr(expected_out, 'shape'):
            assert fn_out.shape == expected_out.shape, \
                'Wrong shape for output {}. Got {}, expected {}'.format(out_name, fn_out.shape, expected_out.shape)

        if type(expected_out) == pd.DataFrame:
            assert set(fn_out.columns) == set(expected_out.columns), \
                'Incorrect columns for output {}\n' \
                'COLUMNS:          {}\n' \
                'EXPECTED COLUMNS: {}'.format(out_name, sorted(fn_out.columns), sorted(expected_out.columns))

        if type(expected_out) in {pd.DataFrame, pd.Series}:
            assert set(fn_out.index) == set(expected_out.index), \
                'Incorrect indices for output {}\n' \
                'INDICES:          {}\n' \
                'EXPECTED INDICES: {}'.format(out_name, sorted(fn_out.index), sorted(expected_out.index))

        out_is_close = np.isclose(fn_out, expected_out, equal_nan=True)

        if not isinstance(out_is_close, bool):
            out_is_close = out_is_close.all()

        assert out_is_close, err_message


def project_test(func):
    def func_wrapper(*args):
        result = func(*args)
        print('Tests Passed')
        return result

    return func_wrapper


@project_test
def test_generate_weighted_returns(fn):
    tickers = _generate_random_tickers(3)
    dates = _generate_random_dates(4)

    fn_inputs = {
        'returns': pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [1.59904743, 1.66397210, 1.67345829],
                [-0.37065629, -0.36541822, -0.36015840],
                [-0.41055669, 0.60004777, 0.00536958]],
            dates, tickers),
        'weights': pd.DataFrame(
            [
                [0.03777059, 0.04733924, 0.05197790],
                [0.82074874, 0.48533938, 0.75792752],
                [0.10196420, 0.05866016, 0.09578226],
                [0.03951647, 0.40866122, 0.09431233]],
            dates, tickers)}
    fn_correct_outputs = OrderedDict([
        (
            'weighted_returns',
                pd.DataFrame(
                    [
                        [np.nan, np.nan, np.nan],
                        [1.31241616, 0.80759119, 1.26836009],
                        [-0.03779367, -0.02143549, -0.03449679],
                        [-0.01622375, 0.24521625, 0.00050642]],
                    dates, tickers))])

    _assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_generate_returns(fn):
    tickers = _generate_random_tickers(3)
    dates = _generate_random_dates(4)

    fn_inputs = {
        'prices': pd.DataFrame(
            [
                [35.4411, 34.1799, 34.0223],
                [92.1131, 91.0543, 90.9572],
                [57.9708, 57.7814, 58.1982],
                [34.1705, 92.453, 58.5107]],
            dates, tickers)}
    fn_correct_outputs = OrderedDict([
        (
            'returns',
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [1.59904743, 1.66397210, 1.67345829],
                    [-0.37065629, -0.36541822, -0.36015840],
                    [-0.41055669, 0.60004777, 0.00536958]],
                dates, tickers))])

    _assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_generate_dollar_volume_weights(fn):
    tickers = _generate_random_tickers(3)
    dates = _generate_random_dates(4)

    fn_inputs = {
        'close': pd.DataFrame(
            [
                [35.4411, 34.1799, 34.0223],
                [92.1131, 91.0543, 90.9572],
                [57.9708, 57.7814, 58.1982],
                [34.1705, 92.453, 58.5107]],
            dates, tickers),
        'volume': pd.DataFrame(
            [
                [9.83683e+06, 1.78072e+07, 8.82982e+06],
                [8.22427e+07, 6.85315e+07, 4.81601e+07],
                [1.62348e+07, 1.30527e+07, 9.51201e+06],
                [1.06742e+07, 5.68313e+07, 9.31601e+06]],
            dates, tickers)}
    fn_correct_outputs = OrderedDict([
        (
            'dollar_volume_weights',
            pd.DataFrame(
                [
                    [0.27719777, 0.48394253, 0.23885970],
                     [0.41632975, 0.34293308, 0.24073717],
                     [0.41848548, 0.33536102, 0.24615350],
                     [0.05917255, 0.85239760, 0.08842984]],
                dates, tickers))])

    _assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_get_optimal_weights(fn):
    fn_inputs = {
        'covariance_returns': np.array(
            [
                [0.143123, 0.0216755, 0.014273],
                [0.0216755, 0.0401826, 0.00663152],
                [0.014273, 0.00663152, 0.044963]]),
        'index_weights': pd.Series([0.23623892, 0.0125628, 0.7511982], ['A', 'B', 'C'])}
    fn_correct_outputs = OrderedDict([
        (
            'x',
            np.array([0.23623897, 0.01256285, 0.75119817]))])

    _assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_calculate_cumulative_returns(fn):
    tickers = _generate_random_tickers(3)
    dates = _generate_random_dates(4)

    fn_inputs = {
        'returns': pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [1.59904743, 1.66397210, 1.67345829],
                [-0.37065629, -0.36541822, -0.36015840],
                [-0.41055669, 0.60004777, 0.00536958]],
            dates, tickers)}
    fn_correct_outputs = OrderedDict([
        (
            'cumulative_returns',
            pd.Series(
                [np.nan, 5.93647782, -0.57128454, -0.68260542],
                dates))])

    _assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_calculate_dividend_weights(fn):
    tickers = _generate_random_tickers(3)
    dates = _generate_random_dates(4)

    fn_inputs = {
        'ex_dividend': pd.DataFrame(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.1],
                [0.0, 1.0, 0.3],
                [0.0, 0.2, 0.0]],
            dates, tickers)}
    fn_correct_outputs = OrderedDict([
        (
            'dividend_weights',
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [0.00000000, 0.00000000, 1.00000000],
                    [0.00000000, 0.71428571, 0.28571429],
                    [0.00000000, 0.75000000, 0.25000000]],
                dates, tickers))])

    _assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_get_covariance_returns(fn):
    tickers = _generate_random_tickers(3)
    dates = _generate_random_dates(4)

    fn_inputs = {
        'returns': pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [1.59904743, 1.66397210, 1.67345829],
                [-0.37065629, -0.36541822, -0.36015840],
                [-0.41055669, 0.60004777, 0.00536958]],
            dates, tickers)}
    fn_correct_outputs = OrderedDict([(
        'returns_covariance',
        np.array(
            [
                [0.89856076, 0.7205586, 0.8458721],
                [0.7205586, 0.78707297, 0.76450378],
                [0.8458721, 0.76450378, 0.83182775]]))])

    _assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_rebalance_portfolio(fn):
    tickers = _generate_random_tickers(3)
    dates = _generate_random_dates(11)

    fn_inputs = {
        'returns': pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [-0.02202381, 0.02265285, 0.01441961],
                [0.01947657, 0.00551985, 0.00047382],
                [0.00537313, -0.00803232, 0.01160313],
                [0.00593824, -0.00567773, 0.02247191],
                [0.02479339, 0.01758824, -0.00824176],
                [-0.0109447, -0.00383568, 0.01361958],
                [0.01164822, 0.01558719, 0.00614894],
                [0.0109384, -0.00182079, 0.02900868],
                [0.01138952, 0.00218049, -0.00954495],
                [0.0106982, 0.00644535, -0.01815329]],
            dates, tickers),
        'median_index_weights': pd.DataFrame(
            [
                [0.00449404, 0.11586048, 0.00359727],
                [0.00403487, 0.12534048, 0.0034428, ],
                [0.00423485, 0.12854258, 0.00347404],
                [0.00395679, 0.1243466, 0.00335064],
                [0.00368729, 0.11750295, 0.00333929],
                [0.00369562, 0.11447422, 0.00325973],
                [0.00379612, 0.11088075, 0.0031734, ],
                [0.00366501, 0.10806014, 0.00314648],
                [0.00361268, 0.10376514, 0.00323257],
                [0.00358844, 0.10097531, 0.00319009],
                [0.00362045, 0.09791232, 0.00318071]],
            dates, tickers),
        'shift_size': 2,
        'chunk_size': 4}
    fn_correct_outputs = OrderedDict([
        (
            'all_rebalance_weights',
            [
                np.array([0.29341237, 0.41378419, 0.29280344]),
                np.array([0.29654088, 0.40731481, 0.29614432]),
                np.array([0.29868214, 0.40308791, 0.29822995]),
                np.array([0.30100044, 0.39839644, 0.30060312])]
        )])

    _assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_get_portfolio_turnover(fn):
    fn_inputs = {
        'all_rebalance_weights': [
            np.array([0.00012205033508460705, 0.0003019915743383353, 0.999575958090577]),
            np.array([1.305709815242165e-05, 8.112998801084706e-06, 0.9999788299030465]),
            np.array([0.3917481750142896, 0.5607687848565064, 0.0474830401292039])],
        'shift_size': 3,
        'rebalance_count': 11}
    fn_correct_outputs = OrderedDict([('portfolio_turnover', 14.553361377)])

    _assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_tracking_error(fn):
    dates = _generate_random_dates(4)

    fn_inputs = {
        'index_weighted_cumulative_returns': pd.Series(
                [np.nan, 0.99880148, 0.99876653, 1.00024411],
                dates),
        'etf_weighted_cumulative_returns': pd.Series(
                [np.nan, 0.63859274, 0.93475823, 2.57295727],
                dates)}
    fn_correct_outputs = OrderedDict([
        (
            'tracking_error',
            pd.Series([np.nan, 0.03243758, 0.00102427, 0.61835667], dates))])

    _assert_output(fn, fn_inputs, fn_correct_outputs)
