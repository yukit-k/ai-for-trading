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
                [np.nan, -0.0355852, -0.00461228, 0.00435667],
                [np.nan, -0.0114943, -0.00106678, 0.016446],
                [np.nan, -0.00326797, 0.00721311, 0.00537109]],
            tickers, dates),
        'weights': pd.DataFrame(
            [
                [0.0045101, 0.00761073, 0.0050893, 0.00593444],
                [0.0980038, 0.0780279, 0.0742108, 0.0854871],
                [0.0121753, 0.00943077, 0.0093783, 0.00886865]],
            tickers, dates)}
    fn_correct_outputs = OrderedDict([
        (
            'weighted_returns',
                pd.DataFrame(
                [
                    [np.nan, -0.000270829, -2.34733e-05, 2.58544e-05],
                    [np.nan, -0.000896876, -7.91666e-05, 0.00140592],
                    [np.nan, -3.08195e-05, 6.76467e-05, 4.76343e-05]],
                tickers, dates))])

    _assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_generate_returns(fn):
    tickers = _generate_random_tickers(3)
    dates = _generate_random_dates(4)

    fn_inputs = {
        'close': pd.DataFrame(
            [
                [35.4411, 34.1799, 34.0223, 34.1705],
                [92.1131, 91.0543, 90.9572, 92.453],
                [57.9708, 57.7814, 58.1982, 58.5107]],
            tickers, dates)}
    fn_correct_outputs = OrderedDict([
        (
            'returns',
            pd.DataFrame(
                [
                    [np.nan, -0.0355858, -0.0046109, 0.00435597],
                    [np.nan, -0.0114946, -0.0010664, 0.0164451],
                    [np.nan, -0.00326716, 0.00721339, 0.00536958]],
                tickers, dates))])

    _assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_generate_dollar_volume_weights(fn):
    tickers = _generate_random_tickers(3)
    dates = _generate_random_dates(4)

    fn_inputs = {
        'close': pd.DataFrame(
            [
                [35.4411, 34.1799, 34.0223, 34.1705],
                [92.1131, 91.0543, 90.9572, 92.453],
                [57.9708, 57.7814, 58.1982, 58.5107]],
            tickers, dates),
        'volume': pd.DataFrame(
            [
                [9.83683e+06, 1.78072e+07, 8.82982e+06, 1.06742e+07],
                [8.22427e+07, 6.85315e+07, 4.81601e+07, 5.68313e+07],
                [1.62348e+07, 1.30527e+07, 9.51201e+06, 9.31601e+06]],
            tickers, dates)}
    fn_correct_outputs = OrderedDict([
        (
            'dollar_volume_weights',
            pd.DataFrame(
                [
                    [0.0393246, 0.0800543, 0.0573905, 0.0591726],
                    [0.854516, 0.820747, 0.836853, 0.852398],
                    [0.106159, 0.0991989, 0.105756, 0.0884298]],
                tickers, dates))])

    _assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_solve_qp(fn):
    fn_inputs = {
        'P': np.array(
            [
                [0.143123, 0.0216755, 0.014273],
                [0.0216755, 0.0401826, 0.00663152],
                [0.014273, 0.00663152, 0.044963]]),
        'q': np.array([0.0263612, 0.0156879, 0.0129376])}
    fn_correct_outputs = OrderedDict([
        (
            'x',
            np.array([0.13780886, 0.48383408, 0.37835706]))])

    _assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_calculate_cumulative_returns(fn):
    tickers = _generate_random_tickers(3)
    dates = _generate_random_dates(4)

    fn_inputs = {
        'returns': pd.DataFrame(
            [
                [np.nan, -0.000270829, -2.34733e-05, 2.58544e-05],
                [np.nan, -0.000896873, -7.91666e-05, 0.00140592],
                [np.nan, -3.08195e-05, 6.76468e-05, 4.76344e-05]],
            tickers, dates)}
    fn_correct_outputs = OrderedDict([
        (
            'cumulative_returns',
            pd.Series(
                [np.nan, 0.99880148, 0.99876653, 1.00024411],
                dates))])

    _assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_calculate_dividend_weights(fn):
    tickers = _generate_random_tickers(3)
    dates = _generate_random_dates(4)

    fn_inputs = {
        'ex_dividend': pd.DataFrame(
            [
                [0.0, 0.0, 0.1, 0.0],
                [0.0, 0.0, 0.0, 0.2],
                [0.0, 0.0, 0.0, 0.3]],
            tickers, dates)}
    fn_correct_outputs = OrderedDict([
        (
            'dividend_weights',
            pd.DataFrame(
                [
                    [np.nan, np.nan, 1.0, 0.16666666],
                    [np.nan, np.nan, 0.0, 0.33333333],
                    [np.nan, np.nan, 0.0, 0.5]],
                tickers, dates))])

    _assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_get_covariance(fn):
    tickers = _generate_random_tickers(3)
    dates = _generate_random_dates(4)

    fn_inputs = {
        'returns': pd.DataFrame(
            [
                [np.nan, -0.0355852, -0.00461228, 0.00435667],
                [np.nan, -0.0114943, -0.00106678, 0.016446],
                [np.nan, -0.00326797, 0.00721311, 0.00537109]],
            tickers, dates),
        'weighted_index_returns': pd.DataFrame(
            [
                [np.nan, -0.000270829, -2.34733e-05, 2.58544e-05],
                [np.nan, -0.000896873, -7.91666e-05, 0.00140592],
                [np.nan, -3.08195e-05, 6.76468e-05, 4.76344e-05]],
            tickers, dates)}
    fn_correct_outputs = OrderedDict([
        (
            'xtx',
            np.array(
                [
                    [0.00130656, 0.000485597, 0.000106423],
                    [0.000485597, 0.000403728, 0.000118201],
                    [0.000106423, 0.000118201, 9.15572e-05]])),
        (
            'xty',
            np.array([4.92563e-05, 3.81439e-05, 1.16104e-05]))])

    _assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_rebalance_portfolio(fn):
    tickers = _generate_random_tickers(3)
    dates = _generate_random_dates(11)

    fn_inputs = {
        'returns': pd.DataFrame(
            [
                [np.nan, -0.0355852, -0.00461228, 0.00435667, -0.0396183, -0.0121951,
                 0.00685871, -0.0027248, 0.0251973,-0.026947, -0.0465612],
                [np.nan, -0.0114943, -0.00106678, 0.016446, -0.0104013, -0.0040344,
                 -0.00557701, 0.000754961, 0.00678952, -0.00974095, -0.0234569],
                [np.nan, -0.00326797, 0.00721311, 0.00537109, -0.00501862, 0.0143183,
                 0.00272698, 0.019037, 0.000627943, -0.0163163, -0.00334928]],
            tickers, dates),
        'weighted_index_returns': pd.DataFrame(
            [
                [np.nan, -0.000270829, -2.34733e-05, 2.58544e-05, -0.000291808, -8.56712e-05,
                 5.10542e-05, -1.63907e-05, 0.000127297, -0.000126851, -0.000330526],
                [np.nan, -0.000896873, -7.91666e-05, 0.00140592, -0.000653316, -0.000246364,
                 -0.000395049, 4.47478e-05, 0.000389117, -0.000449979, -0.00254699],
                [np.nan, -3.08195e-05, 6.76468e-05, 4.76344e-05, -4.24937e-05, 0.000136497,
                 3.14274e-05, 0.000226068, 8.55098e-06, -0.000161634, -3.06379e-05]],

            tickers, dates),
        'shift_size': 3,
        'chunk_size': 2}
    fn_correct_outputs = OrderedDict([
        (
            'all_rebalance_weights',
            [
                np.array([0.00012205033508460705, 0.0003019915743383353, 0.999575958090577]),
                np.array([1.305709815242165e-05, 8.112998801084706e-06, 0.9999788299030465]),
                np.array([0.3917481750142896, 0.5607687848565064, 0.0474830401292039])])])

    _assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_get_rebalance_cost(fn):
    fn_inputs = {
        'all_rebalance_weights': [
            np.array([0.00012205033508460705, 0.0003019915743383353, 0.999575958090577]),
            np.array([1.305709815242165e-05, 8.112998801084706e-06, 0.9999788299030465]),
            np.array([0.3917481750142896, 0.5607687848565064, 0.0474830401292039])],
        'shift_size': 3,
        'rebalance_count': 11}
    fn_correct_outputs = OrderedDict([('rebalancing_cost', 0.51976290)])

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
            pd.Series([np.nan, 0.36020874, 0.06400830, -1.57271316], dates))])

    _assert_output(fn, fn_inputs, fn_correct_outputs)
