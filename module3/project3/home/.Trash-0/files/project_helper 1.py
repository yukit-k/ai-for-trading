import numpy as np
from IPython.core.display import display, HTML
import plotly.graph_objs as go
from plotly import tools
import cvxopt

import plotly.offline as offline_py
offline_py.init_notebook_mode(connected=True)


_color_scheme = {
    'index': '#B6B2CF',
    'etf': '#2D3ECF',
    'tracking_error': '#6F91DE',
    'df_header': 'silver',
    'df_value': 'white',
    'df_line': 'silver',
    'heatmap_colorscale': [(0, '#6F91DE'), (0.5, 'grey'), (1, 'red')]}


def _generate_config():
    return {'showLink': False, 'displayModeBar': False, 'showAxisRangeEntryBoxes': True}


def _generate_heatmap_trace(df, x_label, y_label, z_label, scale_min, scale_max):
    x_hover_text_values = np.tile(df.columns, (len(df.index), 1))
    y_hover_text_values = np.tile(df.index, (len(df.columns), 1))

    padding_len = np.full(3, max(len(x_label), len(y_label), len(z_label))) -\
                  [len(x_label), len(y_label), len(z_label)]
    # Additional padding added to ticker and date to align
    hover_text = y_label + ':  ' + padding_len[1] * ' ' + y_hover_text_values.T + '<br>' + \
                 x_label + ':  ' + padding_len[0] * ' ' + x_hover_text_values + '<br>' + \
                 z_label + ': ' + padding_len[2] * ' ' + df.applymap('{:.3f}'.format)

    return go.Heatmap(
        x=df.columns,
        y=df.index,
        z=df.values,
        zauto=False,
        zmax=scale_max,
        zmin=scale_min,
        colorscale=_color_scheme['heatmap_colorscale'],
        text=hover_text.values,
        hoverinfo='text')


def _sanatize_string(string):
    return ''.join([i for i in string if i.isalpha()])


def large_dollar_volume_stocks(df, price_column, volume_column, top_percent):
    """
    Get the stocks with the largest dollar volume stocks.

    Parameters
    ----------
    df : DataFrame
        Stock prices with dates and ticker symbols
    price_column : str
        The column with the price data in `df`
    volume_column : str
        The column with the volume in `df`
    top_percent : float
        The top x percent to consider largest in the stock universe

    Returns
    -------
    large_dollar_volume_stocks_symbols : List of str
        List of of large dollar volume stock symbols
    """
    dollar_traded = df.groupby('ticker').apply(lambda row: sum(row[volume_column] * row[price_column]))

    return dollar_traded.sort_values().tail(int(len(dollar_traded) * top_percent)).index.values.tolist()


def plot_benchmark_returns(index_data, etf_data, title):
    config = _generate_config()
    index_trace = go.Scatter(
        name='Index',
        x=index_data.index,
        y=index_data,
        line={'color': _color_scheme['index']})
    etf_trace = go.Scatter(
        name='ETF',
        x=etf_data.index,
        y=etf_data,
        line={'color': _color_scheme['etf']})

    layout = go.Layout(
        title=title,
        xaxis={'title': 'Date'},
        yaxis={'title': 'Cumulative Returns', 'range': [0, 3]})

    fig = go.Figure(data=[index_trace, etf_trace], layout=layout)
    offline_py.iplot(fig, config=config)


def plot_tracking_error(tracking_error, title):
    config = _generate_config()
    trace = go.Scatter(
        x=tracking_error.index,
        y=tracking_error,
        line={'color': _color_scheme['tracking_error']})

    layout = go.Layout(
        title=title,
        xaxis={'title': 'Date'},
        yaxis={'title': 'Error', 'range': [-1.5, 1.5]})

    fig = go.Figure(data=[trace], layout=layout)
    offline_py.iplot(fig, config=config)


def print_dataframe(df, n_rows=10, n_columns=3):
    missing_val_str = '...'
    config = _generate_config()

    formatted_df = df.iloc[:n_rows, :n_columns]
    formatted_df = formatted_df.applymap('{:.3f}'.format)

    if len(df.columns) > n_columns:
        formatted_df[missing_val_str] = [missing_val_str]*len(formatted_df.index)
    if len(df.index) > n_rows:
        formatted_df.loc[missing_val_str] = [missing_val_str]*len(formatted_df.columns)

    trace = go.Table(
        type='table',
        columnwidth=[1, 3],
        header={
            'values': [''] + list(formatted_df.columns.values),
            'line': {'color': _color_scheme['df_line']},
            'fill': {'color': _color_scheme['df_header']},
            'font': {'size': 13}},
        cells={
            'values': formatted_df.reset_index().values.T,
            'line': {'color': _color_scheme['df_line']},
            'fill': {'color': [_color_scheme['df_header'], _color_scheme['df_value']]},
            'font': {'size': 13}})

    offline_py.iplot([trace], config=config)


def plot_weights(weights, title):
    config = _generate_config()
    graph_path = 'graphs/{}.html'.format(_sanatize_string(title))
    trace = _generate_heatmap_trace(weights, 'Date', 'Ticker', 'Weight', 0.0, 0.2)
    layout = go.Layout(
        title=title,
        xaxis={'title': 'Dates'},
        yaxis={'title': 'Tickers'})

    fig = go.Figure(data=[trace], layout=layout)
    offline_py.plot(fig, config=config, filename=graph_path, auto_open=False)
    display(HTML('The graph for {} is too large. You can view it <a href="{}" target="_blank">here</a>.'
                 .format(title, graph_path)))


def plot_returns(returns, title):
    config = _generate_config()
    graph_path = 'graphs/{}.html'.format(_sanatize_string(title))
    trace = _generate_heatmap_trace(returns, 'Date', 'Ticker', 'Weight', -0.3, 0.3)
    layout = go.Layout(
        title=title,
        xaxis={'title': 'Dates'},
        yaxis={'title': 'Tickers'})

    fig = go.Figure(data=[trace], layout=layout)
    offline_py.plot(fig, config=config, filename=graph_path, auto_open=False)
    display(HTML('The graph for {} is too large. You can view it <a href="{}" target="_blank">here</a>.'
                 .format(title, graph_path)))


def plot_covariance(xty, xtx):
    config = _generate_config()

    xty_trace = go.Bar(
        x=xty.index,
        y=xty.values)
    xtx_trace = _generate_heatmap_trace(xtx, 'Ticker 2', 'Ticker 1', 'Covariance', 0.0, 1.0)

    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=['XTY', 'XTX'], print_grid=False)
    fig.append_trace(xty_trace, 1, 1)
    fig.append_trace(xtx_trace, 1, 2)
    fig['layout']['xaxis1'].update(title='Tickers')
    fig['layout']['yaxis1'].update(title='Covariance')
    fig['layout']['xaxis2'].update(title='Tickers')
    fig['layout']['yaxis2'].update(title='Tickers')

    offline_py.iplot(fig, config=config)


def tracking_error(index_weighted_cumulative_returns, etf_weighted_cumulative_returns):
    assert index_weighted_cumulative_returns.index.equals(etf_weighted_cumulative_returns.index)
    return etf_weighted_cumulative_returns - index_weighted_cumulative_returns


def solve_qp(P, q):
    assert len(P.shape) == 2
    assert len(q.shape) == 1
    assert P.shape[0] == P.shape[1] == q.shape[0]

    nn = len(q)

    g = cvxopt.spmatrix(-1, range(nn), range(nn))
    a = cvxopt.matrix(np.ones(nn), (1, nn))
    b = cvxopt.matrix(1.0)
    h = cvxopt.matrix(np.zeros(nn))

    P = cvxopt.matrix(P)
    q = -cvxopt.matrix(q)

    # Min cov
    # Max return
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, g, h, a, b)

    if 'optimal' not in sol['status']:
        return np.array([])

    return np.array(sol['x']).flatten()
