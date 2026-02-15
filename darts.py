import numpy as np
from scipy.stats import norm

from dash import Dash, html, dcc, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go

board_size = 1000
board_range_mm = np.linspace(-190, 190, 380, endpoint=False)


def score_polar(r, th):
    radial = [
        (6.35, 'bulls_eye'),
        (15.9, 'bull'),
        (99, 'single'),
        (107, 'triple'),
        (162, 'single'),
        (170, 'double')]

    tangential = [
        (9, 20),
        (27, 1),
        (45, 18),
        (63, 4),
        (81, 13),
        (99, 6),
        (117, 10),
        (135, 15),
        (153, 2),
        (171, 17),
        (189, 3),
        (207, 19),
        (225, 7),
        (243, 16),
        (261, 8),
        (279, 11),
        (297, 14),
        (315, 9),
        (333, 12),
        (351, 5),
        (361, 20)]

    bulls_score = {
        'bulls_eye': 50,
        'bull': 25}

    multipliers = {
        'single': 1,
        'double': 2,
        'triple': 3}

    def find_in_ranges(ranges, pos):
        for limit, value in ranges:
            if limit >= pos:
                return value

    radial_segment = find_in_ranges(radial, r)

    if radial_segment in bulls_score:
        return bulls_score[radial_segment]

    multiplier = multipliers.get(radial_segment, 0)
    field_value = find_in_ranges(tangential, np.degrees(th))

    return field_value * multiplier


def score_rec(x, y):
    r = np.sqrt(x**2 + y**2)
    th = np.arctan2(x, y)
    if th < 0:
        th += 2*np.pi

    return score_polar(r, th)


def mm_to_coord(mm):
    return int(mm+board_size/2)


board = np.empty((board_size, board_size))
for x in board_range_mm:
    for y in board_range_mm:
        board[mm_to_coord(y)][mm_to_coord(x)] = score_rec(x, y)


def get_score_map(spread):
    size = int(2.5*spread)
    gauss_1d = np.array([
        norm.pdf(x, loc=size/2, scale=spread/2)
        for x in np.arange(size)])
    shots_distribution = np.outer(gauss_1d, gauss_1d)
    dist_size = shots_distribution.shape[0]
    offset = int(dist_size/2)

    score = np.empty((board_size, board_size))
    for x_mm in board_range_mm:
        x_coord = mm_to_coord(x_mm)
        x_start = x_coord - offset
        x_end = x_start + dist_size
        for y_mm in board_range_mm:
            y_coord = mm_to_coord(y_mm)
            y_start = y_coord - offset
            y_end = y_start + dist_size
            score[y_coord][x_coord] = np.sum(
                board[y_start:y_end, x_start:x_end] * shots_distribution)

    return score


app = Dash()


@callback(
    Output('score-map', 'figure'),
    Input('spread', 'value'))
def update_output(spread):
    score_map = get_score_map(spread)[mm_to_coord(-200):mm_to_coord(200), mm_to_coord(-200):mm_to_coord(200)]  # noqa: 501
    xy_max = np.argmax(score_map)
    y_max = np.floor(xy_max/score_map.shape[0])
    x_max = np.mod(xy_max, score_map.shape[0])

    fig = px.imshow(
        score_map,
        color_continuous_scale='hot',
        zmax=60,
        # width=800,
        # height=800,
        origin='lower')

    fig.update_traces(hovertemplate='Score: %{z}<extra></extra>')
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.add_trace(go.Scatter(
        x=[x_max],
        y=[y_max],
        marker=dict(
            color='white',
            size=16,
            symbol='star'),
        mode='markers+text',
        text=[f'{np.max(score_map):.1f}'],
        textposition="bottom center",
        textfont_color='white'))

    return fig


app.layout = html.Div([
    html.Div(
        'Durchschnittlicher Score nach Ziel und Genauigkeit',
        style=dict(
            fontSize='larger',
            fontFamily='Sans-serif',
            marginBottom='20px')),
    html.Div(
        'Durchmesser der Fläche, in die 70% der Würfe fallen:',
        style=dict(
            fontFamily='Sans-serif',
            marginBottom='10px')),
    html.Div(
        dcc.Slider(
            1, 100, 1,
            value=10,
            allow_direct_input=False,
            id='spread'),
        style=dict(
            width=800)),
    dcc.Loading(
        dcc.Graph(
            id='score-map',
            config=dict(
                displayModeBar=False)),
        overlay_style={'visibility': 'visible', 'filter': 'blur(5px)'},)
])

if __name__ == '__main__':
    app.run()


def get_app():
    return app
