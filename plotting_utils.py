import pandas as pd
import numpy as np
import altair as alt
import altair_saver
import glob
import os
import copy
import collections
import traceback
import json


# ---------------- Plot themes ------------------------

def personal():
    return {
        'config': {
            'view': {
                'height': 300,
                'width': 400,
            },
            'range': {
                'category': {'scheme': 'set2'},
                'ordinal': {'scheme': 'plasma'},
            },
            'legend': {
                'labelLimit': 0,
            },
            'background': 'white',
            'mark': {
                'clip': True,
            },
            'line': {
                'size': 3,
#                 'opacity': 0.4
            },


        }
    }


def publication():
    stroke_color = '333'
    title_size = 24
    label_size = 20
    line_width = 5

    return {
        'config': {
            'view': {
                'height': 500,
                'width': 600,
                'strokeWidth': 0,
                'background': 'white',
            },
            'title': {
                'fontSize': title_size,
            },
            'range': {
                'category': {'scheme': 'set2'},
                'ordinal': {'scheme': 'plasma'},
            },
            'axis': {
                'titleFontSize': title_size,
                'labelFontSize': label_size,
                'grid': False,
                'domainWidth': 5,
                'domainColor': stroke_color,
                'tickWidth': 3,
                'tickSize': 9,
                'tickCount': 4,
                'tickColor': stroke_color,
                'tickOffset': 0,
            },
            'legend': {
                'titleFontSize': title_size,
                'labelFontSize': label_size,
                'labelLimit': 0,
                'titleLimit': 0,
                'orient': 'top-left',
#                 'padding': 10,
                'titlePadding': 10,
#                 'rowPadding': 5,
                'fillColor': '#ffffff88',
#                 'strokeColor': 'black',
                'cornerRadius': 0,
            },
            'rule': {
                'size': 3,
                'color': '999',
                # 'strokeDash': [4, 4],
            },
            'line': {
                'size': line_width,
#                 'opacity': 0.4
            },
        }
    }

alt.themes.register('personal', personal)
alt.themes.register('publication', publication)


# ----------- Data loading -----------------------------

def load_args(path):
    with open(path + '/args.json') as f:
        args = json.load(f)
    return args


def merge_args(df, args_dict):
    df = df.copy()
    for k, v in args_dict.items():
        df[k] = v
    return df


def load_jobs(pattern, subdir='exploration', root='.', title=None):
    jobs = glob.glob(f'{root}/results/{subdir}/{pattern}')
    results = []
    for job in jobs:
        try:
            name = os.path.basename(os.path.normpath(job))
            train_data = pd.read_csv(job + '/train.csv')
            train_data['test'] = False
            test_data = pd.read_csv(job + '/test.csv')
            test_data['test'] = True
            data = pd.concat([train_data, test_data], sort=False)
            data['name'] = name

            args_dict = load_args(job)
            data = merge_args(data, args_dict)

            results.append(data)
        except Exception as e:
            print(e)
    df = pd.concat(results, sort=False)
    if title is None:
        df['title'] = df['name'].str.replace(r'_seed\d', '')
    else:
        df['title'] = title
    return df.reset_index(drop=True)


def load_sac_results(env, task, title='SAC'):
    sac_results = pd.read_csv('results/sac.csv')
    sac_results = sac_results[sac_results.env == f'{env}_{task}']
    sac_results['env'] = env
    sac_results['task'] = task
    sac_results['test'] = True
    sac_results['use_exploration'] = False
    sac_results['score'] = sac_results['episode_reward']
    sac_results['name'] = title
    sac_results['title'] = title
    return sac_results


# ----------------- Plotting functions ------------------------

INCLUDE_FIELDS = {
    'name', 'title', 'test', 'episode', 'score', 'novelty_score',
}

SidecarChart = collections.namedtuple('SidecarChart', ['chart', 'included_fields'])


def strip_columns(chart, included_fields):
    included_fields.discard(None)
    included_fields = {f.rsplit(':')[0] for f in included_fields}
    df = chart.data
    if isinstance(df, pd.DataFrame):
        chart.data = df.loc[:, df.columns.isin(included_fields)]
    return chart


def make_base_chart(data, title, window=5, **kwargs):
    chart = alt.Chart(data, title=title).mark_line().encode(
        x=alt.X('episode', title='Episode'),
        **kwargs
    ).transform_calculate(
        has_score=(alt.datum.score > 0.5),
    ).transform_window(
        sum_novelty='sum(novelty_score)',
        frame=[None, 0],
        groupby=['name', 'test'],
        sort=[{'field': 'episode', 'order': 'ascending'}],
    ).transform_window(
        sum_score='sum(score)',
        frame=[None, 0],
        groupby=['name', 'test'],
        sort=[{'field': 'episode', 'order': 'ascending'}],
    ).transform_window(
        count_score='sum(has_score)',
        frame=[None, 0],
        groupby=['name', 'test'],
        sort=[{'field': 'episode', 'order': 'ascending'}],
    ).transform_window(
        rolling_mean_score='mean(score)',
        frame=[-window, 0],
        groupby=['name', 'test'],
        sort=[{'field': 'episode', 'order': 'ascending'}]
    ).transform_window(
        rolling_mean_novelty='mean(novelty_score)',
        frame=[-window, 0],
        groupby=['name', 'test'],
        sort=[{'field': 'episode', 'order': 'ascending'}],
    ).transform_calculate(
        smoothed_score=((alt.datum.name == 'SAC') * alt.datum.score +
                        (alt.datum.name != 'SAC') * alt.datum.rolling_mean_score)
    )
    used_fields = {
        v.split(':')[0] for k, v in kwargs.items() if isinstance(v, str)
    }
    included_fields = INCLUDE_FIELDS.union(used_fields)
    return SidecarChart(chart, included_fields)


def plot_with_bars(base_chart, y_col, test, extent='ci', strip=True, title_flag=None,
                   included_fields=set(), y_args={}, **kwargs):
    if isinstance(base_chart, SidecarChart):
        base_chart, chart_includes = base_chart
        included_fields = included_fields.union(chart_includes)
    included_fields = included_fields.union({y_col})

    legend_chart = base_chart.mark_line(size=0, opacity=1).encode(
        y=alt.Y(f'mean({y_col}):Q', **y_args),
        **kwargs
    ).transform_filter(alt.datum.test == test)
    mean_chart = base_chart.encode(
        y=alt.Y(f'mean({y_col}):Q', **y_args),
        **kwargs
    ).transform_filter(alt.datum.test == test)
    err_chart = base_chart.encode(
        y=alt.Y(f'{y_col}:Q', **y_args),
    ).transform_filter(alt.datum.test == test).mark_errorband(extent=extent)

    chart = legend_chart + err_chart + mean_chart
    if title_flag is None:
        title_flag = ' [test]' if test else ' [train]'
    chart.title = base_chart.title + title_flag
    if strip:
        return strip_columns(chart, included_fields)
    return chart