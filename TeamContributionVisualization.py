import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUM_EVENTS = 6


def graph_scores(event_df, i):
    event_df = event_df.sort_values(by='RawTotal', ascending=False)

    fig, ax = plt.subplots()
    ax.set(title='Player Score',
           ylabel='Score (Points)',
           xlabel='Player')
    ax.bar(event_df['Name'], event_df['RawTotal'])

    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.4)

    fig.show()
    fig.savefig(os.path.join('visuals', 'event' + str(i) + 'visual.png'))



def calculate_ratings(event_df):
    event_df = event_df.sort_values(by='RawTotal', ascending=False)
    high1 = (event_df.iloc[0])['RawTotal']
    high2 = (event_df.iloc[1])['RawTotal']
    high3 = (event_df.iloc[2])['RawTotal']
    high4 = (event_df.iloc[3])['RawTotal']
    low = (event_df.tail(1))['RawTotal'].iloc[0]

    # =(userValue - minValue) / ((MaxValue - MinValue) / 10)

    event_df['1Score'] = (event_df['RawTotal'] - low) / ((high1 - low) / 10)
    event_df['2Score'] = (event_df['RawTotal'] - low) / ((high2 - low) / 10)
    event_df['3Score'] = (event_df['RawTotal'] - low) / ((high3 - low) / 10)
    event_df['4Score'] = (event_df['RawTotal'] - low) / ((high4 - low) / 10)

    calc1 = abs((event_df['1Score'].median() + event_df['1Score'].mean()) / 2 - 5)
    calc2 = abs((event_df['2Score'].median() + event_df['2Score'].mean()) / 2 - 5)
    calc3 = abs((event_df['3Score'].median() + event_df['3Score'].mean()) / 2 - 5)
    calc4 = abs((event_df['4Score'].median() + event_df['4Score'].mean()) / 2 - 5)

    calcs = [calc1, calc2, calc3, calc4]
    best_calc = np.argmin(calcs)

    if best_calc == 0:
        event_df['Score'] = event_df['1Score']
    if best_calc == 1:
        event_df['Score'] = event_df['2Score']
    if best_calc == 2:
        event_df['Score'] = event_df['3Score']
    if best_calc == 3:
        event_df['Score'] = event_df['4Score']

    event_df = event_df.drop('1Score', axis=1)
    event_df = event_df.drop('2Score', axis=1)
    event_df = event_df.drop('3Score', axis=1)
    event_df = event_df.drop('4Score', axis=1)

    print(event_df)

    return event_df

def merge_df(event_df):
    mask = event_df['Name'].str.contains('^Total-', regex=True)
    event_df = event_df[~mask]

    # Pivot to get points for each game
    df_points = event_df.pivot(index=['Team', 'Name'], columns='Game', values='Temp Points').reset_index()

    # Pivot to get multipliers for each game
    df_multipliers = event_df.pivot(index=['Team', 'Name'], columns='Game', values='Multiplier').reset_index()

    df_points.columns = ['Team', 'Name'] + [f'{col}_points' for col in df_points.columns[2:]]
    df_multipliers.columns = ['Team', 'Name'] + [f'{col}_multiplier' for col in df_multipliers.columns[2:]]

    # Merge the two pivoted DataFrames
    merged_df = pd.merge(df_points, df_multipliers, on=['Team', 'Name'])

    return merged_df


def calculate_total(row):
    games = [col for col in df.columns if col.endswith('_points')]
    total = 0.0
    for game in games:
        total += row[game]
    return total


def calculate_raw_total(row):
    games_pts = [col for col in df.columns if col.endswith('_points')]
    games = []
    for game in games_pts:
        games.append(game[:-len("_points")])

    total = 0.0
    for game in games:
        total += row[game + "_points"] / row[game + "_multiplier"]
    return total


if __name__ == '__main__':
    event_dfs = []

    for i in range(NUM_EVENTS + 1):
        if i > 0:
            event_dfs.append(pd.read_csv(os.path.join('points_files', 'points' + str(i) + '.csv')))

    for i, df in enumerate(event_dfs):
        event_dfs[i] = merge_df(df)

    for i, df in enumerate(event_dfs):
        df["Total"] = df.apply(calculate_total, axis=1)
        df['RawTotal'] = df.apply(calculate_raw_total, axis=1)
        event_dfs[i] = df

    for i, df in enumerate(event_dfs):
        df.to_csv(os.path.join('points_output', 'points' + str(i + 1) + '.csv'), index=False)

    for i, df in enumerate(event_dfs):
        event_dfs[i] = calculate_ratings(df)

