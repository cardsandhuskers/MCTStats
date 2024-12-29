import os
from functools import reduce

import pandas as pd

from TeamContributionVisualization import NUM_EVENTS


def remove_mults(df):
    for i in range(8):
        df[str(i)] = df.iloc[:, i + 2] / df.iloc[:, i + 10]
    return df

def pct_formula(val, total):
    return val / total


def calc_scores(df):
    for i in range(20, 28):
        total = df.iloc[:, i].sum()

        df[str(i-20) + "_PCT"] = df.iloc[:, i].apply(lambda x: pct_formula(x, total))

    df['SUM_PCT'] = df[['0_PCT','1_PCT','2_PCT','3_PCT','4_PCT','5_PCT','6_PCT','7_PCT']].sum(axis=1)

    # find second highest
    second_pct = df['SUM_PCT'].nlargest(2).iloc[-1]
    mult = 10 / second_pct

    # TODO: may want to end up adjusting mult formula to make bottom feeder 0, not true score
    df['SCORE'] = df['SUM_PCT'] * mult

    return df

def team_adjust(df):
    # Compute the average score for each team
    team_averages = df.groupby('Team')['SCORE'].mean().reset_index()
    team_averages.rename(columns={'SCORE': 'TEAM_AVG'}, inplace=True)

    # Merge the average scores back into the original DataFrame
    df = df.merge(team_averages, on='Team', how='left')

    df['DIFF'] = df['SCORE'] - df['TEAM_AVG']
    df['DEVSCORE'] = df['SCORE'] + df['DIFF']

    second_score = df['DEVSCORE'].nlargest(2).iloc[-1]
    min_score = df['DEVSCORE'].min()
    df['FINAL_SCORE'] = (df['DEVSCORE'] - min_score) / ((second_score - min_score) / 10)

    return df


def combine_event_scores(df_list):
    for i, df in enumerate(df_list):
        df.columns = [f"{col}_{i+1}" if col != 'Name' else col for col in df.columns]

    combined_df = reduce(lambda left, right: pd.merge(left, right, on='Name', how='outer'), df_list)

    cols_to_average = combined_df.columns[1:]
    combined_df['AVG'] = combined_df[cols_to_average].mean(axis=1, skipna=True)

    return combined_df


if __name__ == '__main__':
    for x in range(NUM_EVENTS + 1):
        if x > 0:
            print("Event " + str(x))
            base_point_df = remove_mults(pd.read_csv(os.path.join('points_output', 'points' + str(x) + '.csv')))

            score_df = calc_scores(base_point_df)

            score_adj_df = team_adjust(score_df)

            # rename cols
            for i in range(8):
                name = score_adj_df.columns[i + 2]
                name = name.split('_')[0]
                score_adj_df = score_adj_df.rename(columns={str(i): name + "_SCORE", str(i) + "_PCT": name + "_PCT"})

            # delete original datapoints
            score_adj_df = score_adj_df.drop(score_adj_df.columns[2:20], axis=1)

            score_adj_df.to_csv(os.path.join('score_output', 'scores' + str(x) + '.csv'), index=False)

    event_score_dfs = []
    for x in range(NUM_EVENTS + 1):
        if x > 0:
            base_score_df = pd.read_csv(os.path.join('score_output', 'scores' + str(x) + '.csv'))

            # remove cols
            base_score_df = base_score_df.drop(base_score_df.columns[2:23], axis=1)
            base_score_df = base_score_df.drop(base_score_df.columns[0], axis=1)

            print(base_score_df)

            event_score_dfs.append(base_score_df)

    final_df = combine_event_scores(event_score_dfs)
    final_df.to_csv(os.path.join('score_output', 'scores_combined.csv'), index=False)