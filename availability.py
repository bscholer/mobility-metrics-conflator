import itertools

import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import h3

from util import print_full, MdsTripWithMatches, MdsStatusChangeWithMatches


def calculate_availability(status_changes: list[MdsStatusChangeWithMatches],
                           category_columns: list[str], match_types: list[str], time_groups: list[str]):
    to_return = {}
    """Calculate the trip volume for a list of trips"""
    # load the trips into a dataframe
    # print(trips[0])
    df = pd.DataFrame([change.to_dict() for change in status_changes])
    # start_time unix timestamp to datetime
    df['event_time'] = pd.to_datetime(df['event_time'], unit='ms')
    # use as index
    df = df.set_index('event_time')
    print_full(df.head())
    min_time = df.index.min()
    max_time = df.index.max()

    # make propulsion_types a comma separated string
    df['propulsion_types'] = df['propulsion_types'].apply(lambda x: ','.join(sorted(x)))
    # make columns for each match category
    for match_type in match_types:
        results = []
        match_df = df.copy()
        match_df[match_type] = match_df['matches'].apply(lambda x: x[match_type] if match_type in x else None)
        # drop nones
        match_df.dropna(subset=[match_type], inplace=True)
        match_df.drop(columns=['matches'], inplace=True)

        # for each category column make a distinct list of the values in that column
        category_dict = {}
        for category_col in category_columns:
            category_possible_values = list(set(match_df[category_col].to_list()))
            category_possible_values.append('all')
            category_dict[category_col] = category_possible_values

        # make a dictionary of all the possible combinations of the category columns
        category_combinations = {}
        for category_col in category_columns:
            category_combinations[category_col] = category_dict[category_col]
        category_combinations = list(
            dict(zip(category_combinations, x)) for x in itertools.product(*category_combinations.values()))

        # for each combination of categories, filter the dataframe to only include the rows that match that combination
        # then group by the time groups and count the number of rows in each group
        # then add the results to a list of results
        for category_combination in category_combinations:
            print(category_combination)
            match_df_filtered = match_df.copy()
            for category_col in category_columns:
                if category_combination[category_col] != 'all':
                    match_df_filtered = match_df_filtered[
                        match_df_filtered[category_col] == category_combination[category_col]]
            # print(match_df_filtered.head())

            for time_group in time_groups:
                # make a list of all possible values for the time group, between min_time and max_time
                time_group_possible_values = pd.date_range(min_time, max_time, freq=time_group)
                print(time_group_possible_values)

                # for each time group, find the most recent vehicle_state for each vehicle_id before that time group
                for time_group_value in time_group_possible_values:
                    print(time_group_value)
                    # filter the dataframe to only include rows before the time group value
                    match_df_filtered_time_group = match_df_filtered[match_df_filtered.index < time_group_value]
                    # print(match_df_filtered_time_group.head())
                    # group by vehicle_id and propulsion_types and find the most recent vehicle_state for each group
                    match_df_filtered_time_group = match_df_filtered_time_group.groupby(['vehicle_id']).last()
                    print(match_df_filtered_time_group.head())

                # for group_name, group in match_df_filtered_grouped:
                #     # group by the match column and count the number of rows in each group
                #     group = group.groupby(match_type).count()
                #
                #     # add to results
                #     for match_value, count in group.iterrows():
                #         result = category_combination.copy()
                #         result[match_type] = match_value
                #         result['count'] = count[category_columns[0]]
                #         result['time_group'] = time_group
                #         result['time_group_value'] = group_name
                #         # if match_type == 'bins':
                #         #     result['geom'] = [list(c) for c in h3.h3_to_geo_boundary(match_value)]
                #         #     result['geom'] = f'POLYGON(({",".join([f"{c[1]} {c[0]}" for c in result["geom"]])}))'
                #         results.append(result)
                #
        # save results to csv file formatted well
        results_df = pd.DataFrame(results)
        # results_df.to_csv(f'/cache/trip_volume_{match_type}.csv', index=False)
        to_return[match_type] = results_df

    return to_return
