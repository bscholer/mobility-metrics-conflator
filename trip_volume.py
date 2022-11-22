import itertools

import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm

from util import print_full, MdsTripWithMatches


def calculate_trip_volume(trips: list[MdsTripWithMatches], privacy_minimum: int, category_columns: list[str], match_types: list[str], time_groups: list[str]):
    """Calculate the trip volume for a list of trips"""
    # load the trips into a dataframe
    # print(trips[0])
    df = pd.DataFrame([trip.to_dict() for trip in trips])
    # start_time unix timestamp to datetime
    df['start_time'] = pd.to_datetime(df['start_time'], unit='ms')
    # use as index
    df = df.set_index('start_time')

    # make propulsion_types a comma separated string
    df['propulsion_types'] = df['propulsion_types'].apply(lambda x: ','.join(sorted(x)))
    # make columns for each match category
    for match_type in match_types:
        results = []
        match_df = df.copy()
        match_df[match_type] = match_df['matches'].apply(lambda x: list(set(x[match_type])))
        match_df.drop(columns=['matches'], inplace=True)

        match_df = match_df.explode(match_type)
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
        category_combinations = list(dict(zip(category_combinations, x)) for x in itertools.product(*category_combinations.values()))

        # for each combination of categories, filter the dataframe to only include the rows that match that combination
        # then group by the time groups and count the number of rows in each group
        # then add the results to a list of results
        for category_combination in category_combinations:
            print(category_combination)
            match_df_filtered = match_df.copy()
            for category_col in category_columns:
                if category_combination[category_col] != 'all':
                    match_df_filtered = match_df_filtered[match_df_filtered[category_col] == category_combination[category_col]]
            # print(match_df_filtered.head())
            for time_group in time_groups:
                match_df_filtered_grouped = match_df_filtered.groupby(pd.Grouper(freq=time_group))
                for group_name, group in match_df_filtered_grouped:
                    # group by the match column and count the number of rows in each group
                    group = group.groupby(match_type).count()

                    # add to results
                    for match_value, count in group.iterrows():
                        # check if count is greater than privacy minimum
                        if count['trip_id'] > privacy_minimum:
                            result = category_combination.copy()
                            result[match_type] = match_value
                            result['count'] = count[category_columns[0]]
                            result['time_group'] = time_group
                            result['time_group_value'] = group_name
                            results.append(result)

        # save results to csv file formatted well
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'/cache/trip_volume_{match_type}.csv', index=False)
        return results_df



        # make a new df for every combination of category values
        # dfs = []
        # for category_col, category_values in category_dict.items():
        #     for category_value in category_values:
        #         dfs.append(match_df[match_df[category_col] == category_value])

    # divide the df into the smallest chunks, and then run the aggregation on each chunk



    # drop the matches column
    # df = df.drop(columns=['matches'])

    # make copie




    print_full(df.head())


    print('calculating trip volume')