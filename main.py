import os
import pathlib
import time

import numpy as np
import pandas as pd
import shapely
from fastapi import FastAPI
from pandas.core.common import flatten
from shapely.geometry import Point
from tqdm import tqdm

from availability import calculate_availability
from build_structures import create_road_ball_tree, create_zone_ball_tree, create_jurisdiction_ball_tree
from data_lake import DataLake
from flow import calculate_flow
from pickup_dropoff import calculate_pickup_dropoff
from trip_volume import calculate_trip_volume
from util import PointRequestBody, LineRequestBody, TripBasedRequestBody, StatusChangeBasedRequestBody

USE_CACHE = True
DEBUG = False
STAT_TESTING_ONLY = False
CATEGORY_COLUMNS = ['provider_name', 'vehicle_type', 'propulsion_types']
MATCH_TYPES = ['zone', 'street', 'bin', 'jurisdiction']
# using pandas time series offset notation https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
TIME_GROUPS = ['D', 'H', '30min', '15min']
# MATCH_TYPES = ['street']
# CATEGORY_COLUMNS = ['provider_name']
# TIME_GROUPS = ['D']

EARTH_RADIUS = 6371008  # meters

tqdm.pandas()

road_tree, road_ids, road_df = None, None, None
zones_tree, zones_ids, zones_df = None, None, None
jurisdiction_tree, jurisdiction_ids, jurisdiction_df = None, None, None


def find_containing_zone(point, df):
    """This is slow, and should not be used across an entire dataset, unless absolutely necessary"""
    zone = df[df['geometry'].contains(point)].index.values
    return zone[0] if len(zone) > 0 else None


def query_tree(tree, ids, points, k=1, timer_desc=None, return_distance=False):
    """
    Query the ball tree for the nearest k points
    :param tree: the ball tree
    :param ids: the ids of the points in the tree
    :param points: the points to query (shapely Points)
    :param k: the number of nearest neighbors to return
    :param timer_desc: a description to use for the timer
    :param return_distance: whether to return the distance to the nearest neighbor. will return in meters
    """
    start = time.process_time()

    # convert to radians
    points_radians = [[np.deg2rad(point.x), np.deg2rad(point.y)] for point in points]

    # query the tree
    distances = []
    if return_distance:
        distances, indices = tree.query(points_radians, k=k, return_distance=True)
        # convert to meters
        distances = distances * EARTH_RADIUS
    else:
        indices = tree.query(points_radians, k=k, return_distance=False)

    # get the ids of the nearest neighbors
    nested_ids = []
    for index_arr in indices:
        nested_ids.append([ids[index] for index in index_arr])

    # nested_ids = [ids[index] for sub in indices for index in sub]

    if DEBUG and timer_desc:
        print(f'{timer_desc}: {time.process_time() - start} seconds')

    if return_distance:
        return distances, nested_ids
    else:
        return nested_ids


def find_matching_zones(points, tree, ids, df):
    """
    Use the tree to find the closest first_search_size centroids to narrow the search space, and then use the shapely contains function to find the matching zone.
    This is done once more with the closest second_search_size centroids if the first search space does not find a match.
    Shapely contains is very slow, so this is still slow, but much faster than the brute force method.
    """
    first_search_size = 5
    # second_search_size = 100
    second_search_size = df.shape[0] if df.shape[0] < 100 else 100

    start = time.process_time()
    # query the tree
    nested_zones = query_tree(tree, ids, points, k=first_search_size,
                              timer_desc='first zone query for all points' if DEBUG else None)

    # mash em together and find the ids
    unique_zones = list({zone for sub in nested_zones for zone in sub})

    # matching every point against the union of all closest zone is significantly faster than building a new gdf for each point
    matching_zones_df = df[df.index.isin(unique_zones)]

    zones = []
    for i in range(len(points)):
        point = points[i]
        zone = find_containing_zone(point, matching_zones_df)
        if zone is None:
            if DEBUG:
                print('no zone found in initial query, expanding search')
            # if no zone is found, expand the search radius
            closest_zones = \
                query_tree(tree, ids, points, k=second_search_size,
                           timer_desc='second query for zones' if DEBUG else None)[
                    0]

            zone = find_containing_zone(point, df[df.index.isin(closest_zones)])

            if zone is None:
                if DEBUG:
                    print('no zone found in expanded query, finding closest zone using entire dataset')
                # if we still can't find a zone, expand the search to the entire dataset
                zone = find_containing_zone(point, df)

        if zone:
            zones.append(zone)
        else:
            if DEBUG:
                print('the point {} is not in any zone'.format(point.wkt))
    if DEBUG:
        print('matched {} points to {} zone in {} seconds'.format(len(points), len(zones), time.process_time() - start))

    zones = [int(zone) for zone in zones]
    return zones


def save_cache_csv(df: pd.DataFrame, report_date: str, stat_type: str, match_type: str):
    print('saving cache csv')
    filepath = f'/cache/{report_date.replace("-", "/")}/{stat_type}_{match_type}.csv'
    pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)
    return filepath


if not STAT_TESTING_ONLY:
    # This will almost always just be *loading* the data, not doing the processing.
    # Processing is done before the server is started.
    if road_tree is None or road_ids is None or road_df is None:
        road_tree, road_ids, road_df = create_road_ball_tree()
    if zones_tree is None or zones_ids is None or zones_df is None:
        zones_tree, zones_ids, zones_df = create_zone_ball_tree()
    if jurisdiction_tree is None or jurisdiction_ids is None or jurisdiction_df is None:
        jurisdiction_tree, jurisdiction_ids, jurisdiction_df = create_jurisdiction_ball_tree()
    # road_df.to_crs(epsg=4326) .to_csv('/cache/road_df.csv')
    # zones_df.to_crs(epsg=4326).to_csv('/cache/zones_df.csv')
    # turn road_df into a dml statement
    # dml = 'INSERT INTO road_df (road_seg_id, geom) VALUES '
    # for index, row in road_df.iterrows():
    #     dml += f"('{index}', 'ST_ {row['geometry'].wkt}'),"

# initialize the data lake
print('initializing data lake')
data_lake = DataLake(account_name=os.getenv('AZURE_STORAGE_ACCOUNT_NAME'),
                     account_key=os.getenv('AZURE_STORAGE_ACCOUNT_KEY'))

print('initialized!')

# data_lake.list_directory_contents('mdc-data-lake-file-system', '2022/')

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/match_line/")
async def match_line(body: LineRequestBody):
    global road_tree, road_ids, road_df, zones_tree, zones_ids, zones_df

    line = shapely.wkt.loads(body.line)
    points = [Point(c) for c in line.coords]

    roadsegids = query_tree(road_tree, road_ids, points, k=1, timer_desc='road query for all points' if DEBUG else None)
    roadsegids = list(flatten(roadsegids))

    zones = find_matching_zones(points, zones_tree, zones_ids, zones_df)

    jurisdictions = find_matching_zones(points, jurisdiction_tree, jurisdiction_ids, jurisdiction_df)

    return {
        'street': {
            'roadsegid': roadsegids,
            'pickup': roadsegids[0],
            'dropoff': roadsegids[-1],
            'geometry': road_df[road_df.index.isin(roadsegids)]['geometry'].to_json()
        },
        'zone': {
            'zoneid': zones,
            'pickup': zones[0],
            'dropoff': zones[-1]
        },
        'jurisdiction': {
            'jurisdictionid': jurisdictions,
            'pickup': jurisdictions[0],
            'dropoff': jurisdictions[-1]
        }
    }


@app.post("/match_point/")
async def match_line(body: PointRequestBody):
    global road_tree, road_ids, road_df, zones_tree, zones_ids, zones_df, jurisdiction_tree, jurisdiction_ids, jurisdiction_df

    point = shapely.wkt.loads(body.point)

    roadsegid = \
        list(flatten(
            query_tree(road_tree, road_ids, [point], k=1, timer_desc='road query for point' if DEBUG else None)))[
            0]

    zones = find_matching_zones([point], zones_tree, zones_ids, zones_df)
    jurisdictions = find_matching_zones([point], jurisdiction_tree, jurisdiction_ids, jurisdiction_df)

    return {
        'street': {
            'roadsegid': roadsegid,
            'geometry': road_df[road_df.index == roadsegid]['geometry'].to_json()
        },
        'zone': {
            'zoneid': zones[0] if len(zones) else None,
            # 'geometry': zones_df[zones_df['id'] == zone]['geometry'].to_json()
        },
        'jurisdiction': {
            'jurisdictionid': jurisdictions[0] if len(jurisdictions) else None,
        }
    }


@app.post("/trip_volume/")
async def trip_volume(body: TripBasedRequestBody):
    stats = calculate_trip_volume(body.trips, body.privacy_minimum, CATEGORY_COLUMNS, MATCH_TYPES, TIME_GROUPS)
    for match_type in stats:
        filepath = save_cache_csv(stats[match_type], body.report_date, 'trip_volume', match_type)
        data_lake.upload_file('mdc-data-lake-file-system', body.report_date.replace('-', '/'),
                              f'trip_volume_{match_type}.csv', filepath)


@app.post("/pickup/")
async def pickup(body: TripBasedRequestBody):
    match_types = ['zone', 'street', 'bin', 'jurisdiction']
    stats = calculate_pickup_dropoff(mode='pickup', trips=body.trips, category_columns=CATEGORY_COLUMNS,
                                     match_types=match_types, time_groups=TIME_GROUPS)
    for match_type in stats:
        filepath = save_cache_csv(stats[match_type], body.report_date, 'pickup', match_type)
        data_lake.upload_file('mdc-data-lake-file-system', body.report_date.replace('-', '/'),
                              f'pickup_{match_type}.csv', filepath)


@app.post("/dropoff/")
async def pickup(body: TripBasedRequestBody):
    match_types = ['zone', 'street', 'bin', 'jurisdiction']
    stats = calculate_pickup_dropoff(mode='dropoff', trips=body.trips, category_columns=CATEGORY_COLUMNS,
                                     match_types=match_types, time_groups=TIME_GROUPS)
    for match_type in stats:
        filepath = save_cache_csv(stats[match_type], body.report_date, 'dropoff', match_type)
        data_lake.upload_file('mdc-data-lake-file-system', body.report_date.replace('-', '/'),
                              f'dropoff_{match_type}.csv', filepath)


@app.post("/flow/")
async def flow(body: TripBasedRequestBody):
    match_types = ['zone', 'street', 'bin', 'jurisdiction']
    stats = calculate_flow(trips=body.trips, privacy_minimum=body.privacy_minimum, category_columns=CATEGORY_COLUMNS,
                           match_types=match_types,
                           time_groups=TIME_GROUPS)
    for match_type in stats:
        filepath = save_cache_csv(stats[match_type], body.report_date, 'flow', match_type)
        data_lake.upload_file('mdc-data-lake-file-system', body.report_date.replace('-', '/'),
                              f'flow_{match_type}.csv', filepath)


@app.post("/availability/")
async def availability(body: StatusChangeBasedRequestBody):
    match_types = ['zone', 'street', 'bin', 'jurisdiction']
    stats = calculate_availability(status_changes=body.status_changes, category_columns=CATEGORY_COLUMNS,
                                   match_types=match_types,
                                   time_groups=TIME_GROUPS)
    for match_type in stats:
        filepath = save_cache_csv(stats[match_type], body.report_date, 'availability', match_type)
        data_lake.upload_file('mdc-data-lake-file-system', body.report_date.replace('-', '/'),
                              f'availability_{match_type}.csv', filepath)
