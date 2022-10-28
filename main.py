import json
import os
import pickle
import time

import geopandas
import numpy as np
import pandas as pd
import shapely.wkt
from math import radians
from shapely.geometry import Point

from sklearn.neighbors import BallTree
from tqdm import tqdm

from interpolate import load_roads_from_file

USE_CACHE = False

EARTH_RADIUS = 6371008  # meters

tqdm.pandas()

def create_points_dict(roads_df):
    points_dict = {}
    for index, row in tqdm(roads_df.iterrows(), total=len(roads_df)):
        for point in row['geometry'].coords:
            if point not in points_dict:
                points_dict[point] = []
            points_dict[point].append(index)
    # get just points that have multiple ids
    dup_points_dict = {k: v for k, v in points_dict.items() if len(v) > 1}
    # use the keys as geometries and make a multipoint geometry but make points with them
    with open('data/dup_points.geojson', 'w') as f:
        f.write(geopandas.GeoSeries(map(lambda p: Point(p), dup_points_dict.keys())).to_json())

    return points_dict

    # points = {}
    # for road in roads['features']:
    #     for point in road['geometry']['coordinates']:
    #         if tuple(point) not in points:
    #             points[tuple(point)] = []
    #         points[tuple(point)].append(road['properties']['ROADSEGID'])
    # return points


def create_ball_tree(leaf_size=40):
    if os.path.exists('data/ball_tree.pickle') and os.path.exists('data/ball_tree_ids.pickle') and USE_CACHE:
        with open('data/ball_tree.pickle', 'rb') as f:
            tree = pickle.load(f)
        with open('data/ball_tree_ids.pickle', 'rb') as f:
            ids = pickle.load(f)
        return tree, ids
    else:
        roads = load_roads_from_file()
        points_dict = create_points_dict(roads)
        points = list(points_dict.keys())
        points_radians = [[np.deg2rad(point[0]), np.deg2rad(point[1])] for point in points]
        ids = list(points_dict.values())
        tree = BallTree([list(point) for point in points_radians], metric='haversine', leaf_size=leaf_size)
        with open('data/ball_tree.pickle', 'wb') as f:
            pickle.dump(tree, f)
        with open('data/ball_tree_ids.pickle', 'wb') as f:
            pickle.dump(ids, f)
        return tree, ids


def load_mds_from_file():
    df = pd.read_csv('data/mds-trips.csv')
    # load geom column as wkt to geodataframe
    df['geom'] = df['geom'].progress_apply(lambda wkt: shapely.wkt.loads(wkt))
    return df

    with open('data/mds.json') as f:
        data = json.load(f)
        return data['data']['trips']


def extract_points_from_mds(mds):
    points = []
    for trip in mds:
        for point in trip['route']['features']:
            points.append(point['geometry']['coordinates'])
    return points


def generate_geojson_with_mds(mds):
    features = []
    for trip in mds:
        for point in trip['route']['features']:
            features.append(point)
    return {
        "type": "FeatureCollection",
        "features": features
    }


def main():
    leaf_size = 40
    start = time.process_time()
    tree, ids = create_ball_tree()
    print('loaded roads in {} seconds'.format(time.process_time() - start))

    mds_df = load_mds_from_file()
    mds_df['geom_rad'] = mds_df['geom'].apply(lambda p: [np.deg2rad(p.x), np.deg2rad(p.y)])
    # ids = mds_df['id'].tolist()
    points = mds_df['geom_rad'].tolist()
    # points_radians = [[radians(point[0]), radians(point[1])] for point in points]
    start = time.process_time()
    distances, indices = tree.query(points, k=1)
    print('leaf size {}: {} seconds'.format(leaf_size, time.process_time() - start))
    mds_df['idx'] = indices
    mds_df['distance'] = distances
    mds_df['distance_meters'] = mds_df['distance'].apply(lambda d: d * EARTH_RADIUS)
    mds_df['x'] = mds_df['geom'].apply(lambda p: p.x)
    mds_df['y'] = mds_df['geom'].apply(lambda p: p.y)
    # match idx to the indicies in ids
    mds_df['ROADSEGID'] = mds_df['idx'].apply(lambda idx: ids[idx][0])


    mds_df.drop(columns=['distance', 'geom_rad', 'geom', 'idx']).to_csv('data/mds-trips-with-roadsegid.csv'.format(leaf_size), index=False)

    idx_of_max_distance = mds_df['distance'].argmax()
    max_distance = distances[idx_of_max_distance]
    max_distance_id = ids[indices[idx_of_max_distance][0]]
    print(points[idx_of_max_distance], max_distance, max_distance_id)


if __name__ == '__main__':
    main()
