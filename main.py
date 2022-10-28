import json
import os
import pickle
import time

import geopandas
from math import radians
from shapely.geometry import Point

from sklearn.neighbors import BallTree
from tqdm import tqdm

from interpolate import load_roads_from_file

USE_CACHE = False

EARTH_RADIUS = 6371008  # meters


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


def create_ball_tree():
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
        points_radians = [[radians(point[0]), radians(point[1])] for point in points]
        ids = list(points_dict.values())
        tree = BallTree([list(point) for point in points_radians], metric='haversine')
        with open('data/ball_tree.pickle', 'wb') as f:
            pickle.dump(tree, f)
        with open('data/ball_tree_ids.pickle', 'wb') as f:
            pickle.dump(ids, f)
        return tree, ids


def load_mds_from_file():
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
    start = time.process_time()
    tree, ids = create_ball_tree()
    print('loaded roads in {} seconds'.format(time.process_time() - start))

    mds = load_mds_from_file()
    points = extract_points_from_mds(mds)
    points_radians = [[radians(point[0]), radians(point[1])] for point in points]
    start = time.process_time()
    distances, indices = tree.query(points_radians, k=1)
    print('queried {} points in {} seconds'.format(len(points), time.process_time() - start))
    distances = distances * EARTH_RADIUS

    idx_of_max_distance = distances.argmax()
    max_distance = distances[idx_of_max_distance]
    max_distance_id = ids[indices[idx_of_max_distance][0]]
    print(points[idx_of_max_distance], max_distance, max_distance_id)


if __name__ == '__main__':
    main()
