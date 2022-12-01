import os
import pickle
import sys
import time

import geopandas
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.neighbors import BallTree
from tqdm import tqdm

from interpolate import process_roads

tqdm.pandas()

USE_CACHE = True

opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

if "--ignore-cache" in opts:
    USE_CACHE = False


def load_shapefile(name, index_field=None):
    # name should be something like "road" or "zone". This is used to find the correct shapefile
    if os.path.exists(f'/cache/{name}.pickle') and USE_CACHE:
        print(f'loading {name}_df from pickle')
        df = pd.read_pickle(f'/cache/{name}.pickle')
    else:
        print(f'loading {name}_df from shapefile')
        start = time.process_time()
        df = geopandas.read_file(f'/data/{name}/{name}.shp')
        df = df.to_crs('EPSG:4326')
        if index_field:
            df = df.set_index(index_field)
        df.to_pickle(f'/cache/{name}.pickle')
        print(f'loaded {name}_df shapefile in {time.process_time() - start} seconds')
    return df


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
    with open('/data/dup_points.geojson', 'w') as f:
        f.write(geopandas.GeoSeries(map(lambda p: Point(p), dup_points_dict.keys())).to_json())

    return points_dict

    # points = {}
    # for road in roads['features']:
    #     for point in road['geometry']['coordinates']:
    #         if tuple(point) not in points:
    #             points[tuple(point)] = []
    #         points[tuple(point)].append(road['properties']['ROADSEGID'])
    # return points


def create_zone_ball_tree():
    df = load_shapefile('zone', index_field='MGRA')
    if os.path.exists('/cache/zone_ball_tree.pickle') and USE_CACHE:
        with open('/cache/zone_ball_tree.pickle', 'rb') as f:
            tree, ids = pickle.load(f)
        return tree, ids, df
    else:
        # find center of zone
        df['center'] = df['geometry'].progress_apply(lambda geom: Point(geom.centroid))
        df = df.drop(columns=['geometry'])
        df.set_geometry(df['center'], inplace=True)

        # convert to radians
        df['center'] = df['center'].progress_apply(lambda point: [np.deg2rad(point.x), np.deg2rad(point.y)])
        # create ball tree
        ids = df.index.tolist()
        tree = BallTree([list(point) for point in df['center']], metric='haversine', leaf_size=2)
        with open('/cache/zone_ball_tree.pickle', 'wb') as f:
            pickle.dump((tree, ids), f)
        return tree, ids, df


def create_jurisdiction_ball_tree():
    df = load_shapefile('jurisdiction', index_field='OBJECTID')
    if os.path.exists('/cache/jurisdiction_ball_tree.pickle') and USE_CACHE:
        with open('/cache/jurisdiction_ball_tree.pickle', 'rb') as f:
            tree, ids = pickle.load(f)
        return tree, ids, df
    else:
        # find center of zone
        df['center'] = df['geometry'].progress_apply(lambda geom: Point(geom.centroid))
        df = df.drop(columns=['geometry'])
        df.set_geometry(df['center'], inplace=True)

        # convert to radians
        df['center'] = df['center'].progress_apply(lambda point: [np.deg2rad(point.x), np.deg2rad(point.y)])
        # create ball tree
        ids = df.index.tolist()
        tree = BallTree([list(point) for point in df['center']], metric='haversine', leaf_size=2)
        with open('/cache/jurisdiction_ball_tree.pickle', 'wb') as f:
            pickle.dump((tree, ids), f)
        return tree, ids, df


def create_road_ball_tree():
    df = load_shapefile('road', index_field='ROADSEGID')
    if os.path.exists('/cache/road_ball_tree.pickle') and USE_CACHE:
        with open('/cache/road_ball_tree.pickle', 'rb') as f:
            tree, ids = pickle.load(f)
        return tree, ids, df
    else:
        roads = process_roads(df, USE_CACHE)
        points_dict = create_points_dict(roads)
        points = list(points_dict.keys())
        points_radians = [[np.deg2rad(point[0]), np.deg2rad(point[1])] for point in points]
        ids = list(points_dict.values())
        tree = BallTree([list(point) for point in points_radians], metric='haversine')
        with open('/cache/road_ball_tree.pickle', 'wb') as f:
            pickle.dump((tree, ids), f)
        return tree, ids, df


def main():
    if not (os.path.exists('/cache/zone.pickle') and os.path.exists('/cache/zone_ball_tree.pickle') and USE_CACHE):
        print("building zone data structures")
        create_zone_ball_tree()
    else:
        print("zone data structures already built")

    if not (os.path.exists('/cache/jurisdiction.pickle') and os.path.exists(
            '/cache/jurisdiction_ball_tree.pickle') and USE_CACHE):
        print("building jurisdiction data structures")
        create_jurisdiction_ball_tree()
    else:
        print("jurisdiction data structures already built")

    if not (os.path.exists('/cache/road.pickle') and os.path.exists('/cache/road_ball_tree.pickle') and USE_CACHE):
        print("building road data structures")
        create_road_ball_tree()
    else:
        print("road data structures already built")


if __name__ == '__main__':
    main()
