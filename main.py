import argparse
import json
import os
import pickle
import time
from typing import Union

import geopandas
import pandas as pd
import numpy as np
import shapely
from fastapi import FastAPI, Query
from pydantic import BaseModel
from shapely.geometry import Point
from sklearn.neighbors import BallTree
from tqdm import tqdm

from interpolate import load_roads_from_file

USE_CACHE = True

EARTH_RADIUS = 6371008  # meters

tqdm.pandas()

tree, ids, ref_df, zones_tree, zones_ids, zones_df = None, None, None, None, None, None


class LineRequestBody(BaseModel):
    line: str


class PointRequestBody(BaseModel):
    point: str


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


def load_shapefile():
    if os.path.exists('/cache/ref.pickle'):
        print('loading ref_df from cache')
        print('loading roads from pickle')
        df = pd.read_pickle('/cache/ref.pickle')
    else:
        print('loading ref_df from shapefile')
        start = time.process_time()
        df = geopandas.read_file('/data/ref/ref.shp')
        df = df.to_crs('EPSG:4326')
        df.to_pickle('/cache/ref.pickle')
        print('loaded shapefile in {} seconds'.format(time.process_time() - start))
    return df


def load_zones_shapefile():
    if os.path.exists('/cache/zones.pickle'):
        print('loading zones from pickle')
        df = pd.read_pickle('/cache/zones.pickle')
    else:
        print('loading zones from shapefile')
        start = time.process_time()
        df = geopandas.read_file('/data/zones/zones.shp')
        df = df.to_crs('EPSG:4326')
        df.to_pickle('/cache/zones.pickle')
        print('loaded zones shapefile in {} seconds'.format(time.process_time() - start))
    return df


def create_zone_ball_tree():
    zones_df = load_zones_shapefile()
    if os.path.exists('/data/zone_ball_tree.pickle') and os.path.exists(
            '/data/zone_ball_tree_ids.pickle') and USE_CACHE:
        with open('/data/zone_ball_tree.pickle', 'rb') as f:
            tree = pickle.load(f)
        with open('/data/zone_ball_tree_ids.pickle', 'rb') as f:
            ids = pickle.load(f)
        return tree, ids, zones_df
    else:
        # find center of zones
        zones_df['center'] = zones_df['geometry'].progress_apply(lambda geom: Point(geom.centroid))
        zones_df = zones_df.drop(columns=['geometry'])
        # print(zones_df['center'].dtype)
        zones_df.set_geometry(zones_df['center'], inplace=True)

        # print(zones_df['geometry'])
        # with open('/data/zones_centers.geojson', 'w') as f:
        #     f.write(zones_df.drop(columns=['center']).to_json())
        # convert to radians
        zones_df['center'] = zones_df['center'].progress_apply(lambda point: [np.deg2rad(point.x), np.deg2rad(point.y)])
        # create ball tree
        ids = zones_df['id'].tolist()
        tree = BallTree([list(point) for point in zones_df['center']], metric='haversine', leaf_size=2)
        with open('/data/zone_ball_tree.pickle', 'wb') as f:
            pickle.dump(tree, f)
        with open('/data/zone_ball_tree_ids.pickle', 'wb') as f:
            pickle.dump(ids, f)
        return tree, ids, zones_df


def create_ball_tree(leaf_size=40):
    if os.path.exists('/data/ball_tree.pickle') and os.path.exists('/data/ball_tree_ids.pickle') and USE_CACHE:
        with open('/data/ball_tree.pickle', 'rb') as f:
            tree = pickle.load(f)
        with open('/data/ball_tree_ids.pickle', 'rb') as f:
            ids = pickle.load(f)
        return tree, ids
    else:
        roads = load_roads_from_file()
        points_dict = create_points_dict(roads)
        points = list(points_dict.keys())
        points_radians = [[np.deg2rad(point[0]), np.deg2rad(point[1])] for point in points]
        ids = list(points_dict.values())
        tree = BallTree([list(point) for point in points_radians], metric='haversine', leaf_size=leaf_size)
        with open('/data/ball_tree.pickle', 'wb') as f:
            pickle.dump(tree, f)
        with open('/data/ball_tree_ids.pickle', 'wb') as f:
            pickle.dump(ids, f)
        return tree, ids


def load_mds_from_file(file):
    with open(file, 'r') as f:
        return json.load(f)
    # df = pd.read_csv('data/mds-trips.csv')
    # load geom column as wkt to geodataframe
    # df['geom'] = df['geom'].progress_apply(lambda wkt: shapely.wkt.loads(wkt))
    # return df


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


def main(file):
    with open('config.json') as f:
        config = json.load(f)
        # print(requests.get(config['roads_reference_url']).text)
        # start = time.process_time()
        # with open('cache/ref_test.zip', 'wb') as f:
        #     f.write(requests.get(config['roads_reference_url']).content)
        # print('downloaded roads reference file')
        # print('downloaded roads reference file in {} seconds'.format(time.process_time() - start))

        parser = argparse.ArgumentParser()

        leaf_size = 40
        start = time.process_time()
        tree, ids = create_ball_tree()
        print('loaded roads in {} seconds'.format(time.process_time() - start))

        mds_df = load_mds_from_file(file)
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
        # mds_df['x'] = mds_df['geom'].apply(lambda p: p.x)
        # mds_df['y'] = mds_df['geom'].apply(lambda p: p.y)
        # match idx to the indicies in ids
        mds_df['ROADSEGID'] = mds_df['idx'].apply(lambda idx: ids[idx][0])

        # mds_df.drop(columns=['distance', 'geom_rad', 'geom', 'idx']).to_csv('data/mds-trips-with-roadsegid.csv'.format(leaf_size), index=False)

        mds_df.drop(columns=['distance', 'distance_meters', 'geom_rad', 'geom', 'idx']).to_csv(
            'data/mds-trips-with-roadsegid.csv', index=False)

        idx_of_max_distance = mds_df['distance'].argmax()
        max_distance = distances[idx_of_max_distance]
        max_distance_id = ids[indices[idx_of_max_distance][0]]
        print(points[idx_of_max_distance], max_distance, max_distance_id)


def find_matching_zone_brute_force(point, zones_df):
    """This is CRAZY slow, and should only be used as a last resort."""
    print('BRUTE FORCE')
    zone = zones_df[zones_df['geometry'].contains(point)]['id'].values
    return zone[0] if len(zone) > 0 else None


def find_matching_zones(points, tree, ids, df):
    """Find the matching zones for the given points. Start with a small ball tree search, and if no zones are found, increase the search radius, and if that doesn't work, brute force"""
    """We use the ball tree to find the closest centroid to narrow the search space, and then use the shapely contains function to find the matching zone"""
    start = time.process_time()
    points_radians = [[np.deg2rad(point[0]), np.deg2rad(point[1])] for point in points]

    # query for the 5 closest zones
    zone_indices = tree.query(points_radians, k=5, return_distance=False)
    # mash em together and find the ids
    zones = [ids[z] for sub_group in zone_indices for z in sub_group]
    unique_zones = list(set(zones))

    # matching every point against the union of all closest zones is significantly faster than building a new gdf for each point
    matching_zones_df = df[df['id'].isin(unique_zones)]
    # print('matched zones in {} seconds'.format(time.process_time() - start))

    zones = []
    for i in range(len(points)):
        point = points[i]
        point = shapely.geometry.Point(point)
        matching_zones = matching_zones_df[matching_zones_df['geometry'].contains(point)]['id'].values
        if len(matching_zones) > 0:
            zone = matching_zones[0]
        else:
            print('no zone found in initial query, expanding search')
            # if no zone is found, expand the search radius
            zone_indexes = tree.query([points_radians[i]], k=100, return_distance=False)
            # mash em together and find the ids
            zones_for_point = [zones_ids[z] for sub_group in zone_indexes for z in sub_group]
            matching_zones_for_point_df = df[df['id'].isin(zones_for_point)]
            matching_zones = matching_zones_for_point_df[matching_zones_for_point_df['geometry'].contains(point)]['id'].values
            if len(matching_zones) > 0:
                zone = matching_zones[0]
            else:
                print('no zone found in expanded query, finding closest zone')
                # if we still can't find a zone, find the closest zone with brute force
                zone = find_matching_zone_brute_force(point, matching_zones_for_point_df)
                # if still no zone is found, find the closest zone

        # print('matches', [zones_ids[z] for z in zone_indices[i]])
        if zone:
            zones.append(zone)
        else:
            print('the point {} is not in any zone'.format(point.wkt))
    zones = [int(zone) for zone in set(zones)]
    print('matched {} points to {} zones in {} seconds'.format(len(points), len(zones), time.process_time() - start))
    return zones



if tree is None or ids is None:
    tree, ids = create_ball_tree()
if ref_df is None:
    ref_df = load_shapefile()
if zones_tree is None or zones_ids is None or zones_df is None:
    zones_tree, zones_ids, zones_df = create_zone_ball_tree()

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/match/")
async def match(lat: float, lon: float):
    global tree, ids
    if tree is None or ids is None:
        tree, ids = create_ball_tree()

    point = [np.deg2rad(lat), np.deg2rad(lon)]

    distance, index = tree.query([point], k=1)
    # print('leaf size {}: {} seconds'.format(time.process_time() - start))
    distance_meters = distance * EARTH_RADIUS
    # match idx to the indicies in ids
    roadsegid = ids[index[0][0]][0]
    return {
        'distance_meters': distance_meters[0][0],
        'roadsegid': roadsegid
    }


@app.post("/match_line/")
async def match_line(body: LineRequestBody):
    global tree, ids, ref_df, zones_tree, zones_ids, zones_df
    if tree is None or ids is None:
        tree, ids = create_ball_tree()
    if ref_df is None:
        ref_df = load_shapefile()
    if zones_tree is None or zones_ids is None or zones_df is None:
        zones_tree, zones_ids, zones_df = create_zone_ball_tree()

    line = shapely.wkt.loads(body.line)
    points = list(line.coords)
    if len(points) > 200:
        print(body.line)

    start = time.process_time()
    points_radians = [[np.deg2rad(point[0]), np.deg2rad(point[1])] for point in points]
    distances, indices = tree.query(points_radians, k=1)
    # print('leaf size {}: {} seconds'.format(time.process_time() - start))

    # distances_meters = [dist[0] * EARTH_RADIUS for dist in distances]
    # match idx to the indicies in ids
    roadsegids = [ids[index[0]][0] for index in indices]
    # remove consecutive duplicates
    roadsegids = [roadsegids[i] for i in range(len(roadsegids)) if i == 0 or roadsegids[i] != roadsegids[i - 1]]
    geometries = ref_df[ref_df['ROADSEGID'].isin(roadsegids)][['ROADSEGID', 'geometry']]
    # print('matched road segments in {} seconds'.format(time.process_time() - start))

    zones = find_matching_zones(points, zones_tree, zones_ids, zones_df)

    return {
        'streets': {
            'roadsegid': roadsegids,
            'geometry': geometries.to_json()
        },
        'zones': {
            'zoneid': zones,
            # 'geometry': zones_df[zones_df['id'].isin(zones)][['id', 'geometry']].to_json()
        }
    }


@app.post("/match_point/")
async def match_line(body: PointRequestBody):
    global tree, ids, ref_df, zones_tree, zones_ids, zones_df
    if tree is None or ids is None:
        tree, ids = create_ball_tree()
    if ref_df is None:
        ref_df = load_shapefile()
    if zones_tree is None or zones_ids is None or zones_df is None:
        zones_tree, zones_ids, zones_df = create_zone_ball_tree()

    point = shapely.wkt.loads(body.point)
    point_radians = [np.deg2rad(point.x), np.deg2rad(point.y)]

    distance, index = tree.query([point_radians], k=1)
    # print('leaf size {}: {} seconds'.format(time.process_time() - start))

    # distances_meters = [dist[0] * EARTH_RADIUS for dist in distances]
    # match index to the indicies in ids
    print(index)
    roadsegid = ids[index[0][0]][0]
    print(roadsegid)

    geometry = ref_df[ref_df['ROADSEGID'] == roadsegid][['ROADSEGID', 'geometry']]

    # zone_distance, zone_indices = zones_tree.query([point_radians], k=15)
    # potential_zones = [zones_ids[index] for index in zone_indices[0]]
    # print(potential_zones)
    #
    # # find zone
    # point = shapely.geometry.Point(point.x, point.y)
    # # print(zones_df[zones_df['geometry'].contains(point)]['id'].values)
    # if len(zones_df[zones_df['geometry'].contains(point)]['id'].values):
    #     zone = zones_df[zones_df['geometry'].contains(point)]['id'].values[0]
    # else:
    #     zone = None
    #     print('no zone found for point', point.wkt)
    # print(zone)
    point_coords = [point.x, point.y]
    print(point_coords)
    zones = find_matching_zones([point_coords], zones_tree, zones_ids, zones_df)
    return {
        'streets': {
            'roadsegid': roadsegid,
            'geometry': geometry.to_json()
        },
        'zones': {
            'zoneid': zones[0] if len(zones) else None,
            # 'geometry': zones_df[zones_df['id'] == zone]['geometry'].to_json()
        }
    }

# if __name__ == '__main__':
# main()
