import json
import os.path
import time

import shapely
from shapely.geometry import LineString, Point
from tqdm import tqdm
import numpy as np
import pandas as pd

import geopandas

tqdm.pandas()

UTM_ZONE_EPSG_CODE = "32611"
INTERPOLATION_DISTANCE = 5  # the distance between interpolated points in meters
LINE_PAD_DISTANCE = 0.1 # how much to pad the line by to make sure we don't get duplicate points at intersections
MIN_LINE_LENGTH = 1 # meters
USE_CACHE = False
USE_SUBSET = False
WRITE_DEBUG_SHAPEFILES = False
xmin = -117.186
ymin = 32.692
xmax = -117.129
ymax = 32.741


def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


def split_line(row):
    line = row['geometry']
    # pad the line to avoid duplicate points
    target_length = line.length - LINE_PAD_DISTANCE * 2
    distances = np.arange(LINE_PAD_DISTANCE, target_length, INTERPOLATION_DISTANCE)
    distances = np.append(distances, [target_length - LINE_PAD_DISTANCE])
    if len(distances) < 2:
        # print geojson of line
        print(row.name)
        print(line.length, target_length, distances)
    points = [line.interpolate(distance) for distance in distances] # + [line.interpolate(target_length - LINE_PAD_DISTANCE)]
    return LineString(points)


# Load the data, convert to UTM for shapely, then interpolate points
def load_roads_from_file():
    if os.path.exists('cache/final.pickle') and USE_CACHE:
        return pd.read_pickle('cache/final.pickle')

    if os.path.exists('cache/ref.pickle'):
        print('loading roads from pickle')
        df = pd.read_pickle('cache/ref.pickle')
    else:
        start = time.process_time()
        df = geopandas.read_file('data/ref/ref.shp')
        df = df.to_crs('EPSG:4326')
        df = df.set_index('ROADSEGID')
        df.to_pickle('cache/ref.pickle')
        print('loaded geojson in {} seconds'.format(time.process_time() - start))

    if USE_SUBSET:
        df = df.cx[xmin:xmax, ymin:ymax]
        print('subset to {} roads'.format(len(df)))

    total_points_count = df['geometry'].apply(lambda x: len(x.coords)).sum()
    print('total points: {}'.format(total_points_count))

    if os.path.exists('cache/ref_utm.pickle') and USE_CACHE:
        print('loading utm roads from pickle')
        df = pd.read_pickle('cache/ref_utm.pickle')
    else:
        start = time.process_time()
        df = df.to_crs(UTM_ZONE_EPSG_CODE)
        df.to_pickle('cache/ref_utm.pickle')
        print('converted to UTM in {} seconds'.format(time.process_time() - start))

    # remove roads that are too short and show how many were removed
    orig_len = df.shape[0]
    df = df[df['geometry'].apply(lambda x: x.length) > MIN_LINE_LENGTH]
    print('removed {} roads that were too short'.format(orig_len - df.shape[0]))

    # make points every INTERPOLATION_DISTANCE meters evenly along the line
    if os.path.exists('cache/ref_utm_split.pickle') and USE_CACHE:
        print('loading split utm roads from pickle')
        df = pd.read_pickle('cache/ref_utm_split.pickle')
    else:
        start = time.process_time()
        df['geometry'] = df.progress_apply(lambda row: split_line(row), axis=1)
        df.to_pickle('cache/ref_utm_split.pickle')
        print('interpolated points in {} seconds'.format(time.process_time() - start))

    total_points_count = df['geometry'].apply(lambda x: len(x.coords)).sum()
    print('total points: {}'.format(total_points_count))

    # reproject to wgs84 for geojson
    df = df.to_crs('EPSG:4326')

    # write to shapefile
    if WRITE_DEBUG_SHAPEFILES:
        df.drop(columns=['POSTDATE', 'ADDSEGDT']).to_file('data/ref_utm_split/shp.shp')

    points_dict = {}
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # if len(points_dict) > 5569564:
        #     print(index, row['geometry'], len(points_dict))
        for point in row['geometry'].coords:
            if point not in points_dict:
                points_dict[point] = []
            points_dict[point].append(index)
    # get just points that have multiple ids
    dup_points_dict = {k: v for k, v in points_dict.items() if len(v) > 1}
    print('found {} duplicate points'.format(len(dup_points_dict)))
    # remove coords from lines that are duplicates
    # for index, row in tqdm(df.iterrows(), total=len(df), desc='removing duplicate points'):
    #     orig_len = len(row['geometry'].coords)
    #     coords = row['geometry'].coords
    #     coords = [c for c in coords if c not in dup_points_dict]
    #     print('removed {} points from road {}'.format(orig_len - len(coords), index))
    #     if len(coords) < 2:
    #         print(index, coords, row['LENGTH'])
    #     else:
    #         df.at[index, 'geometry'] = LineString(coords)
    # write to shapefile
    if WRITE_DEBUG_SHAPEFILES:
        df.drop(columns=['POSTDATE', 'ADDSEGDT']).to_file('data/ref_utm_split_no_dup/shp.shp')


    # use the keys as geometries and make a multipoint geometry but make points with them
    # with open('data/dup_points.geojson', 'w') as f:
    #     f.write(geopandas.GeoSeries(map(lambda p: Point(p), dup_points_dict.keys())).to_json())

    df.to_pickle('cache/final.pickle')
    return df
    # data = json.load(f)
    # with open('cache/ref.pickle', 'wb') as f:
    #     pickle.dump(data, f)
    # return data
