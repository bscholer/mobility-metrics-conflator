import collections
from pydantic import BaseModel
from geojson_pydantic.features import FeatureCollection
import pandas as pd


def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


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


class LineRequestBody(BaseModel):
    line: str


class PointRequestBody(BaseModel):
    point: str


class PickupDropoffStat(BaseModel):
    street: int
    zone: int
    bin: str


class FlowStat(BaseModel):
    street: str
    zone: str
    bin: str


# TODO name this better
class StatMatch(BaseModel):
    """ the matches object that gets sent with requests to stat functions. The matching functions do not return all these fields. """
    zones: list[int]
    streets: list[int]
    bins: list[str]
    pickup: PickupDropoffStat
    dropoff: PickupDropoffStat
    flow: FlowStat

    def to_dict(self):
        return self.dict()


class MdsTripWithMatches(BaseModel):
    provider_name: str
    vehicle_id: str
    vehicle_type: str
    trip_id: str
    trip_distance: float
    trip_duration: float
    start_time: int
    end_time: int
    propulsion_types: list[str]
    matches: StatMatch

    def to_dict(self):
        return self.dict()


class TripBasedRequestBody(BaseModel):
    trips: list[MdsTripWithMatches]
    privacy_minimum: int = 0
