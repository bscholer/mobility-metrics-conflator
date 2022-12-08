import collections
from typing import Literal, Dict, Optional

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
    line: Optional[str]
    lines: Optional[Dict[str, str]]


class PointRequestBody(BaseModel):
    point: Optional[str]
    points: Optional[Dict[str, str]]


class PickupDropoffStat(BaseModel):
    street: int
    zone: int
    jurisdiction: int
    bin: str


class FlowStat(BaseModel):
    street: str
    zone: str
    jurisdiction: str
    bin: str


# TODO name this better
class StatMatch(BaseModel):
    """ the matches object that gets sent with requests to stat functions. The matching functions do not return all these fields. """
    zone: list[int]
    jurisdiction: list[int]
    street: list[int]
    bin: list[str]
    pickup: PickupDropoffStat
    dropoff: PickupDropoffStat
    flow: FlowStat

    def to_dict(self):
        return self.dict()


class StatusChangeMatch(BaseModel):
    zone: int
    jurisdiction: int
    street: int
    bin: str

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


class MdsStatusChangeWithMatches(BaseModel):
    provider_name: str
    vehicle_id: str
    vehicle_type: str
    propulsion_types: list[str]
    vehicle_state: Literal['removed', 'available', 'non_operational', 'reserved', 'on_trip', 'elsewhere', 'unknown']
    event_time: int
    matches: StatusChangeMatch

    def to_dict(self):
        return self.dict()


class TripBasedRequestBody(BaseModel):
    report_date: str  # YYYY-MM-DD
    trips: list[MdsTripWithMatches]
    privacy_minimum: int = 0


class StatusChangeBasedRequestBody(BaseModel):
    report_date: str  # YYYY-MM-DD
    status_changes: list[MdsStatusChangeWithMatches]
    privacy_minimum: int = 0
