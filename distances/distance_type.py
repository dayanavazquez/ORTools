from math import radians, sin, cos, sqrt, asin, hypot
from enum import Enum


class DistanceType(Enum):
    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'
    HAVERSINE = 'haversine'
    CHEBYSHEV = 'chebyshev'


def calculate_distance(point_1, point_2, distance_type=None):
    if distance_type == DistanceType.EUCLIDEAN:
        return hypot((point_1[0] - point_2[0]), (point_1[1] - point_2[1]))
    if distance_type == DistanceType.MANHATTAN or not distance_type:
        return abs(point_1[0] - point_2[0]) + abs(point_1[1] - point_2[1])
    if distance_type == DistanceType.HAVERSINE:
        long_1, lat_1, long_2, lat_2 = map(radians, [point_1.longitude_1, point_1.latitude_1, point_2.longitude_2,
                                                     point_2.latitude_2])
        a = sin(lat_2 - lat_1 / 2) ** 2 + cos(lat_1) * cos(lat_2) * sin(long_2 - long_1 / 2) ** 2
        return 6371 * 2 * asin(sqrt(a))
    if distance_type == DistanceType.CHEBYSHEV:
        return max(abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))
    return "The distance type is not supported"
