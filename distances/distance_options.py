from scipy.spatial.distance import euclidean
from math import radians, sin, cos, sqrt, asin


def calculate_distance(distance_type, point_1, point_2):
    if distance_type == 'euclidean':
        return euclidean(point_1, point_2)
    elif distance_type == 'manhattan':
        return abs(point_1.coordinate_x - point_2.coordinate_x) + abs(point_1.coordinate_y - point_2.coordinate_y)
    elif distance_type == 'haversine':
        long_1, lat_1, long_2, lat_2 = map(radians, [point_1.longitude_1, point_1.latitude_1, point_2.longitude_2,
                                                     point_2.latitude_2])
        a = sin(lat_2 - lat_1 / 2) ** 2 + cos(lat_1) * cos(lat_2) * sin(long_2 - long_1 / 2) ** 2
        return 6371 * 2 * asin(sqrt(a))
    elif distance_type == 'chebyshev':
        return max(abs(point_1.coordinate_x - point_2.coordinate_x), abs(point_1.coordinate_y - point_2.coordinate_y))
    return None
