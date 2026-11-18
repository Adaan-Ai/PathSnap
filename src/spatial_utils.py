import math
import numpy as np

# Centroid of Darmstadt dataset region
REF_LON = 8.58
REF_LAT = 49.85
EARTH_RADIUS = 6371000.0  # meters

def to_xy(lon, lat):
    """
    Project angular (lon, lat) to local 2D Cartesian plane in meters.
    Accurate for regional scale (error < 0.01% within 50km).
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    ref_lat_rad = math.radians(REF_LAT)
    ref_lon_rad = math.radians(REF_LON)
    
    x = (lon_rad - ref_lon_rad) * np.cos((lat_rad + ref_lat_rad) / 2.0) * EARTH_RADIUS
    y = (lat_rad - ref_lat_rad) * EARTH_RADIUS
    return x, y

def to_lonlat(x, y):
    """
    Inverse projection from local Cartesian (x, y) back to (lon, lat).
    """
    ref_lat_rad = math.radians(REF_LAT)
    ref_lon_rad = math.radians(REF_LON)
    
    lat_rad = (y / EARTH_RADIUS) + ref_lat_rad
    lon_rad = (x / (EARTH_RADIUS * math.cos((lat_rad + ref_lat_rad) / 2.0))) + ref_lon_rad
    return np.degrees(lon_rad), np.degrees(lat_rad)

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Computes great-circle distance between two GPS points in meters.
    """
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS * c

def point_to_segment_distance(px, py, ax, ay, bx, by):
    """
    Calculate perpendicular distance from point P to line segment AB,
    along with projection factor t in [0, 1].
    """
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    
    if seg_len_sq == 0:
        return math.hypot(px - ax, py - ay), 0.0, ax, ay
        
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t_clamped = max(0.0, min(1.0, t))
    
    proj_x = ax + t_clamped * dx
    proj_y = ay + t_clamped * dy
    dist = math.hypot(px - proj_x, py - proj_y)
    return dist, t_clamped, proj_x, proj_y

def calculate_heading(x1, y1, x2, y2):
    """
    Computes compass heading angle in degrees [0, 360) from (x1, y1) to (x2, y2).
    """
    angle = math.degrees(math.atan2(x2 - x1, y2 - y1))
    return (angle + 360.0) % 360.0

def angle_difference(angle1, angle2):
    """
    Minimum angular difference between two headings in degrees [0, 180].
    """
    diff = abs(angle1 - angle2) % 360.0
    return 360.0 - diff if diff > 180.0 else diff
