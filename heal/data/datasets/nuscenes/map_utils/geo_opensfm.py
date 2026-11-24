# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Tuple

import numpy as np
from numpy import ndarray

WGS84_a = 6378137.0
WGS84_b = 6356752.314245


def ecef_from_lla(lat, lon, alt: float) -> Tuple[float, ...]:
    """
    Compute ECEF XYZ from latitude, longitude and altitude.

    All using the WGS84 model.
    Altitude is the distance to the WGS84 ellipsoid.
    Check results here http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm

    >>> lat, lon, alt = 10, 20, 30
    >>> x, y, z = ecef_from_lla(lat, lon, alt)
    >>> np.allclose(lla_from_ecef(x,y,z), [lat, lon, alt])
    True
    """
    a_sq = WGS84_a**2
    b_sq = WGS84_b**2
    phi = np.radians(lat)
    lam = np.radians(lon)
    # Keep the same floating-point association as the prior implementation.
    radius_scale = 1.0 / np.sqrt(
        a_sq * np.cos(phi) ** 2 + b_sq * np.sin(phi) ** 2
    )
    xy_factor = a_sq * radius_scale + alt
    z_factor = b_sq * radius_scale + alt
    cos_phi = np.cos(phi)
    return (
        xy_factor * cos_phi * np.cos(lam),
        xy_factor * cos_phi * np.sin(lam),
        z_factor * np.sin(phi),
    )


def lla_from_ecef(x, y, z):
    """Convert ECEF XYZ coordinates back to latitude, longitude, altitude.

    Uses the WGS84 ellipsoid; altitude is measured from the ellipsoid surface.
    """
    semi_major = WGS84_a
    semi_minor = WGS84_b
    ecc_first = np.sqrt((semi_major**2 - semi_minor**2) / semi_major**2)
    ecc_second = np.sqrt((semi_major**2 - semi_minor**2) / semi_minor**2)
    p_xy = np.sqrt(x**2 + y**2)
    aux_angle = np.arctan2(z * semi_major, p_xy * semi_minor)
    longitude = np.arctan2(y, x)
    latitude = np.arctan2(
        z + ecc_second**2 * semi_minor * np.sin(aux_angle) ** 3,
        p_xy - ecc_first**2 * semi_major * np.cos(aux_angle) ** 3,
    )
    prime_vertical_n = semi_major / np.sqrt(
        1 - ecc_first**2 * np.sin(latitude) ** 2
    )
    altitude = p_xy / np.cos(latitude) - prime_vertical_n
    return np.degrees(latitude), np.degrees(longitude), altitude


def ecef_from_topocentric_transform(lat, lon, alt: float) -> ndarray:
    """
    Build the 4x4 transform from a local topocentric frame into ECEF.

    Origin sits at (lat, lon, alt). Local axes: X east, Y north, Z up
    (normal to the WGS84 ellipsoid).
    >>> a = ecef_from_topocentric_transform(30, 20, 10)
    >>> b = ecef_from_topocentric_transform_finite_diff(30, 20, 10)
    >>> np.allclose(a, b)
    True
    """
    ox, oy, oz = ecef_from_lla(lat, lon, alt)
    phi = np.radians(lat)
    lam = np.radians(lon)
    s_phi = np.sin(phi)
    c_phi = np.cos(phi)
    s_lam = np.sin(lam)
    c_lam = np.cos(lam)
    return np.array(
        [
            [-s_lam, -s_phi * c_lam, c_phi * c_lam, ox],
            [c_lam, -s_phi * s_lam, c_phi * s_lam, oy],
            [0, c_phi, s_phi, oz],
            [0, 0, 0, 1],
        ]
    )


def _unit_central_diff(plus_xyz, minus_xyz, step):
    """Normalize (plus - minus) / (2 * step)."""
    axis = (np.array(plus_xyz) - np.array(minus_xyz)) / 2 / step
    return axis / np.linalg.norm(axis)


def ecef_from_topocentric_transform_finite_diff(
    lat, lon, alt: float
) -> ndarray:
    """
    Transform from a topocentric frame at reference position to ECEF.

    The topocentric reference frame is a metric one with the origin
    at the given (lat, lon, alt) position, with the X axis heading east,
    the Y axis heading north and the Z axis vertical to the ellipsoid.
    """
    eps = 1e-2
    ox, oy, oz = ecef_from_lla(lat, lon, alt)
    # East / north / up via central differences on LLA.
    east = _unit_central_diff(
        ecef_from_lla(lat, lon + eps, alt),
        ecef_from_lla(lat, lon - eps, alt),
        eps,
    )
    north = _unit_central_diff(
        ecef_from_lla(lat + eps, lon, alt),
        ecef_from_lla(lat - eps, lon, alt),
        eps,
    )
    up = _unit_central_diff(
        ecef_from_lla(lat, lon, alt + eps),
        ecef_from_lla(lat, lon, alt - eps),
        eps,
    )
    return np.array(
        [
            [east[0], north[0], up[0], ox],
            [east[1], north[1], up[1], oy],
            [east[2], north[2], up[2], oz],
            [0, 0, 0, 1],
        ]
    )


def _affine_apply(mat, x, y, z):
    """Apply 4x4 affine transform to a 3D point (ignore homogeneous w)."""
    px = mat[0, 0] * x + mat[0, 1] * y + mat[0, 2] * z + mat[0, 3]
    py = mat[1, 0] * x + mat[1, 1] * y + mat[1, 2] * z + mat[1, 3]
    pz = mat[2, 0] * x + mat[2, 1] * y + mat[2, 2] * z + mat[2, 3]
    return px, py, pz


def topocentric_from_lla(lat, lon, alt: float, reflat, reflon, refalt: float):
    """
    Transform from lat, lon, alt to topocentric XYZ.

    >>> lat, lon, alt = -10, 20, 100
    >>> np.allclose(topocentric_from_lla(lat, lon, alt, lat, lon, alt),
    ...     [0,0,0])
    True
    >>> x, y, z = topocentric_from_lla(lat, lon, alt, 0, 0, 0)
    >>> np.allclose(lla_from_topocentric(x, y, z, 0, 0, 0),
    ...     [lat, lon, alt])
    True
    """
    inv_t = np.linalg.inv(
        ecef_from_topocentric_transform(reflat, reflon, refalt)
    )
    ex, ey, ez = ecef_from_lla(lat, lon, alt)
    return _affine_apply(inv_t, ex, ey, ez)


def lla_from_topocentric(x, y, z, reflat, reflon, refalt: float):
    """Transform from topocentric XYZ to lat, lon, alt."""

    t_mat = ecef_from_topocentric_transform(reflat, reflon, refalt)
    ex, ey, ez = _affine_apply(t_mat, x, y, z)
    return lla_from_ecef(ex, ey, ez)


class TopocentricConverter(object):
    """Convert to and from a topocentric reference frame."""

    def __init__(self, reflat, reflon, refalt):
        """Init the converter given the reference origin."""
        self.lat = reflat
        self.lon = reflon
        self.alt = refalt

    def to_topocentric(self, lat, lon, alt):
        """Convert lat, lon, alt to topocentric x, y, z."""
        return topocentric_from_lla(
            lat, lon, alt, self.lat, self.lon, self.alt
        )

    def to_lla(self, x, y, z):
        """Convert topocentric x, y, z to lat, lon, alt."""
        return lla_from_topocentric(x, y, z, self.lat, self.lon, self.alt)

    def __eq__(self, o):
        return np.allclose(
            [self.lat, self.lon, self.alt], (o.lat, o.lon, o.alt)
        )
