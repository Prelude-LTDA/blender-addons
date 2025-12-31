"""Inverse UV mapping functions for direction indicators."""

from __future__ import annotations

import math

from mathutils import Vector


def inverse_uv_planar(
    u: float,
    v: float,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
) -> Vector:
    """Convert UV coordinates back to 3D position for planar mapping.

    Forward chain: raw_uv -> tile -> rotate -> flip -> offset -> final_uv
    Reverse chain: final_uv -> un-offset -> un-flip -> un-rotate -> un-tile -> raw_uv
    Then: raw_uv -> 3D position

    Planar formula: U = x * 0.5, V = y * 0.5
    Inverse: x = U * 2, y = V * 2, z = 0
    """
    # 1. Remove offset
    u1 = u - u_offset
    v1 = v - v_offset

    # 2. Reverse flip (flip formula is: tile - value)
    if u_flip:
        u1 = u_tile - u1
    if v_flip:
        v1 = v_tile - v1

    # 3. Reverse rotation (apply negative rotation)
    cos_r = math.cos(-uv_rotation)
    sin_r = math.sin(-uv_rotation)
    u2 = u1 * cos_r - v1 * sin_r
    v2 = u1 * sin_r + v1 * cos_r

    # 4. Reverse tiling
    u_raw = u2 / u_tile if u_tile != 0 else u2
    v_raw = v2 / v_tile if v_tile != 0 else v2

    # 5. Reverse mapping formula (Planar: U = x * 0.5, V = y * 0.5)
    x = u_raw * 2.0
    y = v_raw * 2.0
    z = 0.0

    return Vector((x, y, z))


def inverse_uv_cylindrical(
    u: float,
    v: float,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
) -> Vector:
    """Convert UV coordinates back to 3D position for cylindrical mapping.

    The cylindrical mapping uses:
    - U: angle around Z axis (one full rotation per U unit, before tiling)
    - V: height along Z axis

    Internal scale: -4x on U (so U=-4 = one rotation), 0.75x on V
    """
    # Reverse UV processing chain
    u1 = u - u_offset
    v1 = v - v_offset

    if u_flip:
        u1 = u_tile - u1
    if v_flip:
        v1 = v_tile - v1

    cos_r = math.cos(-uv_rotation)
    sin_r = math.sin(-uv_rotation)
    u2 = u1 * cos_r - v1 * sin_r
    v2 = u1 * sin_r + v1 * cos_r

    u_raw = u2 / u_tile if u_tile != 0 else u2
    v_raw = v2 / v_tile if v_tile != 0 else v2

    # Convert to rotation fraction (0-1 = full rotation)
    # Internal scale is -4, so U_raw=1 means -0.25 rotations = +0.75 rotations
    u_frac = -u_raw / 4.0

    # Cylindrical coordinates
    # U fraction -> angle theta (0 to 2π for full rotation)
    theta = u_frac * 2.0 * math.pi

    x = math.sin(theta)
    y = math.cos(theta)
    # V has internal scale of 0.75, so reverse it
    z = v_raw / 0.75

    return Vector((x, y, z))


def inverse_uv_spherical(
    u: float,
    v: float,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
) -> Vector:
    """Convert UV coordinates back to 3D position for spherical mapping.

    The spherical mapping in nodes.py uses:
    - U: atan2(x,y) / 2π, then scaled by -4
    - V: (acos(z/length) / π - 0.5), then scaled by -2

    The -0.5 offset centers V=0 at the equator:
    - V=0 → equator (z=0)
    - V>0 → northern hemisphere
    - V<0 → southern hemisphere

    Inverse:
        atan2(x,y) = U_output * π / (-2) = -U_output * π / 2
        acos(z_norm) = (V_output / (-2) + 0.5) * π = -V_output * π / 2 + π/2
        z_norm = cos(-V_output * π / 2 + π/2) = sin(V_output * π / 2)
    """
    # Reverse UV processing chain
    u1 = u - u_offset
    v1 = v - v_offset

    if u_flip:
        u1 = u_tile - u1
    if v_flip:
        v1 = v_tile - v1

    cos_r = math.cos(-uv_rotation)
    sin_r = math.sin(-uv_rotation)
    u2 = u1 * cos_r - v1 * sin_r
    v2 = u1 * sin_r + v1 * cos_r

    u_raw = u2 / u_tile if u_tile != 0 else u2
    v_raw = v2 / v_tile if v_tile != 0 else v2

    # Reverse the spherical mapping
    # atan2(x,y) = -u_raw * π / 2 (after accounting for -4x scale)
    phi = -u_raw * math.pi / 2.0

    # With equator offset: V_raw = (-2) * (acos(z_norm)/π - 0.5)
    # Solving: acos(z_norm) = π * (0.5 - V_raw/2)
    # z_norm = cos(π * (0.5 - V_raw/2)) = cos(π/2 - V_raw*π/2) = sin(V_raw*π/2)
    # theta (polar angle from +Z) = acos(z_norm) = π/2 - V_raw*π/2
    theta = math.pi / 2.0 - v_raw * math.pi / 2.0
    theta = max(0.0, min(math.pi, theta))  # Clamp to valid range

    # The theta here is the polar angle from +Z axis
    # sin(theta) gives the xy-plane distance, cos(theta) gives z
    z = math.cos(theta)
    xy_radius = math.sin(theta)

    # phi is atan2(x, y), so x = r*sin(phi), y = r*cos(phi)
    x = xy_radius * math.sin(phi)
    y = xy_radius * math.cos(phi)

    return Vector((x, y, z))


def inverse_uv_shrink_wrap(
    u: float,
    v: float,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
) -> Vector:
    """Convert UV coordinates back to 3D position for shrink wrap mapping.

    Shrink wrap (azimuthal equidistant) formula:
    theta = acos(z / length)
    phi = atan2(y, x)
    r = theta / π
    U = r * cos(phi), V = r * sin(phi)

    Inverse:
    r = sqrt(U² + V²)
    phi = atan2(V, U)
    theta = r * π
    x = sin(theta) * cos(phi), y = sin(theta) * sin(phi), z = cos(theta)
    """
    # Reverse UV processing chain
    u1 = u - u_offset
    v1 = v - v_offset

    if u_flip:
        u1 = u_tile - u1
    if v_flip:
        v1 = v_tile - v1

    cos_r = math.cos(-uv_rotation)
    sin_r = math.sin(-uv_rotation)
    u2 = u1 * cos_r - v1 * sin_r
    v2 = u1 * sin_r + v1 * cos_r

    u_raw = u2 / u_tile if u_tile != 0 else u2
    v_raw = v2 / v_tile if v_tile != 0 else v2

    # Reverse shrink wrap mapping
    r = math.sqrt(u_raw * u_raw + v_raw * v_raw)

    if r < 1e-6:
        return Vector((0.0, 0.0, 1.0))

    phi = math.atan2(v_raw, u_raw)
    theta = r * math.pi

    # Clamp theta
    theta = min(theta, math.pi)

    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)

    return Vector((x, y, z))
