"""Vectorized 2D raycast over a binary line mask.

Port of the Unity C# Raycast.cs algorithm:
- origin: bottom-center of the mask (x = W/2, y = H-1)
- N rays span an FOV cone symmetric around the up direction (90° image-frame)
- each ray walks pixel-by-pixel; distance = number of steps until it hits a
  foreground pixel (line) OR exits the image bounds
- output: ndarray of shape (n_rays,) with int distances in pixels

The implementation is fully vectorized: all rays are advanced in parallel
using numpy boolean masks, so it scales linearly with the image diagonal,
not with N rays.
"""

from __future__ import annotations

import math

import numpy as np


def cast_rays(
    mask: np.ndarray,
    n_rays: int = 50,
    fov: float = 180.0,
    origin: tuple[int, int] | None = None,
) -> np.ndarray:
    """Cast n_rays rays from origin into the binary mask, return distances.

    Parameters
    ----------
    mask : ndarray, shape (H, W) or (H, W, C)
        Binary line mask. Foreground (line) is True/non-zero. If a 3-channel
        mask is passed, it is binarized as min(R,G,B) > 127.
    n_rays : int
        Number of rays to cast (1..50 in the team's protocol).
    fov : float
        Field of view in degrees (1..180), centered on the up direction.
    origin : (x, y), optional
        Ray origin in pixel coordinates. Defaults to bottom-center of the mask.

    Returns
    -------
    distances : ndarray of shape (n_rays,), dtype=int32
        For each ray, the number of pixel-sized steps until it hit a line
        pixel; or the maximum step count walked if it exited the image
        without a hit.
    """
    if not 1 <= n_rays:
        raise ValueError(f"n_rays must be >= 1, got {n_rays}")
    if not 1.0 <= fov <= 180.0:
        raise ValueError(f"fov must be in [1, 180] degrees, got {fov}")

    if mask.ndim == 3:
        mask_bin = mask.min(axis=-1) > 127
    else:
        mask_bin = mask.astype(bool)

    h, w = mask_bin.shape
    if origin is None:
        ox, oy = w / 2.0, h - 1.0
    else:
        ox, oy = float(origin[0]), float(origin[1])

    # Ray angles, mirroring the C# math:
    #   angle_k = k * step + (180 - fov) / 2     (in degrees, image frame)
    # → for fov=180, angles go from 0 to 180.
    # → for fov=60,  angles go from 60 to 120 (narrow forward cone).
    if n_rays == 1:
        angles_deg = np.array([90.0], dtype=np.float64)
    else:
        step = fov / (n_rays - 1)
        angles_deg = np.arange(n_rays, dtype=np.float64) * step + (180.0 - fov) / 2.0
    angles_rad = np.deg2rad(angles_deg)

    cos_a = np.cos(angles_rad)  # (n_rays,)
    sin_a = np.sin(angles_rad)

    # Walk all rays in lockstep. The longest possible walk is the image diagonal.
    max_steps = int(math.ceil(math.hypot(w, h)))

    # Distance accumulator — initialized to max_steps (the "no-hit" sentinel).
    distances = np.full(n_rays, max_steps, dtype=np.int32)
    active = np.ones(n_rays, dtype=bool)

    for step_idx in range(1, max_steps + 1):
        # x(k) = ox + step * cos(angle_k)
        # y(k) = oy - step * sin(angle_k)   (image y grows downward)
        xs = ox + step_idx * cos_a
        ys = oy - step_idx * sin_a

        ix = np.floor(xs).astype(np.int32)
        iy = np.floor(ys).astype(np.int32)

        in_bounds = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)

        # A ray that exited the image is finalized at the previous step count.
        exited = active & ~in_bounds
        if exited.any():
            distances[exited] = step_idx - 1
            active &= ~exited

        if not active.any():
            break

        # Sample mask only for still-active in-bounds rays.
        sample_idx = active & in_bounds
        if sample_idx.any():
            ix_a = ix[sample_idx]
            iy_a = iy[sample_idx]
            hits_in_active = mask_bin[iy_a, ix_a]
            # Map back to global ray indices.
            global_idx = np.flatnonzero(sample_idx)[hits_in_active]
            if global_idx.size:
                distances[global_idx] = step_idx
                active[global_idx] = False
                if not active.any():
                    break

    return distances


def draw_rays(
    mask: np.ndarray,
    distances: np.ndarray,
    n_rays: int,
    fov: float,
    origin: tuple[int, int] | None = None,
    ray_color: tuple[int, int, int] = (0, 0, 255),
    hit_color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Render a debug visualization: mask in white, ray paths in blue, hits in red.

    Useful for sanity-checking raycast output during development.
    """
    if mask.ndim == 2:
        canvas = np.stack([mask.astype(np.uint8) * 255] * 3, axis=-1)
    else:
        canvas = mask.copy()

    h, w = canvas.shape[:2]
    ox, oy = (w / 2.0, h - 1.0) if origin is None else (float(origin[0]), float(origin[1]))

    if n_rays == 1:
        angles_deg = np.array([90.0])
    else:
        step = fov / (n_rays - 1)
        angles_deg = np.arange(n_rays) * step + (180.0 - fov) / 2.0
    angles_rad = np.deg2rad(angles_deg)

    for k in range(n_rays):
        d = int(distances[k])
        for s in range(1, d):
            x = int(ox + s * math.cos(angles_rad[k]))
            y = int(oy - s * math.sin(angles_rad[k]))
            if 0 <= x < w and 0 <= y < h:
                canvas[y, x] = ray_color
        x_end = int(ox + d * math.cos(angles_rad[k]))
        y_end = int(oy - d * math.sin(angles_rad[k]))
        if 0 <= x_end < w and 0 <= y_end < h:
            canvas[y_end, x_end] = hit_color

    return canvas
