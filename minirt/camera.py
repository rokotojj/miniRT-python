"""Камера: ортонормированный базис и первичные лучи через пиксели.

Картинная плоскость стоит на расстоянии 1 перед камерой, её половина ширины —
tan(fov / 2), половина высоты — tan(fov / 2) · H / W. Пикселю (i, j) соответствуют
u = (2 (i + 0,5) / W − 1) · tan(fov / 2) и v = (1 − 2 (j + 0,5) / H) · tan(fov / 2) · H / W,
направление луча — normalize(forward + u · right + v · up).
"""

import math

import numpy as np

from minirt.scene import Camera, Vec3
from minirt.vec import FloatArray, cross, norm, normalize, vec3

WORLD_UP = vec3((0.0, 1.0, 0.0))
# Если взгляд направлен строго вверх или вниз, «верхом» мира служит ось z.
FALLBACK_UP = vec3((0.0, 0.0, 1.0))
_PARALLEL = 1e-9


def camera_basis(direction: Vec3) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Базис камеры (forward, right, up) по вектору взгляда.

    forward — нормированный взгляд, right = normalize(up_мира × forward), up = forward × right.
    """
    forward = normalize(vec3(direction))
    right = cross(WORLD_UP, forward)
    if norm(right) < _PARALLEL:
        right = cross(FALLBACK_UP, forward)
    right = normalize(right)
    up = cross(forward, right)
    return forward, right, up


def ray_directions(
    camera: Camera,
    width: int,
    height: int,
    *,
    offset: tuple[float, float] = (0.5, 0.5),
    rows: tuple[int, int] | None = None,
) -> FloatArray:
    """Единичные направления лучей через пиксели — массив формы (H, W, 3).

    offset — положение точки внутри пикселя по x и y (0.5, 0.5 — центр);
    rows — полуинтервал строк [начало, конец), если нужен не весь кадр.
    """
    forward, right, up = camera_basis(camera.direction)
    half_width = math.tan(math.radians(camera.fov) / 2.0)
    half_height = half_width * height / width
    first, last = (0, height) if rows is None else rows
    i = np.arange(width, dtype=np.float64) + offset[0]
    j = np.arange(first, last, dtype=np.float64) + offset[1]
    u = (2.0 * i / width - 1.0) * half_width
    v = (1.0 - 2.0 * j / height) * half_height
    directions = forward + u[np.newaxis, :, np.newaxis] * right + v[:, np.newaxis, np.newaxis] * up
    return normalize(directions)
