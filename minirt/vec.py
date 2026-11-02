"""Векторная математика над массивами NumPy.

Все функции работают с массивами формы (..., 3): одним вектором (3,),
набором лучей (N, 3) или кадром (H, W, 3). Скалярное произведение берётся
по последней оси.
"""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from minirt.scene import Color, Vec3

FloatArray: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
IntArray: TypeAlias = npt.NDArray[np.intp]


def vec3(v: Vec3) -> FloatArray:
    """Кортеж (x, y, z) → массив формы (3,)."""
    return np.array(v, dtype=np.float64)


def color01(color: Color) -> FloatArray:
    """Цвет 0..255 → массив (3,) с компонентами 0..1."""
    return np.array(color, dtype=np.float64) / 255.0


def dot(a: FloatArray, b: FloatArray) -> FloatArray:
    """Скалярное произведение по последней оси: (..., 3) · (..., 3) → (...)."""
    result: FloatArray = np.einsum("...i,...i->...", a, b)
    return result


def norm(a: FloatArray) -> FloatArray:
    """Длина векторов по последней оси."""
    return np.sqrt(dot(a, a))


def normalize(a: FloatArray) -> FloatArray:
    """Единичные векторы того же направления. Нулевые векторы не допускаются."""
    return a / norm(a)[..., np.newaxis]


def cross(a: FloatArray, b: FloatArray) -> FloatArray:
    """Векторное произведение по последней оси."""
    result: FloatArray = np.cross(a, b)
    return result
