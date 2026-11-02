"""Модель сцены: неизменяемые описания элементов файла .rt.

Векторы хранятся кортежами из трёх float, цвета — кортежами из трёх int 0..255.
Направления (камера, нормаль плоскости, ось цилиндра) парсер уже нормировал,
размеры переведены из диаметров в радиусы. checker — флаг шахматной раскраски
(дополнительное задание).
"""

from dataclasses import dataclass
from typing import TypeAlias

Vec3: TypeAlias = tuple[float, float, float]
Color: TypeAlias = tuple[int, int, int]


@dataclass(frozen=True)
class Ambient:
    """A — фоновый свет."""

    ratio: float
    color: Color


@dataclass(frozen=True)
class Camera:
    """C — камера: положение, единичный вектор взгляда, горизонтальный угол обзора в градусах."""

    position: Vec3
    direction: Vec3
    fov: float


@dataclass(frozen=True)
class Light:
    """L — точечный источник света."""

    position: Vec3
    brightness: float
    color: Color


@dataclass(frozen=True)
class Sphere:
    """sp — сфера."""

    center: Vec3
    radius: float
    color: Color
    checker: bool = False


@dataclass(frozen=True)
class Plane:
    """pl — плоскость через точку point с единичной нормалью normal."""

    point: Vec3
    normal: Vec3
    color: Color
    checker: bool = False


@dataclass(frozen=True)
class Cylinder:
    """cy — конечный цилиндр с крышками; center — середина оси, крышки на ±height/2."""

    center: Vec3
    axis: Vec3
    radius: float
    height: float
    color: Color
    checker: bool = False


@dataclass(frozen=True)
class Cone:
    """co — конус с основанием (дополнительное задание).

    apex — вершина, axis — единичная ось от вершины к основанию, radius — радиус основания,
    height — расстояние от вершины до основания.
    """

    apex: Vec3
    axis: Vec3
    radius: float
    height: float
    color: Color
    checker: bool = False


Shape: TypeAlias = Sphere | Plane | Cylinder | Cone


@dataclass(frozen=True)
class Scene:
    """Сцена целиком: ровно один A, ровно одна C, хотя бы один L, любые объекты."""

    ambient: Ambient
    camera: Camera
    lights: tuple[Light, ...]
    objects: tuple[Shape, ...]
