"""Тесты пересечений и нормалей на ответах, посчитанных на бумаге."""

import math

import numpy as np
import pytest

from minirt.intersect import (
    EPSILON,
    closest_hit,
    cylinder_normals,
    intersect_cylinder,
    intersect_plane,
    intersect_sphere,
    occluded,
    plane_normals,
    sphere_normals,
)
from minirt.scene import Cylinder, Plane, Sphere, Vec3
from minirt.vec import FloatArray, normalize


def rays(origin: Vec3, *directions: Vec3) -> tuple[FloatArray, FloatArray]:
    """Пучок лучей из одной точки: начала (N, 3) и единичные направления (N, 3)."""
    d = normalize(np.array(directions, dtype=np.float64))
    return np.broadcast_to(np.array(origin, dtype=np.float64), d.shape), d


UNIT_SPHERE = Sphere((0.0, 0.0, 5.0), 1.0, (255, 0, 0))


def test_sphere_hit_from_outside_takes_near_root() -> None:
    o, d = rays((0, 0, 0), (0, 0, 1))
    assert intersect_sphere(o, d, UNIT_SPHERE)[0] == pytest.approx(4.0)


def test_sphere_miss() -> None:
    o, d = rays((0, 2, 0), (0, 0, 1))
    assert intersect_sphere(o, d, UNIT_SPHERE)[0] == math.inf


def test_sphere_tangent_ray_touches_once() -> None:
    o, d = rays((0, 1, 0), (0, 0, 1))
    assert intersect_sphere(o, d, UNIT_SPHERE)[0] == pytest.approx(5.0)


def test_sphere_from_inside_takes_far_root() -> None:
    o, d = rays((0, 0, 5), (0, 0, 1), (1, 0, 0))
    np.testing.assert_allclose(intersect_sphere(o, d, UNIT_SPHERE), [1.0, 1.0])


def test_sphere_behind_camera_is_not_hit() -> None:
    o, d = rays((0, 0, 10), (0, 0, 1))
    assert intersect_sphere(o, d, UNIT_SPHERE)[0] == math.inf


def test_ray_starting_on_surface_ignores_start_point() -> None:
    o, d = rays((0, 0, 4), (0, 0, 1), (0, 0, -1))
    t = intersect_sphere(o, d, UNIT_SPHERE)
    assert t[0] == pytest.approx(2.0)  # насквозь до дальней стенки
    assert t[1] == math.inf  # уходит от сферы
    assert EPSILON < 1e-3


def test_sphere_batch_returns_one_distance_per_ray() -> None:
    o, d = rays((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 0.1, 1))
    t = intersect_sphere(o, d, UNIT_SPHERE)
    assert t.shape == (3,)
    assert t[0] == pytest.approx(4.0)
    assert t[1] == math.inf
    assert 4.0 < t[2] < 5.0


def test_sphere_normal_points_outwards() -> None:
    points = np.array([[0.0, 0.0, 4.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]])
    np.testing.assert_allclose(
        sphere_normals(UNIT_SPHERE, points), [[0, 0, -1], [1, 0, 0], [0, 1, 0]]
    )


def test_closest_hit_picks_nearest_object() -> None:
    far = Sphere((0.0, 0.0, 10.0), 1.0, (0, 0, 255))
    o, d = rays((0, 0, 0), (0, 0, 1), (0, 1, 0))
    t, index = closest_hit([far, UNIT_SPHERE], o, d)
    assert t[0] == pytest.approx(4.0)
    assert index.tolist() == [1, -1]
    assert t[1] == math.inf


def test_closest_hit_nested_spheres() -> None:
    outer = Sphere((0.0, 0.0, 5.0), 3.0, (0, 255, 0))
    o, d = rays((0, 0, 0), (0, 0, 1))
    t, index = closest_hit([UNIT_SPHERE, outer], o, d)
    assert (t[0], index[0]) == (pytest.approx(2.0), 1)
    o, d = rays((0, 0, 5), (0, 0, 1))  # изнутри обеих: ближе стенка малой сферы
    t, index = closest_hit([outer, UNIT_SPHERE], o, d)
    assert (t[0], index[0]) == (pytest.approx(1.0), 1)


def test_closest_hit_without_objects() -> None:
    o, d = rays((0, 0, 0), (0, 0, 1))
    t, index = closest_hit([], o, d)
    assert t[0] == math.inf
    assert index[0] == -1


FLOOR = Plane((0.0, -1.0, 0.0), (0.0, 1.0, 0.0), (200, 200, 200))


def test_plane_hit() -> None:
    o, d = rays((0, 3, 0), (0, -1, 0), (0, -1, 1))
    np.testing.assert_allclose(intersect_plane(o, d, FLOOR), [4.0, 4.0 * math.sqrt(2.0)])


def test_plane_behind_camera() -> None:
    o, d = rays((0, 3, 0), (0, 1, 0))
    assert intersect_plane(o, d, FLOOR)[0] == math.inf


def test_ray_parallel_to_plane() -> None:
    o, d = rays((0, 3, 0), (1, 0, 0), (0, 0, 1))
    assert intersect_plane(o, d, FLOOR).tolist() == [math.inf, math.inf]


def test_plane_hit_from_below() -> None:
    o, d = rays((0, -3, 0), (0, 1, 0))
    assert intersect_plane(o, d, FLOOR)[0] == pytest.approx(2.0)


def test_plane_normal_is_constant() -> None:
    normals = plane_normals(FLOOR, np.zeros((4, 3)))
    np.testing.assert_allclose(normals, [[0, 1, 0]] * 4)


# вертикальный цилиндр: ось y, радиус 1, высота 2 (крышки на y = ±1), центр (0, 0, 5)
PILLAR = Cylinder((0.0, 0.0, 5.0), (0.0, 1.0, 0.0), 1.0, 2.0, (0, 0, 255))


def test_cylinder_side_hit() -> None:
    o, d = rays((0, 0, 0), (0, 0, 1), (0, 0.1, 1))
    t = intersect_cylinder(o, d, PILLAR)
    assert t[0] == pytest.approx(4.0)
    assert t[1] == pytest.approx(4.0 * math.hypot(1.0, 0.1))


def test_cylinder_cap_hit() -> None:
    o, d = rays((0, 5, 5), (0, -1, 0))
    assert intersect_cylinder(o, d, PILLAR)[0] == pytest.approx(4.0)


def test_cylinder_miss_above_cap() -> None:
    o, d = rays((0, 1.5, 0), (0, 0, 1))
    assert intersect_cylinder(o, d, PILLAR)[0] == math.inf


def test_cylinder_miss_beside() -> None:
    o, d = rays((2, 0, 0), (0, 0, 1))
    assert intersect_cylinder(o, d, PILLAR)[0] == math.inf


def test_ray_parallel_to_axis() -> None:
    o, d = rays((0.5, -5, 5), (0, 1, 0))
    assert intersect_cylinder(o, d, PILLAR)[0] == pytest.approx(4.0)  # нижняя крышка
    o, d = rays((1.5, -5, 5), (0, 1, 0))
    assert intersect_cylinder(o, d, PILLAR)[0] == math.inf  # снаружи радиуса


def test_cylinder_from_inside() -> None:
    o, d = rays((0, 0, 5), (0, 0, 1), (0, 1, 0))
    np.testing.assert_allclose(intersect_cylinder(o, d, PILLAR), [1.0, 1.0])


def test_cylinder_side_hit_through_cap_edge() -> None:
    # луч сверху наискосок: входит через крышку, а не через бок
    o, d = rays((0, 3, 3), (0, -1, 1))
    assert intersect_cylinder(o, d, PILLAR)[0] == pytest.approx(2.0 * math.sqrt(2.0))


def test_tilted_cylinder() -> None:
    half = math.sqrt(0.5)
    tilted = Cylinder((0.0, 0.0, 5.0), (half, half, 0.0), 1.0, 10.0, (0, 0, 255))
    o, d = rays((0, 0, 0), (0, 0, 1))
    assert intersect_cylinder(o, d, tilted)[0] == pytest.approx(4.0)


def test_cylinder_normals() -> None:
    points = np.array([[0.0, 0.0, 4.0], [1.0, 0.5, 5.0], [0.2, 1.0, 5.0], [0.0, -1.0, 5.3]])
    np.testing.assert_allclose(
        cylinder_normals(PILLAR, points),
        [[0, 0, -1], [1, 0, 0], [0, 1, 0], [0, -1, 0]],
        atol=1e-12,
    )


def test_cylinder_rim_normal_is_unit() -> None:
    normals = cylinder_normals(PILLAR, np.array([[1.0, 1.0, 5.0], [0.0, 0.0, 5.0 - 1.0]]))
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), [1.0, 1.0])


def test_occluded_respects_distance() -> None:
    o, d = rays((0, 0, 0), (0, 0, 1), (0, 0, 1), (0, 1, 0))
    blocked = occluded([UNIT_SPHERE], o, d, np.array([10.0, 3.0, 10.0]))
    assert blocked.tolist() == [True, False, False]
