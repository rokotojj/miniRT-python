"""Тесты камеры: базис, центральный и крайние лучи, вертикальный взгляд."""

import math

import numpy as np
import pytest

from minirt.camera import camera_basis, ray_directions
from minirt.scene import Camera, Vec3
from minirt.vec import dot, norm


@pytest.mark.parametrize(
    "direction",
    [(0.0, 0.0, 1.0), (0.0, -0.2, 1.0), (1.0, 1.0, 1.0), (-0.3, 0.9, -0.1), (0.0, -1.0, 0.0)],
)
def test_basis_is_orthonormal(direction: Vec3) -> None:
    forward, right, up = camera_basis(direction)
    for axis in (forward, right, up):
        assert norm(axis) == pytest.approx(1.0)
    assert dot(forward, right) == pytest.approx(0.0, abs=1e-12)
    assert dot(forward, up) == pytest.approx(0.0, abs=1e-12)
    assert dot(right, up) == pytest.approx(0.0, abs=1e-12)


def test_basis_for_straight_view() -> None:
    forward, right, up = camera_basis((0.0, 0.0, 1.0))
    np.testing.assert_allclose(forward, [0, 0, 1])
    np.testing.assert_allclose(right, [1, 0, 0])
    np.testing.assert_allclose(up, [0, 1, 0])


def test_right_is_horizontal_and_up_points_up() -> None:
    _, right, up = camera_basis((0.3, -0.4, 0.8))
    assert right[1] == pytest.approx(0.0)
    assert up[1] > 0.0


def test_looking_straight_down_uses_z_as_world_up() -> None:
    forward, right, up = camera_basis((0.0, -1.0, 0.0))
    np.testing.assert_allclose(forward, [0, -1, 0])
    np.testing.assert_allclose(right, [1, 0, 0])
    np.testing.assert_allclose(up, [0, 0, 1])
    assert not np.isnan(ray_directions(Camera((0, 5, 0), (0, -1, 0), 60), 4, 3)).any()


def test_rays_shape_and_unit_length() -> None:
    rays = ray_directions(Camera((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 70.0), 8, 6)
    assert rays.shape == (6, 8, 3)
    np.testing.assert_allclose(norm(rays), np.ones((6, 8)))


def test_central_ray_matches_view_direction() -> None:
    direction = (0.0, -0.2 / math.hypot(1.0, 0.2), 1.0 / math.hypot(1.0, 0.2))
    rays = ray_directions(Camera((0.0, 3.0, -10.0), direction, 70.0), 5, 3)
    np.testing.assert_allclose(rays[1, 2], direction, atol=1e-12)


@pytest.mark.parametrize("fov", [30.0, 70.0, 120.0])
def test_edge_rays_deviate_by_half_fov(fov: float) -> None:
    camera = Camera((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), fov)
    # смещение 0 по x — левая граница кадра, нечётная высота — центральная строка
    left = ray_directions(camera, 4, 3, offset=(0.0, 0.5))[1, 0]
    angle = math.degrees(math.acos(left[2]))
    assert angle == pytest.approx(fov / 2.0)
    assert left[0] < 0.0


def test_corner_pixels_are_symmetric() -> None:
    rays = ray_directions(Camera((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0), 10, 6)
    top_left, bottom_right = rays[0, 0], rays[-1, -1]
    np.testing.assert_allclose(top_left * [-1, -1, 1], bottom_right)
    assert top_left[0] < 0.0 < top_left[1]


def test_vertical_extent_follows_aspect_ratio() -> None:
    camera = Camera((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)
    top = ray_directions(camera, 3, 2, offset=(0.5, 0.0))[0, 1]
    # половина высоты = tan(45°) · 2 / 3
    assert top[1] / top[2] == pytest.approx(2.0 / 3.0)


def test_rows_select_part_of_frame() -> None:
    camera = Camera((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 60.0)
    full = ray_directions(camera, 7, 9)
    np.testing.assert_allclose(ray_directions(camera, 7, 9, rows=(3, 6)), full[3:6])
