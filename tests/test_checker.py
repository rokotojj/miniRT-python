"""Тесты дополнительного задания «Шахматная раскраска» (флаг checker)."""

from pathlib import Path

import numpy as np
import pytest

from minirt.errors import SceneFormatError
from minirt.parser import parse_scene_file, parse_scene_text
from minirt.pattern import CHECKER_DARK, checker_odd, surface_colors
from minirt.render import render
from minirt.scene import Cylinder, Plane, Sphere

SCENES = Path(__file__).resolve().parent.parent / "scenes"
HEADER = "A 0.2 255,255,255\nC 0,0,0 0,0,1 60\nL 0,5,0 0.5 255,255,255\n"
UP = np.array([0.0, 1.0, 0.0])


def test_checker_flag_is_parsed_for_all_objects() -> None:
    scene = parse_scene_text(
        HEADER
        + "sp 0,0,5 2 1,2,3 checker\npl 0,-1,0 0,1,0 4,5,6 checker\n"
        + "cy 0,0,9 0,1,0 2 2 7,8,9 checker\nsp 0,0,7 2 1,1,1\n"
    )
    sphere, plane, cylinder, plain = scene.objects
    assert isinstance(sphere, Sphere)
    assert isinstance(plane, Plane)
    assert isinstance(cylinder, Cylinder)
    assert sphere.checker
    assert plane.checker
    assert cylinder.checker
    assert not plain.checker
    assert sphere.radius == 1.0


def test_unknown_flag_is_rejected() -> None:
    with pytest.raises(SceneFormatError) as info:
        parse_scene_text(HEADER + "sp 0,0,5 2 1,2,3 striped\n")
    assert info.value.line == 4
    assert "неизвестный флаг 'striped'" in info.value.reason


def test_checker_without_color_is_rejected() -> None:
    with pytest.raises(SceneFormatError, match="цвет сферы"):
        parse_scene_text(HEADER + "sp 0,0,5 2 checker\n")


def test_cell_parity_follows_floor_sum() -> None:
    points = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [1.5, 1.5, 0.5], [-0.5, 0.5, 0.5]])
    normals = np.broadcast_to(UP, points.shape)
    # суммы номеров клеток: 0, 1, 2, −1
    assert checker_odd(points, normals).tolist() == [False, True, False, True]


def test_surface_on_cell_boundary_does_not_flicker() -> None:
    # плоскость y = −1 лежит ровно на границе клеток; погрешность ±1e-12 не меняет клетку
    xs = np.linspace(0.05, 0.95, 50)
    points = np.stack([xs, -1.0 + np.resize([1e-12, -1e-12], 50), xs * 0], axis=1)
    odd = checker_odd(points, np.broadcast_to(UP, points.shape))
    assert odd.all() or not odd.any()


def test_surface_colors_alternate() -> None:
    plane = Plane((0.0, -1.0, 0.0), (0.0, 1.0, 0.0), (200, 100, 50), checker=True)
    # клетки (0, −2, 0) и (1, −2, 0): суммы −2 (чётная) и −1 (нечётная)
    points = np.array([[0.5, -1.0, 0.5], [1.5, -1.0, 0.5]])
    colors = surface_colors(plane, points, np.broadcast_to(UP, points.shape))
    base = np.array([200, 100, 50]) / 255.0
    np.testing.assert_allclose(colors, [base, base * CHECKER_DARK])


def test_plain_object_has_uniform_color() -> None:
    sphere = Sphere((0.0, 0.0, 0.0), 5.0, (10, 20, 30))
    points = np.array([[0.1, 0.2, 0.3], [3.7, 1.1, -2.2]])
    colors = surface_colors(sphere, points, points / np.linalg.norm(points, axis=1)[:, None])
    np.testing.assert_allclose(colors, [[10 / 255, 20 / 255, 30 / 255]] * 2)


def test_checker_floor_in_render_alternates() -> None:
    image = render(parse_scene_file(SCENES / "checker.rt"), 200, 150).astype(int)
    row = image[-3]  # ближний к камере ряд клеток пола
    brightness = row.sum(axis=1)
    assert brightness.max() > 3 * brightness.min()
    changes = np.count_nonzero(np.diff(brightness > brightness.mean()))
    assert changes >= 6
