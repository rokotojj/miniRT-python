"""Тесты построения кадра целиком."""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from minirt import render as render_module
from minirt.parser import parse_scene_file, parse_scene_text
from minirt.render import bands, render, to_rgb8

SCENES = Path(__file__).resolve().parent.parent / "scenes"


def test_image_size_and_type() -> None:
    image = render(parse_scene_file(SCENES / "sphere.rt"), 40, 30)
    assert image.shape == (30, 40, 3)
    assert image.dtype == np.uint8


def test_background_is_black_and_sphere_is_in_center() -> None:
    image = render(parse_scene_file(SCENES / "sphere.rt"), 41, 31)
    for corner in (image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]):
        assert corner.tolist() == [0, 0, 0]
    red, green, blue = image[15, 20].tolist()
    assert red > max(green, blue)


def test_flat_lighting_reproduces_object_color() -> None:
    # фоновый свет 1, источник с нулевой яркостью: цвет пикселя равен цвету объекта
    text = "A 1 255,255,255\nC 0,0,0 0,0,1 60\nL 0,0,0 0 255,255,255\nsp 0,0,10 4 10,128,250\n"
    image = render(parse_scene_text(text), 9, 9)
    assert image[4, 4].tolist() == [10, 128, 250]


def test_to_rgb8_clips_and_rounds() -> None:
    colors = np.array([[-0.5, 0.5, 2.0], [0.0, 1.0 / 255.0, 0.999]])
    assert to_rgb8(colors).tolist() == [[0, 128, 255], [0, 1, 255]]


def test_underside_of_plane_faces_viewer() -> None:
    # камера и источник под плоскостью с нормалью вверх: нижняя сторона должна быть освещена
    text = (
        "A 0 255,255,255\nC 0,-5,0 0,1,0 60\nL 0,-3,0 1 255,255,255\npl 0,0,0 0,1,0 255,255,255\n"
    )
    image = render(parse_scene_text(text), 11, 11)
    assert image[5, 5].tolist() == [255, 255, 255]


def test_inside_of_sphere_is_lit_from_inside() -> None:
    text = "A 0 255,255,255\nC 0,0,0 0,0,1 60\nL 0,0,0 1 255,255,255\nsp 0,0,0 10 200,100,50\n"
    image = render(parse_scene_text(text), 9, 9)
    assert image[4, 4].tolist() == [200, 100, 50]


def test_three_rt_black_sky_and_lit_floor() -> None:
    image = render(parse_scene_file(SCENES / "three.rt"), 80, 60)
    assert image.shape == (60, 80, 3)
    assert not image[:5].any()  # сверху — фон, чёрный
    floor = image[-5:].astype(int)
    assert (floor > 0).all()  # снизу — освещённый пол
    assert (floor[..., 0] == floor[..., 1]).all()  # серый пол: R = G = B
    assert floor.mean() > 100


def test_shadow_darker_than_lit_floor() -> None:
    image = render(parse_scene_file(SCENES / "three.rt"), 800, 600).astype(int)
    shadow = image[330, 360]  # тень красного шара справа от него (см. docs/screenshot.png)
    lit = image[330, 120]
    assert shadow.sum() < lit.sum() / 2


def test_bands_cover_frame_without_gaps() -> None:
    for width, height in [(800, 600), (1, 1), (100000 // 3, 2), (7, 1000)]:
        parts = bands(width, height)
        assert parts[0][0] == 0
        assert parts[-1][1] == height
        assert all(a[1] == b[0] for a, b in pairwise(parts))


def test_banded_render_equals_single_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    scene = parse_scene_file(SCENES / "three.rt")
    banded = render(scene, 64, 48)
    monkeypatch.setattr(render_module, "BAND_PIXELS", 10**9)
    np.testing.assert_array_equal(render(scene, 64, 48), banded)
