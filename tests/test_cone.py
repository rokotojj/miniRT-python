"""Тесты дополнительного задания «Конус» (элемент co)."""

import math
from pathlib import Path

import numpy as np
import pytest

from minirt.intersect import closest_hit, cone_normals, intersect_cone
from minirt.parser import parse_scene_file, parse_scene_text
from minirt.render import render
from minirt.scene import Cone, Vec3
from minirt.vec import FloatArray, normalize

SCENES = Path(__file__).resolve().parent.parent / "scenes"

# стоящий конус: вершина (0, 2, 5), ось вниз, радиус основания 1, высота 2 (основание на y = 0)
STANDING = Cone((0.0, 2.0, 5.0), (0.0, -1.0, 0.0), 1.0, 2.0, (230, 190, 50))


def rays(origin: Vec3, *directions: Vec3) -> tuple[FloatArray, FloatArray]:
    d = normalize(np.array(directions, dtype=np.float64))
    return np.broadcast_to(np.array(origin, dtype=np.float64), d.shape), d


def test_parse_cone() -> None:
    header = "A 0.2 255,255,255\nC 0,0,0 0,0,1 60\nL 0,5,0 0.5 255,255,255\n"
    scene = parse_scene_text(header + "co 0,2,5 0,-1,0 2 2 230,190,50 checker\n")
    assert scene.objects == (
        Cone((0.0, 2.0, 5.0), (0.0, -1.0, 0.0), 1.0, 2.0, (230, 190, 50), checker=True),
    )


def test_side_hit() -> None:
    # на высоте y = 1 радиус сечения 0,5: луч вдоль z входит в z = 4,5
    o, d = rays((0, 1, 0), (0, 0, 1))
    assert intersect_cone(o, d, STANDING)[0] == pytest.approx(4.5)


def test_base_hit_from_below() -> None:
    o, d = rays((0.3, -5, 5), (0, 1, 0))
    assert intersect_cone(o, d, STANDING)[0] == pytest.approx(5.0)


def test_mirror_half_is_ignored() -> None:
    # выше вершины лежит зеркальная половина двойного конуса — её быть не должно
    o, d = rays((0, 3, 0), (0, 0, 1))
    assert intersect_cone(o, d, STANDING)[0] == math.inf


def test_miss_beside() -> None:
    o, d = rays((2, 1, 0), (0, 0, 1))
    assert intersect_cone(o, d, STANDING)[0] == math.inf


def test_from_inside() -> None:
    # изнутри на высоте m = 1,5 радиус сечения 0,75
    o, d = rays((0, 0.5, 5), (1, 0, 0))
    assert intersect_cone(o, d, STANDING)[0] == pytest.approx(0.75)


def test_ray_parallel_to_generator() -> None:
    # квадратичный коэффициент равен нулю — работает линейная ветка
    cone = Cone((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 2.0, (255, 255, 255))
    o, d = rays((-1, 0, 0), (0.5, 0, 1))
    assert intersect_cone(o, d, cone)[0] == pytest.approx(math.hypot(0.5, 1.0))


def test_side_normal_is_perpendicular_to_generator() -> None:
    normals = cone_normals(STANDING, np.array([[0.0, 1.0, 4.5]]))
    np.testing.assert_allclose(normals, [[0.0, 1.0 / math.sqrt(5), -2.0 / math.sqrt(5)]])
    generator = np.array([0.0, -1.0, -0.5])
    assert float(normals[0] @ generator) == pytest.approx(0.0, abs=1e-12)


def test_base_normal_is_axis() -> None:
    normals = cone_normals(STANDING, np.array([[0.3, 0.0, 5.2], [0.0, 0.0, 5.0]]))
    np.testing.assert_allclose(normals, [[0, -1, 0], [0, -1, 0]])


def test_apex_normal_is_defined() -> None:
    normals = cone_normals(STANDING, np.array([[0.0, 2.0, 5.0]]))
    assert np.isfinite(normals).all()
    assert np.linalg.norm(normals[0]) == pytest.approx(1.0)


def test_closest_hit_with_cone() -> None:
    o, d = rays((0, 1, 0), (0, 0, 1))
    t, index = closest_hit([STANDING], o, d)
    assert (t[0], index[0]) == (pytest.approx(4.5), 0)


def test_cone_scene_renders() -> None:
    image = render(parse_scene_file(SCENES / "cone.rt"), 80, 60)
    assert image.shape == (60, 80, 3)
    assert not image[:3].any()
    yellow = image[22:30, 14:24].reshape(-1, 3).astype(int)
    assert (yellow[:, 0] > yellow[:, 2]).any()  # жёлтый конус слева
