"""Тесты освещения: формулы фонового и диффузного света."""

import numpy as np
import pytest

from minirt.lighting import in_shadow, lambert, shade
from minirt.scene import Ambient, Camera, Cylinder, Light, Plane, Scene, Shape, Sphere
from minirt.vec import FloatArray

NO_AMBIENT = Ambient(0.0, (255, 255, 255))
TOP_LIGHT = Light((0.0, 10.0, 0.0), 1.0, (255, 255, 255))


def make_scene(
    *objects: Shape, ambient: Ambient = NO_AMBIENT, lights: tuple[Light, ...] = (TOP_LIGHT,)
) -> Scene:
    return Scene(ambient, Camera((0.0, 0.0, -10.0), (0.0, 0.0, 1.0), 60.0), lights, objects)


def point(x: float, y: float, z: float) -> FloatArray:
    return np.array([[x, y, z]], dtype=np.float64)


WHITE = np.array([[1.0, 1.0, 1.0]])
UP = point(0.0, 1.0, 0.0)


def test_lambert_cosine() -> None:
    normals = np.array([[0.0, 1.0, 0.0]] * 3)
    to_light = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    np.testing.assert_allclose(lambert(normals, to_light), [1.0, 0.0, 0.0])


def test_light_along_normal_gives_full_brightness() -> None:
    color = shade(make_scene(), point(0, 0, 0), UP, np.array([[0.8, 0.4, 0.2]]))
    np.testing.assert_allclose(color, [[0.8, 0.4, 0.2]])


def test_brightness_scales_diffuse_term() -> None:
    scene = make_scene(lights=(Light((0.0, 10.0, 0.0), 0.5, (255, 255, 255)),))
    np.testing.assert_allclose(shade(scene, point(0, 0, 0), UP, WHITE), [[0.5, 0.5, 0.5]])


def test_grazing_light_gives_zero_diffuse() -> None:
    scene = make_scene(
        ambient=Ambient(0.25, (255, 255, 255)),
        lights=(Light((10.0, 0.0, 0.0), 1.0, (255, 255, 255)),),
    )
    np.testing.assert_allclose(shade(scene, point(0, 0, 0), UP, WHITE), [[0.25, 0.25, 0.25]])


def test_light_from_behind_gives_only_ambient() -> None:
    scene = make_scene(
        ambient=Ambient(0.1, (255, 255, 255)),
        lights=(Light((0.0, -5.0, 0.0), 1.0, (255, 255, 255)),),
    )
    np.testing.assert_allclose(shade(scene, point(0, 0, 0), UP, WHITE), [[0.1, 0.1, 0.1]])


def test_colors_multiply_componentwise() -> None:
    scene = make_scene(
        ambient=Ambient(0.5, (255, 0, 0)),
        lights=(Light((0.0, 10.0, 0.0), 1.0, (0, 255, 0)),),
    )
    color = shade(scene, point(0, 0, 0), UP, np.array([[1.0, 0.5, 1.0]]))
    np.testing.assert_allclose(color, [[0.5, 0.5, 0.0]])


def test_cosine_of_sixty_degrees_halves_diffuse() -> None:
    scene = make_scene(lights=(Light((np.sqrt(3.0), 1.0, 0.0), 1.0, (255, 255, 255)),))
    color = shade(scene, point(0, 0, 0), UP, WHITE)
    assert color[0, 0] == pytest.approx(0.5)


def test_on_sphere_lit_side_brighter_than_dark_side() -> None:
    sphere = Sphere((0.0, 0.0, 0.0), 1.0, (255, 255, 255))
    points = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    scene = make_scene(sphere, ambient=Ambient(0.2, (255, 255, 255)))
    color = shade(scene, points, points.copy(), np.ones((2, 3)))
    np.testing.assert_allclose(color[:, 0], [1.2, 0.2])


def test_light_behind_object_gives_only_ambient() -> None:
    blocker = Sphere((0.0, 5.0, 0.0), 1.0, (255, 255, 255))
    scene = make_scene(blocker, ambient=Ambient(0.2, (255, 255, 255)))
    np.testing.assert_allclose(shade(scene, point(0, 0, 0), UP, WHITE), [[0.2, 0.2, 0.2]])


def test_object_beyond_light_casts_no_shadow() -> None:
    beyond = Sphere((0.0, 20.0, 0.0), 1.0, (255, 255, 255))
    np.testing.assert_allclose(shade(make_scene(beyond), point(0, 0, 0), UP, WHITE), WHITE)


def test_in_shadow_mask() -> None:
    blocker = Sphere((0.0, 5.0, 0.0), 1.0, (255, 255, 255))
    points = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    normals = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    assert in_shadow([blocker], TOP_LIGHT, points, normals).tolist() == [True, False]


def test_lit_hemisphere_has_no_shadow_acne() -> None:
    sphere = Sphere((0.0, 0.0, 0.0), 1.0, (255, 255, 255))
    rng = np.random.default_rng(0)
    normals = rng.normal(size=(5000, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    points = normals * sphere.radius
    to_light = np.array(TOP_LIGHT.position) - points
    lit = np.einsum("ij,ij->i", normals, to_light) > 1e-6  # освещённая часть сферы
    assert lit.sum() > 1000
    assert not in_shadow([sphere], TOP_LIGHT, points[lit], normals[lit]).any()


def test_sphere_casts_shadow_on_plane() -> None:
    floor = Plane((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (255, 255, 255))
    ball = Sphere((0.0, 2.0, 0.0), 2.0, (255, 0, 0))
    scene = make_scene(floor, ball, ambient=Ambient(0.1, (255, 255, 255)))
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5], [6.0, 0.0, 0.0]])
    normals = np.array([[0.0, 1.0, 0.0]] * 3)
    color = shade(scene, points, normals, np.ones((3, 3)))
    np.testing.assert_allclose(color[:2], [[0.1] * 3] * 2)  # под шаром — только фон
    assert color[2, 0] > 0.5  # в стороне — освещено


def test_cylinder_casts_shadow() -> None:
    pillar = Cylinder((0.0, 5.0, 0.0), (0.0, 1.0, 0.0), 1.0, 2.0, (255, 255, 255))
    color = shade(make_scene(pillar), point(0, 0, 0), UP, WHITE)
    np.testing.assert_allclose(color, [[0.0, 0.0, 0.0]])
