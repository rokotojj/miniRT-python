"""Тесты дополнительного задания «Цветные источники»: несколько L, тени для каждого."""

from pathlib import Path

import numpy as np

from minirt.lighting import shade
from minirt.parser import parse_scene_file
from minirt.render import render
from minirt.scene import Ambient, Camera, Light, Scene, Shape, Sphere

SCENES = Path(__file__).resolve().parent.parent / "scenes"
UP = np.array([[0.0, 1.0, 0.0]])
ORIGIN = np.zeros((1, 3))
WHITE = np.ones((1, 3))

RED_ABOVE = Light((0.0, 10.0, 0.0), 1.0, (255, 0, 0))
BLUE_ASIDE = Light((10.0, 10.0, 0.0), 1.0, (0, 0, 255))


def scene_with(lights: tuple[Light, ...], *objects: Shape) -> Scene:
    return Scene(
        Ambient(0.0, (255, 255, 255)),
        Camera((0.0, 0.0, -10.0), (0.0, 0.0, 1.0), 60.0),
        lights,
        objects,
    )


def test_contributions_of_lights_add_up() -> None:
    color = shade(scene_with((RED_ABOVE, BLUE_ASIDE)), ORIGIN, UP, WHITE)
    np.testing.assert_allclose(color, [[1.0, 0.0, np.sqrt(0.5)]])


def test_light_color_multiplies_object_color() -> None:
    red_object = np.array([[1.0, 0.0, 0.0]])
    blue_only = shade(scene_with((BLUE_ASIDE,)), ORIGIN, UP, red_object)
    np.testing.assert_allclose(blue_only, [[0.0, 0.0, 0.0]])


def test_shadows_are_computed_per_light() -> None:
    blocker = Sphere((0.0, 5.0, 0.0), 1.0, (255, 255, 255))  # заслоняет только верхний
    color = shade(scene_with((RED_ABOVE, BLUE_ASIDE), blocker), ORIGIN, UP, WHITE)
    np.testing.assert_allclose(color, [[0.0, 0.0, np.sqrt(0.5)]])


def test_parsed_scene_has_two_colored_lights() -> None:
    scene = parse_scene_file(SCENES / "lights.rt")
    assert [light.color for light in scene.lights] == [(255, 80, 140), (80, 140, 255)]


def test_colored_shadows_in_render() -> None:
    image = render(parse_scene_file(SCENES / "lights.rt"), 800, 600).astype(int)
    pink_shadow = image[300, 650]  # тень от розового источника освещена голубым
    blue_shadow = image[320, 100]  # тень от голубого источника освещена розовым
    both = image[315, 380]  # тень от обоих — только фоновый свет
    assert pink_shadow[2] > pink_shadow[0]
    assert blue_shadow[0] > blue_shadow[2]
    assert both.tolist() == [23, 23, 23]
