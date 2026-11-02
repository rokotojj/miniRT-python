"""Тесты дополнительного задания «Блики Фонга» (--specular)."""

from pathlib import Path

import numpy as np
import pytest

from minirt.cli import main
from minirt.lighting import SPECULAR_K, SPECULAR_POWER, phong, shade
from minirt.parser import parse_scene_file
from minirt.render import RenderOptions, render
from minirt.scene import Ambient, Camera, Light, Scene, Sphere

SCENES = Path(__file__).resolve().parent.parent / "scenes"
UP = np.array([[0.0, 1.0, 0.0]])
ORIGIN = np.zeros((1, 3))
DARK = np.array([[0.2, 0.2, 0.2]])


def scene_with(*objects: Sphere, light: tuple[float, float, float] = (0.0, 10.0, 0.0)) -> Scene:
    return Scene(
        Ambient(0.0, (255, 255, 255)),
        Camera((0.0, 0.0, -10.0), (0.0, 0.0, 1.0), 60.0),
        (Light(light, 1.0, (255, 255, 255)),),
        objects,
    )


def test_mirror_direction_gives_full_highlight() -> None:
    # свет и наблюдатель вдоль нормали: h = n, n·h = 1, блик = k
    color = shade(scene_with(), ORIGIN, UP, DARK, to_viewer=UP)
    np.testing.assert_allclose(color, DARK + SPECULAR_K)


def test_highlight_falls_off_quickly() -> None:
    # наблюдатель в 0,6 рад от нормали, свет по нормали: h — посередине, в 0,3 рад
    viewer = np.array([[np.sin(0.6), np.cos(0.6), 0.0]])
    highlight = phong(UP, UP, viewer, np.ones(1))
    assert highlight[0] == pytest.approx(SPECULAR_K * np.cos(0.3) ** SPECULAR_POWER)
    assert highlight[0] < 0.25 * SPECULAR_K


def test_no_highlight_without_specular() -> None:
    np.testing.assert_allclose(shade(scene_with(), ORIGIN, UP, DARK), DARK)


def test_no_highlight_in_shadow() -> None:
    blocker = Sphere((0.0, 5.0, 0.0), 1.0, (255, 255, 255))
    color = shade(scene_with(blocker), ORIGIN, UP, DARK, to_viewer=UP)
    np.testing.assert_allclose(color, [[0.0, 0.0, 0.0]])


def test_no_highlight_when_light_is_behind_surface() -> None:
    color = shade(scene_with(light=(0.0, -10.0, 0.0)), ORIGIN, UP, DARK, to_viewer=UP)
    np.testing.assert_allclose(color, [[0.0, 0.0, 0.0]])


def test_highlight_has_light_color_not_object_color() -> None:
    scene = scene_with()
    red_object = np.array([[1.0, 0.0, 0.0]])
    color = shade(scene, ORIGIN, UP, red_object, to_viewer=UP)
    np.testing.assert_allclose(color, [[1.0 + SPECULAR_K, SPECULAR_K, SPECULAR_K]])


def test_specular_render_is_brighter_only_on_highlights() -> None:
    scene = parse_scene_file(SCENES / "phong.rt")
    plain = render(scene, 80, 60).astype(int)
    shiny = render(scene, 80, 60, RenderOptions(specular=True)).astype(int)
    assert (shiny >= plain).all()
    assert (shiny > plain).any()
    assert not shiny[:5].any()  # фон остаётся чёрным


@pytest.mark.parametrize("flag", [[], ["--specular"]])
def test_cli_accepts_specular(flag: list[str], tmp_path: Path) -> None:
    out = tmp_path / "p.png"
    argv = [str(SCENES / "phong.rt"), "--width", "16", "--height", "12", "--no-window"]
    assert main([*argv, *flag, "--out", str(out)]) == 0
    assert out.exists()
