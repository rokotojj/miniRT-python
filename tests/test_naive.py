"""Сверка векторного рендера с попиксельным эталоном tools/naive.py на чистом Python."""

from pathlib import Path

import numpy as np
import pytest

from minirt.parser import parse_scene_file
from minirt.render import render
from tools.naive import render_naive, to_image

SCENES = Path(__file__).resolve().parent.parent / "scenes"


@pytest.mark.parametrize("name", ["three.rt", "cylinders.rt", "sphere.rt"])
def test_numpy_render_matches_naive_loop(name: str) -> None:
    scene = parse_scene_file(SCENES / name)
    expected = to_image(render_naive(scene, 48, 36))
    actual = render(scene, 48, 36)
    difference = np.abs(actual.astype(int) - expected.astype(int))
    assert difference.max() <= 1
    assert (difference > 0).mean() < 0.01
