"""Тесты дополнительного задания «Многопроцессность» (--workers N)."""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from minirt.cli import main
from minirt.parser import parse_scene_file
from minirt.render import RenderOptions, render, stripes

SCENES = Path(__file__).resolve().parent.parent / "scenes"


@pytest.mark.parametrize(("height", "count"), [(600, 8), (7, 16), (1, 4), (1000, 3)])
def test_stripes_cover_frame(height: int, count: int) -> None:
    parts = stripes(height, count)
    assert parts[0][0] == 0
    assert parts[-1][1] == height
    assert all(a[1] == b[0] for a, b in pairwise(parts))
    assert all(a < b for a, b in parts)
    sizes = [b - a for a, b in parts]
    assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize("name", ["three.rt", "checker.rt", "cone.rt"])
def test_parallel_render_equals_serial(name: str) -> None:
    scene = parse_scene_file(SCENES / name)
    serial = render(scene, 60, 45, RenderOptions(specular=True, samples=2))
    parallel = render(scene, 60, 45, RenderOptions(specular=True, samples=2, workers=2))
    np.testing.assert_array_equal(parallel, serial)


def test_cli_workers_zero_means_all_cores(tmp_path: Path) -> None:
    out = tmp_path / "p.png"
    argv = [str(SCENES / "three.rt"), "--width", "24", "--height", "18", "--workers", "0"]
    assert main([*argv, "--no-window", "--out", str(out)]) == 0
    assert out.exists()


@pytest.mark.parametrize("value", ["-1", "65", "many"])
def test_bad_workers_rejected(value: str) -> None:
    assert main([str(SCENES / "three.rt"), "--workers", value, "--no-window"]) == 2
