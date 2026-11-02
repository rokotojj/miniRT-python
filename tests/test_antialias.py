"""Тесты дополнительного задания «Сглаживание» (--samples N)."""

from pathlib import Path

import numpy as np
import pytest

from minirt.cli import main
from minirt.parser import parse_scene_file, parse_scene_text
from minirt.render import RenderOptions, render

SCENES = Path(__file__).resolve().parent.parent / "scenes"
# сфера ровного цвета: фоновый свет 1, источник выключен
FLAT = "A 1 255,255,255\nC 0,0,0 0,0,1 60\nL 0,0,0 0 255,255,255\nsp 0,0,10 6 200,100,40\n"


def test_one_sample_equals_plain_render() -> None:
    scene = parse_scene_file(SCENES / "three.rt")
    np.testing.assert_array_equal(
        render(scene, 40, 30, RenderOptions(samples=1)), render(scene, 40, 30)
    )


def test_samples_equal_supersampling_and_averaging() -> None:
    # N² лучей по сетке (k + 0,5) / N — то же, что кадр в N раз больше, усреднённый блоками N×N
    scene = parse_scene_file(SCENES / "three.rt")
    smooth = render(scene, 40, 30, RenderOptions(samples=2)).astype(float)
    big = render(scene, 80, 60).astype(float)
    averaged = big.reshape(30, 2, 40, 2, 3).mean(axis=(1, 3))
    assert np.abs(smooth - averaged).max() <= 1.0


def test_edges_get_intermediate_colors() -> None:
    scene = parse_scene_text(FLAT)
    sharp = render(scene, 41, 41)
    smooth = render(scene, 41, 41, RenderOptions(samples=4))
    colors_sharp = {tuple(c) for c in sharp.reshape(-1, 3).tolist()}
    colors_smooth = {tuple(c) for c in smooth.reshape(-1, 3).tolist()}
    assert colors_sharp == {(0, 0, 0), (200, 100, 40)}  # без сглаживания — «лесенка»
    assert len(colors_smooth) > 10
    assert smooth[20, 20].tolist() == [200, 100, 40]  # внутри — цвет без изменений
    assert smooth[0, 0].tolist() == [0, 0, 0]  # фон — чёрный


@pytest.mark.parametrize("value", ["0", "17", "two"])
def test_bad_samples_rejected(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(SCENES / "three.rt"), "--samples", value, "--no-window"]) == 2
    assert "--samples" in capsys.readouterr().err


def test_cli_samples(tmp_path: Path) -> None:
    out = tmp_path / "aa.png"
    argv = [str(SCENES / "three.rt"), "--width", "20", "--height", "15", "--samples", "3"]
    assert main([*argv, "--no-window", "--out", str(out)]) == 0
    assert out.exists()
