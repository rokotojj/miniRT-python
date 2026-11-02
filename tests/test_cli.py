"""Тесты точки входа: коды возврата, сообщения Error, сохранение PNG."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from minirt import cli
from minirt.cli import main
from minirt.errors import WindowError
from minirt.image_io import decode_png

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "scenes"


def test_no_arguments_prints_help_and_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    err = capsys.readouterr().err
    assert "usage: python main.py" in err
    assert "--no-window" in err


@pytest.mark.parametrize(
    "argv",
    [
        ["scenes/three.rt", "--width", "0"],
        ["scenes/three.rt", "--height", "abc"],
        ["scenes/three.rt", "--width", "100000"],
        ["scenes/three.rt", "--unknown"],
        ["a.rt", "b.rt"],
    ],
)
def test_bad_arguments_return_2(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(argv) == 2
    assert "usage:" in capsys.readouterr().err


def test_help_returns_0(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    assert "--out" in capsys.readouterr().out


def test_missing_file_reports_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["no/such/scene.rt", "--no-window"]) == 1
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == "Error"
    assert "no/such/scene.rt" in lines[1]


@pytest.mark.parametrize("path", sorted((SCENES / "bad").iterdir()), ids=lambda p: p.name)
def test_every_bad_scene_gives_error_and_code_1(
    path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(path), "--no-window"]) == 1
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == "Error"
    assert len(lines) == 2
    assert lines[1]


def test_render_to_png(tmp_path: Path) -> None:
    out = tmp_path / "three.png"
    argv = [str(SCENES / "three.rt"), "--width", "64", "--height", "48", "--no-window"]
    assert main([*argv, "--out", str(out)]) == 0
    image = decode_png(out.read_bytes())
    assert image.shape == (48, 64, 3)
    assert image.any()


def test_unwritable_output_reports_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "missing_dir" / "x.png"
    argv = [str(SCENES / "sphere.rt"), "--width", "8", "--height", "8", "--no-window"]
    assert main([*argv, "--out", str(out)]) == 1
    lines = capsys.readouterr().err.splitlines()
    assert lines[-2] == "Error"
    assert lines[-1].startswith("не удалось записать")


def test_main_py_exit_codes(tmp_path: Path) -> None:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / "main.py"), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            check=False,
        )

    assert run().returncode == 2
    bad = run(str(SCENES / "bad" / "ambient_ratio.rt"), "--no-window")
    assert bad.returncode == 1
    assert bad.stderr.splitlines() == [
        "Error",
        "строка 2: яркость фонового света должна быть в диапазоне 0..1, получено 1.5",
    ]
    out = tmp_path / "s.png"
    ok = run(
        str(SCENES / "sphere.rt"),
        "--width",
        "16",
        "--height",
        "12",
        "--no-window",
        "--out",
        str(out),
    )
    assert ok.returncode == 0
    assert out.exists()


def test_render_time_is_printed(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(SCENES / "sphere.rt"), "--width", "8", "--height", "6", "--no-window"]) == 0
    err = capsys.readouterr().err
    assert "8x6, построено за" in err
    assert err.rstrip().endswith("с")


def test_window_is_shown_without_no_window(monkeypatch: pytest.MonkeyPatch) -> None:
    shown: list[tuple[tuple[int, ...], str]] = []

    def fake_show(image: npt.NDArray[np.uint8], title: str) -> None:
        shown.append((image.shape, title))

    monkeypatch.setattr(cli, "show_image", fake_show)
    assert main([str(SCENES / "sphere.rt"), "--width", "10", "--height", "7"]) == 0
    assert len(shown) == 1
    assert shown[0][0] == (7, 10, 3)
    assert "sphere.rt" in shown[0][1]


def test_window_failure_reports_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken_show(image: npt.NDArray[np.uint8], title: str) -> None:
        raise WindowError("нет дисплея")

    monkeypatch.setattr(cli, "show_image", broken_show)
    assert main([str(SCENES / "sphere.rt"), "--width", "4", "--height", "4"]) == 1
    assert capsys.readouterr().err.splitlines()[-2:] == ["Error", "нет дисплея"]
