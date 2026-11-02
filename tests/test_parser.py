"""Тесты парсера сцены: разбор корректных сцен и все ошибки формата из scenes/bad/."""

import math
from pathlib import Path

import pytest

from minirt.errors import RtError, SceneFormatError
from minirt.parser import parse_scene_file, parse_scene_text
from minirt.scene import Ambient, Camera, Cylinder, Light, Plane, Sphere

SCENES = Path(__file__).resolve().parent.parent / "scenes"

MINIMAL = "A 0.2 255,255,255\nC 0,0,0 0,0,1 60\nL 0,5,0 0.5 255,255,255\n"


def test_three_rt_parsed_completely() -> None:
    scene = parse_scene_file(SCENES / "three.rt")

    assert scene.ambient == Ambient(0.2, (255, 255, 255))
    assert scene.camera.position == (0.0, 3.0, -10.0)
    assert scene.camera.fov == 70.0
    assert scene.lights == (Light((-6.0, 8.0, -4.0), 0.8, (255, 255, 255)),)
    assert len(scene.objects) == 4
    plane, big, small, cylinder = scene.objects
    assert plane == Plane((0.0, -1.0, 0.0), (0.0, 1.0, 0.0), (200, 200, 200))
    assert big == Sphere((-2.5, 0.2, 4.0), 1.5, (220, 50, 50))
    assert small == Sphere((2.5, -0.2, 2.5), 0.8, (60, 200, 80))
    assert cylinder == Cylinder((5.0, 0.0, 7.0), (0.0, 1.0, 0.0), 1.0, 4.0, (70, 110, 230))


def test_camera_direction_is_normalized() -> None:
    scene = parse_scene_file(SCENES / "three.rt")
    x, y, z = scene.camera.direction
    length = math.sqrt(1.0 + 0.2**2)
    assert (x, y, z) == pytest.approx((0.0, -0.2 / length, 1.0 / length))
    assert math.hypot(x, y, z) == pytest.approx(1.0)


def test_plane_normal_and_cylinder_axis_are_normalized() -> None:
    scene = parse_scene_text(MINIMAL + "pl 0,0,0 0,0.5,0 1,2,3\ncy 0,0,0 1,1,0 2 2 1,2,3\n")
    plane, cylinder = scene.objects
    assert isinstance(plane, Plane)
    assert isinstance(cylinder, Cylinder)
    assert plane.normal == pytest.approx((0.0, 1.0, 0.0))
    half = math.sqrt(0.5)
    assert cylinder.axis == pytest.approx((half, half, 0.0))


def test_comments_blank_lines_and_tabs_are_ignored() -> None:
    text = (
        "# заголовок\n"
        "\n"
        "   \n"
        "A\t0.1   10,20,30   # хвостовой комментарий\n"
        "C 0,0,0 0,0,1 90#без пробела\n"
        "\t\n"
        "L 1,2,3 1 255,255,255\n"
        "sp 0,0,5 2 1,2,3\n"
    )
    scene = parse_scene_text(text)
    assert scene.ambient == Ambient(0.1, (10, 20, 30))
    assert scene.camera == Camera((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)
    assert scene.objects == (Sphere((0.0, 0.0, 5.0), 1.0, (1, 2, 3)),)


def test_scene_without_objects_is_valid() -> None:
    scene = parse_scene_text(MINIMAL)
    assert scene.objects == ()


def test_several_lights_are_kept_in_order() -> None:
    scene = parse_scene_text(MINIMAL + "L 1,1,1 0.3 255,0,0\n")
    assert [light.position for light in scene.lights] == [(0.0, 5.0, 0.0), (1.0, 1.0, 1.0)]


@pytest.mark.parametrize("token", ["1", "-2.5", "+3.", ".5", "1e1", "2.5E-1"])
def test_number_formats_accepted(token: str) -> None:
    scene = parse_scene_text(MINIMAL + f"sp 0,0,{token} 1 0,0,0\n")
    sphere = scene.objects[0]
    assert isinstance(sphere, Sphere)
    assert sphere.center[2] == float(token)


@pytest.mark.parametrize("token", ["abc", "1,5", "nan", "inf", "1..2", "0x10", "1_0", ""])
def test_bad_numbers_rejected(token: str) -> None:
    with pytest.raises(SceneFormatError) as info:
        parse_scene_text(MINIMAL + f"sp 0,0,5 {token} 0,0,0\n")
    assert info.value.line == 4


def test_huge_number_rejected() -> None:
    with pytest.raises(SceneFormatError, match="вне допустимого диапазона"):
        parse_scene_text(MINIMAL + "sp 0,0,1e999 1 0,0,0\n")


@pytest.mark.parametrize(
    ("name", "line", "reason"),
    [
        (
            "ambient_ratio.rt",
            2,
            "яркость фонового света должна быть в диапазоне 0..1, получено 1.5",
        ),
        ("sphere_fields.rt", 7, "у элемента sp ожидается 3 полей, получено 2"),
        ("unknown_element.rt", 8, "неизвестный элемент 'cube'"),
        ("bad_number.rt", 6, "диаметр сферы: ожидается число, получено 'abc'"),
        (
            "bad_vector.rt",
            3,
            "позиция камеры: ожидается вектор из трёх чисел через запятую, получено '0,3'",
        ),
        (
            "color_range.rt",
            6,
            "цвет сферы: ожидаются три целых 0..255 через запятую, получено '256,50,50'",
        ),
        (
            "camera_direction_range.rt",
            3,
            "компоненты направления камеры должны быть в диапазоне [-1, 1], получено 0,-2,1",
        ),
        ("camera_direction_zero.rt", 3, "направление камеры не может быть нулевым вектором"),
        ("fov_range.rt", 3, "угол обзора должен быть в диапазоне (0, 180), получено 180"),
        (
            "light_brightness.rt",
            4,
            "яркость источника света должна быть в диапазоне 0..1, получено 2",
        ),
        ("sphere_diameter.rt", 6, "диаметр сферы должен быть больше 0, получено -3"),
        ("cylinder_height.rt", 8, "высота цилиндра должна быть больше 0, получено 0"),
        ("cylinder_fields.rt", 8, "у элемента cy ожидается 5 полей, получено 4"),
        ("plane_normal_zero.rt", 5, "нормаль плоскости не может быть нулевым вектором"),
        (
            "duplicate_ambient.rt",
            3,
            "фоновый свет A должен быть ровно один (уже задан в строке 2)",
        ),
        ("duplicate_camera.rt", 4, "камера C должна быть ровно одна (уже задана в строке 3)"),
        ("missing_ambient.rt", None, "в сцене нет фонового света A"),
        ("missing_camera.rt", None, "в сцене нет камеры C"),
        ("missing_light.rt", None, "в сцене нет источника света L"),
        ("empty.rt", None, "в сцене нет фонового света A"),
        ("not_a_scene.txt", None, "файл сцены должен иметь расширение .rt"),
        ("cone_diameter.rt", 6, "диаметр основания конуса должен быть больше 0, получено -2"),
        (
            "unknown_flag.rt",
            5,
            "неизвестный флаг 'shiny' у элемента pl: допускается только checker",
        ),
    ],
)
def test_bad_scene_reports_reason_and_line(name: str, line: int | None, reason: str) -> None:
    with pytest.raises(SceneFormatError) as info:
        parse_scene_file(SCENES / "bad" / name)
    assert info.value.line == line
    assert info.value.reason == reason
    expected = reason if line is None else f"строка {line}: {reason}"
    assert str(info.value) == expected


def test_every_bad_scene_is_covered() -> None:
    files = {path.name for path in (SCENES / "bad").iterdir()}
    assert len(files) >= 20
    for name in files:
        with pytest.raises(SceneFormatError):
            parse_scene_file(SCENES / "bad" / name)


def test_scene_format_error_is_rt_error() -> None:
    assert issubclass(SceneFormatError, RtError)
    assert not issubclass(SceneFormatError, OSError)


def test_missing_file_raises_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError, match=r"no_such\.rt"):
        parse_scene_file(tmp_path / "no_such.rt")


def test_utf8_bom_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "bom.rt"
    path.write_bytes(b"\xef\xbb\xbf" + MINIMAL.encode())
    assert parse_scene_file(path).ambient.ratio == 0.2


def test_binary_file_is_format_error(tmp_path: Path) -> None:
    path = tmp_path / "binary.rt"
    path.write_bytes(b"\xff\xfe\x00\x81garbage")
    with pytest.raises(SceneFormatError, match="UTF-8"):
        parse_scene_file(path)


def test_unknown_lowercase_identifier() -> None:
    with pytest.raises(SceneFormatError) as info:
        parse_scene_text(MINIMAL + "a 0.2 255,255,255\n")
    assert info.value.line == 4
    assert info.value.reason == "неизвестный элемент 'a'"
