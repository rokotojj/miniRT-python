"""Разбор и проверка файла сцены формата .rt.

Каждая непустая строка — один элемент: идентификатор и поля через пробелы.
Векторы и цвета — три числа через запятую без пробелов, текст после # — комментарий.
Любое нарушение формата — SceneFormatError с причиной и номером строки.
Отсутствие файла — OSError, его парсер не перехватывает.
"""

import dataclasses
import math
import os
import re
from collections.abc import Callable, Sequence

from minirt.errors import SceneFormatError
from minirt.scene import (
    Ambient,
    Camera,
    Color,
    Cone,
    Cylinder,
    Light,
    Plane,
    Scene,
    Shape,
    Sphere,
    Vec3,
)

_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_COLOR_PART_RE = re.compile(r"\d{1,3}")
_ZERO_LENGTH = 1e-9


def _expect_fields(ident: str, fields: Sequence[str], count: int, line: int) -> None:
    if len(fields) != count:
        raise SceneFormatError(
            f"у элемента {ident} ожидается {count} полей, получено {len(fields)}", line
        )


def _number(token: str, what: str, line: int) -> float:
    if _NUMBER_RE.fullmatch(token) is None:
        raise SceneFormatError(f"{what}: ожидается число, получено '{token}'", line)
    value = float(token)
    if not math.isfinite(value):
        raise SceneFormatError(f"{what}: число вне допустимого диапазона: '{token}'", line)
    return value


def _vector(token: str, what: str, line: int) -> Vec3:
    parts = token.split(",")
    if len(parts) != 3 or any(_NUMBER_RE.fullmatch(part) is None for part in parts):
        raise SceneFormatError(
            f"{what}: ожидается вектор из трёх чисел через запятую, получено '{token}'", line
        )
    x, y, z = (_number(part, what, line) for part in parts)
    return (x, y, z)


def _color(token: str, what: str, line: int) -> Color:
    parts = token.split(",")
    if (
        len(parts) != 3
        or any(_COLOR_PART_RE.fullmatch(part) is None for part in parts)
        or any(int(part) > 255 for part in parts)
    ):
        raise SceneFormatError(
            f"{what}: ожидаются три целых 0..255 через запятую, получено '{token}'", line
        )
    r, g, b = (int(part) for part in parts)
    return (r, g, b)


def _ratio(token: str, what: str, line: int) -> float:
    value = _number(token, what, line)
    if not 0.0 <= value <= 1.0:
        raise SceneFormatError(f"{what} должна быть в диапазоне 0..1, получено {token}", line)
    return value


def _positive(token: str, what: str, line: int, must: str = "должен") -> float:
    value = _number(token, what, line)
    if value <= 0.0:
        raise SceneFormatError(f"{what} {must} быть больше 0, получено {token}", line)
    return value


def _direction(token: str, genitive: str, nominative: str, line: int) -> Vec3:
    """Направление: компоненты в [-1, 1], вектор не нулевой; возвращается нормированным."""
    x, y, z = _vector(token, nominative, line)
    if any(not -1.0 <= c <= 1.0 for c in (x, y, z)):
        raise SceneFormatError(
            f"компоненты {genitive} должны быть в диапазоне [-1, 1], получено {token}", line
        )
    length = math.sqrt(x * x + y * y + z * z)
    if length < _ZERO_LENGTH:
        raise SceneFormatError(f"{nominative} не может быть нулевым вектором", line)
    return (x / length, y / length, z / length)


def _parse_ambient(fields: Sequence[str], line: int) -> Ambient:
    _expect_fields("A", fields, 2, line)
    ratio = _ratio(fields[0], "яркость фонового света", line)
    return Ambient(ratio, _color(fields[1], "цвет фонового света", line))


def _parse_camera(fields: Sequence[str], line: int) -> Camera:
    _expect_fields("C", fields, 3, line)
    position = _vector(fields[0], "позиция камеры", line)
    direction = _direction(fields[1], "направления камеры", "направление камеры", line)
    fov = _number(fields[2], "угол обзора", line)
    if not 0.0 < fov < 180.0:
        raise SceneFormatError(
            f"угол обзора должен быть в диапазоне (0, 180), получено {fields[2]}", line
        )
    return Camera(position, direction, fov)


def _parse_light(fields: Sequence[str], line: int) -> Light:
    _expect_fields("L", fields, 3, line)
    position = _vector(fields[0], "позиция источника света", line)
    brightness = _ratio(fields[1], "яркость источника света", line)
    return Light(position, brightness, _color(fields[2], "цвет источника света", line))


def _parse_sphere(fields: Sequence[str], line: int) -> Shape:
    _expect_fields("sp", fields, 3, line)
    center = _vector(fields[0], "центр сферы", line)
    diameter = _positive(fields[1], "диаметр сферы", line)
    return Sphere(center, diameter / 2.0, _color(fields[2], "цвет сферы", line))


def _parse_plane(fields: Sequence[str], line: int) -> Shape:
    _expect_fields("pl", fields, 3, line)
    point = _vector(fields[0], "точка плоскости", line)
    normal = _direction(fields[1], "нормали плоскости", "нормаль плоскости", line)
    return Plane(point, normal, _color(fields[2], "цвет плоскости", line))


def _parse_cylinder(fields: Sequence[str], line: int) -> Shape:
    _expect_fields("cy", fields, 5, line)
    center = _vector(fields[0], "центр цилиндра", line)
    axis = _direction(fields[1], "оси цилиндра", "ось цилиндра", line)
    diameter = _positive(fields[2], "диаметр цилиндра", line)
    height = _positive(fields[3], "высота цилиндра", line, "должна")
    color = _color(fields[4], "цвет цилиндра", line)
    return Cylinder(center, axis, diameter / 2.0, height, color)


def _parse_cone(fields: Sequence[str], line: int) -> Shape:
    _expect_fields("co", fields, 5, line)
    apex = _vector(fields[0], "вершина конуса", line)
    axis = _direction(fields[1], "оси конуса", "ось конуса", line)
    diameter = _positive(fields[2], "диаметр основания конуса", line)
    height = _positive(fields[3], "высота конуса", line, "должна")
    color = _color(fields[4], "цвет конуса", line)
    return Cone(apex, axis, diameter / 2.0, height, color)


_OBJECT_PARSERS: dict[str, Callable[[Sequence[str], int], Shape]] = {
    "sp": _parse_sphere,
    "pl": _parse_plane,
    "cy": _parse_cylinder,
    "co": _parse_cone,
}
_OBJECT_FIELDS = {"sp": 3, "pl": 3, "cy": 5, "co": 5}
CHECKER_FLAG = "checker"


def _parse_object(ident: str, fields: Sequence[str], line: int) -> Shape:
    """Объект сцены; необязательное последнее поле checker включает шахматную раскраску."""
    checker = len(fields) == _OBJECT_FIELDS[ident] + 1 and fields[-1] == CHECKER_FLAG
    if len(fields) == _OBJECT_FIELDS[ident] + 1 and not checker:
        raise SceneFormatError(
            f"неизвестный флаг '{fields[-1]}' у элемента {ident}: допускается только checker",
            line,
        )
    shape = _OBJECT_PARSERS[ident](fields[:-1] if checker else fields, line)
    return dataclasses.replace(shape, checker=True) if checker else shape


def parse_scene_text(text: str) -> Scene:
    """Разобрать текст сцены; при нарушении формата — SceneFormatError."""
    ambient: Ambient | None = None
    camera: Camera | None = None
    ambient_line = camera_line = 0
    lights: list[Light] = []
    objects: list[Shape] = []

    for number, raw in enumerate(text.splitlines(), start=1):
        content = raw.split("#", 1)[0].strip()
        if not content:
            continue
        ident, *fields = content.split()
        if ident == "A":
            if ambient is not None:
                raise SceneFormatError(
                    f"фоновый свет A должен быть ровно один (уже задан в строке {ambient_line})",
                    number,
                )
            ambient, ambient_line = _parse_ambient(fields, number), number
        elif ident == "C":
            if camera is not None:
                raise SceneFormatError(
                    f"камера C должна быть ровно одна (уже задана в строке {camera_line})", number
                )
            camera, camera_line = _parse_camera(fields, number), number
        elif ident == "L":
            lights.append(_parse_light(fields, number))
        elif ident in _OBJECT_PARSERS:
            objects.append(_parse_object(ident, fields, number))
        else:
            raise SceneFormatError(f"неизвестный элемент '{ident}'", number)

    if ambient is None:
        raise SceneFormatError("в сцене нет фонового света A")
    if camera is None:
        raise SceneFormatError("в сцене нет камеры C")
    if not lights:
        raise SceneFormatError("в сцене нет источника света L")
    return Scene(ambient, camera, tuple(lights), tuple(objects))


def parse_scene_file(path: str | os.PathLike[str]) -> Scene:
    """Прочитать и разобрать файл сцены. OSError (нет файла, нет прав) не перехватывается."""
    if not os.fspath(path).endswith(".rt"):
        raise SceneFormatError("файл сцены должен иметь расширение .rt")
    with open(path, encoding="utf-8-sig") as file:
        try:
            text = file.read()
        except UnicodeDecodeError:
            raise SceneFormatError("файл сцены должен быть текстовым в кодировке UTF-8") from None
    return parse_scene_text(text)
