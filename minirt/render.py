"""Построение кадра: первичные лучи → ближайшие пересечения → цвет пикселя.

Лучи обрабатываются массивами (N, 3), цикл остаётся только по объектам сцены.
Кадр считается полосами строк примерно по BAND_PIXELS лучей: промежуточные массивы
полосы помещаются в кэш процессора, и память не растёт с размером кадра.
С workers > 1 кадр делится на крупные полосы, которые считаются параллельно
в ProcessPoolExecutor.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from minirt.camera import ray_directions
from minirt.intersect import closest_hit, surface_normals
from minirt.lighting import shade
from minirt.pattern import surface_colors
from minirt.scene import Scene
from minirt.vec import FloatArray, dot, vec3

UInt8Array: TypeAlias = npt.NDArray[np.uint8]

BAND_PIXELS = 1 << 16
# Полос на процесс: несколько полос на каждый процесс выравнивают нагрузку, когда
# объекты сосредоточены в части кадра (небо считается быстрее пола).
STRIPES_PER_WORKER = 4


@dataclass(frozen=True)
class RenderOptions:
    """Параметры построения кадра сверх обязательной части."""

    specular: bool = False  # блики Фонга
    samples: int = 1  # сглаживание: samples² лучей на пиксель по регулярной сетке
    workers: int = 1  # число процессов


def trace(
    scene: Scene, origins: FloatArray, directions: FloatArray, *, specular: bool = False
) -> FloatArray:
    """Цвета (N, 3) с компонентами 0..1 для лучей; промах — чёрный фон."""
    t, index = closest_hit(scene.objects, origins, directions)
    colors = np.zeros(directions.shape)
    hit = index >= 0
    if not hit.any():
        return colors
    hit_index = index[hit]
    hit_directions = directions[hit]
    points = origins[hit] + hit_directions * t[hit][:, np.newaxis]
    normals = np.empty_like(points)
    base_colors = np.empty_like(points)
    for number, shape in enumerate(scene.objects):
        mine = hit_index == number
        if mine.any():
            normals[mine] = surface_normals(shape, points[mine])
            base_colors[mine] = surface_colors(shape, points[mine], normals[mine])
    # нормаль для освещения обращена к наблюдателю: видна обратная сторона — берём −n
    normals[dot(normals, hit_directions) > 0.0] *= -1.0
    to_viewer = -hit_directions if specular else None
    colors[hit] = shade(scene, points, normals, base_colors, to_viewer)
    return colors


def to_rgb8(colors: FloatArray) -> UInt8Array:
    """Цвета 0..1 → байты 0..255 с ограничением диапазона и округлением."""
    return np.rint(np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)


def render_rows(
    scene: Scene, width: int, height: int, first: int, last: int, options: RenderOptions
) -> FloatArray:
    """Цвета 0..1 строк [first, last) кадра — массив (last − first, width, 3).

    При samples = N через пиксель проходят N² лучей по регулярной сетке со смещениями
    (k + 0,5) / N, цвет пикселя — их среднее.
    """
    offsets = (np.arange(options.samples) + 0.5) / options.samples
    total = np.zeros((last - first, width, 3))
    for dy in offsets:
        for dx in offsets:
            directions = ray_directions(
                scene.camera, width, height, offset=(float(dx), float(dy)), rows=(first, last)
            ).reshape(-1, 3)
            origins = np.broadcast_to(vec3(scene.camera.position), directions.shape)
            colors = trace(scene, origins, directions, specular=options.specular)
            total += colors.reshape(last - first, width, 3)
    return total / (options.samples * options.samples)


def bands(
    width: int, height: int, samples: int = 1, first: int = 0, last: int | None = None
) -> list[tuple[int, int]]:
    """Разбиение строк [first, last) на полосы примерно по BAND_PIXELS лучей."""
    last = height if last is None else last
    step = max(1, BAND_PIXELS // (width * samples * samples))
    return [(row, min(last, row + step)) for row in range(first, last, step)]


def stripes(height: int, count: int) -> list[tuple[int, int]]:
    """Разбиение строк кадра на count почти равных полос (пустые отбрасываются)."""
    edges = np.linspace(0, height, count + 1).round().astype(int).tolist()
    return [(a, b) for a, b in pairwise(edges) if a < b]


def render_stripe(
    scene: Scene, width: int, height: int, first: int, last: int, options: RenderOptions
) -> UInt8Array:
    """Строки [first, last) кадра в байтах. Функция модуля — её можно передать в процесс."""
    image = np.empty((last - first, width, 3), dtype=np.uint8)
    for a, b in bands(width, height, options.samples, first, last):
        image[a - first : b - first] = to_rgb8(render_rows(scene, width, height, a, b, options))
    return image


def render(
    scene: Scene, width: int, height: int, options: RenderOptions | None = None
) -> UInt8Array:
    """Изображение сцены — массив (height, width, 3) типа uint8."""
    options = options or RenderOptions()
    if options.workers <= 1:
        return render_stripe(scene, width, height, 0, height, options)
    image = np.empty((height, width, 3), dtype=np.uint8)
    parts = stripes(height, options.workers * STRIPES_PER_WORKER)
    with ProcessPoolExecutor(max_workers=options.workers) as pool:
        futures = [
            pool.submit(render_stripe, scene, width, height, a, b, options) for a, b in parts
        ]
        for (a, b), future in zip(parts, futures, strict=True):
            image[a:b] = future.result()
    return image
