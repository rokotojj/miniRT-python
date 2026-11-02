"""Освещение точек попадания: фоновый свет, диффузное отражение по Ламберту, жёсткие тени.

Цвет точки = ratio_A · c_A · c_obj + Σ яркость_L · c_L · c_obj · max(0, n·l),
где l — единичный вектор из точки к источнику, произведение цветов покомпонентное,
сумма — по всем источникам L. Если отрезок от точки до источника пересекает объект,
вклад этого источника равен нулю.
С бликами к каждому источнику добавляется k · max(0, n·h)^p · яркость_L · c_L.
"""

from collections.abc import Sequence

import numpy as np

from minirt.intersect import occluded
from minirt.scene import Light, Scene, Shape
from minirt.vec import BoolArray, FloatArray, color01, dot, norm, vec3

# Сдвиг начала теневого луча вдоль нормали: без него луч «находит» ту же поверхность
# в точке старта, и сцена покрывается точечным шумом — теневым акне.
SHADOW_BIAS = 1e-3
# Блики Фонга (дополнительное задание, параметр --specular): коэффициент и показатель
SPECULAR_K = 0.5
SPECULAR_POWER = 32.0


def lambert(normals: FloatArray, to_light: FloatArray) -> FloatArray:
    """Косинус угла падения max(0, n·l) для единичных n и l."""
    return np.maximum(dot(normals, to_light), 0.0)


def in_shadow(
    objects: Sequence[Shape], light: Light, points: FloatArray, normals: FloatArray
) -> BoolArray:
    """Маска точек, от которых источник заслонён другим объектом."""
    origins = points + normals * SHADOW_BIAS
    to_light = vec3(light.position) - origins
    distance = np.maximum(norm(to_light), 1e-12)
    return occluded(objects, origins, to_light / distance[:, np.newaxis], distance)


def illumination(
    objects: Sequence[Shape], light: Light, points: FloatArray, normals: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Косинус падения max(0, n·l) с учётом тени (в тени — 0) и единичные векторы l к источнику."""
    to_light = vec3(light.position) - points
    distance = np.maximum(norm(to_light), 1e-12)
    to_light /= distance[:, np.newaxis]
    cosine = lambert(normals, to_light)
    lit = np.flatnonzero(cosine > 0.0)  # теневые лучи нужны только освещённым точкам
    if lit.size and objects:
        cosine[lit[in_shadow(objects, light, points[lit], normals[lit])]] = 0.0
    return cosine, to_light


def phong(
    normals: FloatArray, to_light: FloatArray, to_viewer: FloatArray, cosine: FloatArray
) -> FloatArray:
    """Зеркальный множитель k · max(0, n·h)^p, h = normalize(l + v); 0 там, где нет света."""
    half = to_light + to_viewer
    half /= np.maximum(norm(half), 1e-12)[:, np.newaxis]
    highlight = SPECULAR_K * np.maximum(dot(normals, half), 0.0) ** SPECULAR_POWER
    return np.where(cosine > 0.0, highlight, 0.0)


def shade(
    scene: Scene,
    points: FloatArray,
    normals: FloatArray,
    base_colors: FloatArray,
    to_viewer: FloatArray | None = None,
) -> FloatArray:
    """Цвета (N, 3) точек с нормалями normals и собственными цветами base_colors (0..1).

    to_viewer — единичные векторы из точек к камере; если заданы, добавляются блики Фонга
    k · max(0, n·h)^p · яркость_L · c_L (цвет источника, не объекта).
    """
    colors = base_colors * (scene.ambient.ratio * color01(scene.ambient.color))
    for light in scene.lights:  # вклады источников суммируются, тени — для каждого свои
        cosine, to_light = illumination(scene.objects, light, points, normals)
        intensity = light.brightness * color01(light.color)
        colors += base_colors * intensity * cosine[:, np.newaxis]
        if to_viewer is not None:
            colors += intensity * phong(normals, to_light, to_viewer, cosine)[:, np.newaxis]
    return colors
