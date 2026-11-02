"""Шахматная раскраска объектов с флагом checker (дополнительное задание).

Клетка точки — сумма номеров ⌊x/s⌋ + ⌊y/s⌋ + ⌊z/s⌋. При чётной сумме точка получает
цвет объекта, при нечётной — его затемнённый вариант.
"""

import numpy as np

from minirt.scene import Shape
from minirt.vec import BoolArray, FloatArray, color01

CHECKER_SIZE = 1.0  # сторона клетки s в единицах сцены
CHECKER_DARK = 0.2  # второй цвет — цвет объекта, умноженный на это число
# Точка сдвигается внутрь объекта на эту величину: грань, лежащая ровно на границе
# клеток (плоскость y = −1), иначе мерцала бы между двумя клетками из-за погрешности.
_INSET = 1e-4


def checker_odd(points: FloatArray, normals: FloatArray) -> BoolArray:
    """Нечётные клетки для точек (N, 3) с внешними нормалями (N, 3)."""
    cells = np.floor((points - normals * _INSET) / CHECKER_SIZE).astype(np.int64)
    odd: BoolArray = cells.sum(axis=-1) % 2 == 1
    return odd


def surface_colors(shape: Shape, points: FloatArray, normals: FloatArray) -> FloatArray:
    """Собственные цвета (N, 3) объекта в точках попадания с учётом шахматной раскраски."""
    base = np.broadcast_to(color01(shape.color), points.shape)
    if not shape.checker:
        return base.copy()
    odd = checker_odd(points, normals)
    return np.where(odd[:, np.newaxis], base * CHECKER_DARK, base)
