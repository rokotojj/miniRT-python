"""Пересечения лучей с объектами сцены и нормали в точках попадания.

Функции пересечения получают начала (N, 3) и единичные направления (N, 3) всех лучей
сразу и возвращают расстояния t формы (N,): np.inf там, где пересечения нет.
Корни не больше EPSILON считаются точкой на поверхности, с которой луч выходит.
"""

from collections.abc import Sequence

import numpy as np

from minirt.scene import Cone, Cylinder, Plane, Shape, Sphere
from minirt.vec import BoolArray, FloatArray, IntArray, dot, norm, vec3

EPSILON = 1e-4
# |D·n| или квадрат поперечной составляющей меньше этого — луч считается параллельным
PARALLEL = 1e-12


def no_hit(directions: FloatArray) -> FloatArray:
    """Массив расстояний «пересечения нет» для заданных лучей."""
    return np.full(directions.shape[:-1], np.inf)


def nearest_root(t1: FloatArray, t2: FloatArray, valid: BoolArray) -> FloatArray:
    """Наименьший из корней t1 ≤ t2, больший EPSILON; np.inf, если таких нет или valid ложно."""
    t = np.where(t1 > EPSILON, t1, np.where(t2 > EPSILON, t2, np.inf))
    return np.where(valid, t, np.inf)


def intersect_sphere(origins: FloatArray, directions: FloatArray, sphere: Sphere) -> FloatArray:
    """Пересечение со сферой: t² + 2bt + c = 0, b = (O − C)·D, c = |O − C|² − r²."""
    oc = origins - vec3(sphere.center)
    b = dot(oc, directions)
    c = dot(oc, oc) - sphere.radius * sphere.radius
    discriminant = b * b - c
    root = np.sqrt(np.maximum(discriminant, 0.0))
    return nearest_root(-b - root, -b + root, discriminant >= 0.0)


def sphere_normals(sphere: Sphere, points: FloatArray) -> FloatArray:
    """Внешняя нормаль сферы (P − C) / r."""
    return (points - vec3(sphere.center)) / sphere.radius


def intersect_plane(origins: FloatArray, directions: FloatArray, plane: Plane) -> FloatArray:
    """Пересечение с плоскостью: t = (P₀ − O)·n / (D·n); параллельный луч и t ≤ ε — промах."""
    normal = vec3(plane.normal)
    denominator = dot(directions, normal)
    facing = np.abs(denominator) > PARALLEL
    t = dot(vec3(plane.point) - origins, normal) / np.where(facing, denominator, 1.0)
    return np.where(facing & (t > EPSILON), t, np.inf)


def plane_normals(plane: Plane, points: FloatArray) -> FloatArray:
    """Нормаль плоскости постоянна."""
    return np.broadcast_to(vec3(plane.normal), points.shape).copy()


def bounding_candidates(
    origins: FloatArray, directions: FloatArray, center: FloatArray, radius: float
) -> IntArray:
    """Номера лучей, которые проходят через описанную сферу объекта впереди начала.

    Дешёвая проверка отсекает большинство лучей до дорогого точного пересечения.
    """
    oc = origins - center
    b = dot(oc, directions)
    c = dot(oc, oc) - radius * radius
    discriminant = b * b - c
    far = -b + np.sqrt(np.maximum(discriminant, 0.0))
    return np.flatnonzero((discriminant >= 0.0) & (far > EPSILON))


def intersect_cylinder(
    origins: FloatArray, directions: FloatArray, cylinder: Cylinder
) -> FloatArray:
    """Пересечение с конечным цилиндром; лучи вне описанной сферы сразу получают np.inf."""
    bound = float(np.hypot(cylinder.radius, cylinder.height / 2.0))
    rays = bounding_candidates(origins, directions, vec3(cylinder.center), bound)
    t = no_hit(directions)
    if rays.size:
        t[rays] = _cylinder_distances(origins[rays], directions[rays], cylinder)
    return t


def _cylinder_distances(
    origins: FloatArray, directions: FloatArray, cylinder: Cylinder
) -> FloatArray:
    """Пересечение с конечным цилиндром: боковая поверхность и две крышки-диска.

    D и v = O − C раскладываются на составляющие вдоль оси a и поперёк:
    Dₚ = D − (D·a)a, vₚ = v − (v·a)a; бок — корни |vₚ + t·Dₚ|² = r² с проекцией
    m = (v + tD)·a в пределах [−h/2, h/2]; крышки — плоскости m = ±h/2 внутри радиуса.
    """
    axis = vec3(cylinder.axis)
    radius2 = cylinder.radius * cylinder.radius
    half = cylinder.height / 2.0
    v = origins - vec3(cylinder.center)
    d_a = dot(directions, axis)
    v_a = dot(v, axis)
    d_p = directions - d_a[:, np.newaxis] * axis
    v_p = v - v_a[:, np.newaxis] * axis
    a = dot(d_p, d_p)
    b = dot(d_p, v_p)
    c = dot(v_p, v_p) - radius2

    # боковая поверхность: a·t² + 2b·t + c = 0; при a ≈ 0 луч параллелен оси
    discriminant = b * b - a * c
    side = (a > PARALLEL) & (discriminant >= 0.0)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    safe_a = np.where(side, a, 1.0)
    best = no_hit(directions)
    for t in ((-b - root) / safe_a, (-b + root) / safe_a):
        within = np.abs(v_a + t * d_a) <= half
        best = np.minimum(best, np.where(side & within & (t > EPSILON), t, np.inf))

    # крышки: |vₚ + t·Dₚ|² = (c + r²) + 2b·t + a·t² ≤ r²
    across = np.abs(d_a) > PARALLEL
    safe_d_a = np.where(across, d_a, 1.0)
    for level in (half, -half):
        t = (level - v_a) / safe_d_a
        inside = c + 2.0 * b * t + a * t * t <= 0.0
        best = np.minimum(best, np.where(across & inside & (t > EPSILON), t, np.inf))
    return best


def cylinder_normals(cylinder: Cylinder, points: FloatArray) -> FloatArray:
    """Нормаль бока normalize(w − m·a), w = P − C, m = w·a; нормаль крышки ±a.

    Точка относится к той части, уравнению которой она удовлетворяет точнее.
    """
    axis = vec3(cylinder.axis)
    w = points - vec3(cylinder.center)
    m = dot(w, axis)
    radial = w - m[:, np.newaxis] * axis
    distance = norm(radial)
    on_cap = np.abs(np.abs(m) - cylinder.height / 2.0) < np.abs(distance - cylinder.radius)
    side_normals = radial / np.maximum(distance, 1e-12)[:, np.newaxis]
    cap_normals = np.sign(m)[:, np.newaxis] * axis
    return np.where(on_cap[:, np.newaxis], cap_normals, side_normals)


def intersect_cone(origins: FloatArray, directions: FloatArray, cone: Cone) -> FloatArray:
    """Пересечение с конусом; лучи вне описанной сферы сразу получают np.inf."""
    axis = vec3(cone.axis)
    center = vec3(cone.apex) + axis * (cone.height / 2.0)
    bound = float(np.hypot(cone.radius, cone.height / 2.0))
    rays = bounding_candidates(origins, directions, center, bound)
    t = no_hit(directions)
    if rays.size:
        t[rays] = _cone_distances(origins[rays], directions[rays], cone)
    return t


def _cone_distances(origins: FloatArray, directions: FloatArray, cone: Cone) -> FloatArray:
    """Пересечение с конусом: боковая поверхность и основание-диск.

    w = P − V (V — вершина), m = w·a. Бок: |w|² = (1 + k²)·m², k = R / h, — квадратное
    уравнение по t с коэффициентами a′ = 1 − (1 + k²)(D·a)², b′ = D·v − (1 + k²)(D·a)(v·a),
    c′ = v·v − (1 + k²)(v·a)², v = O − V. Корень засчитывается при 0 ≤ m ≤ h: m < 0 —
    зеркальная половина конуса за вершиной. Основание — диск m = h радиуса R.
    """
    axis = vec3(cone.axis)
    k2 = 1.0 + (cone.radius / cone.height) ** 2
    v = origins - vec3(cone.apex)
    d_a = dot(directions, axis)
    v_a = dot(v, axis)
    d_v = dot(directions, v)
    v_v = dot(v, v)
    a = 1.0 - k2 * d_a * d_a
    b = d_v - k2 * d_a * v_a
    c = v_v - k2 * v_a * v_a

    discriminant = b * b - a * c
    quadratic = (np.abs(a) > PARALLEL) & (discriminant >= 0.0)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    safe_a = np.where(np.abs(a) > PARALLEL, a, 1.0)
    # a ≈ 0: луч параллелен образующей, уравнение линейное 2b′·t + c′ = 0
    linear = (np.abs(a) <= PARALLEL) & (np.abs(b) > PARALLEL)
    safe_b = np.where(linear, b, 1.0)
    best = no_hit(directions)
    candidates = (
        ((-b - root) / safe_a, quadratic),
        ((-b + root) / safe_a, quadratic),
        (-c / (2.0 * safe_b), linear),
    )
    for t, solvable in candidates:
        m = v_a + t * d_a
        valid = solvable & (t > EPSILON) & (m >= 0.0) & (m <= cone.height)
        best = np.minimum(best, np.where(valid, t, np.inf))

    across = np.abs(d_a) > PARALLEL
    t = (cone.height - v_a) / np.where(across, d_a, 1.0)
    radial2 = v_v + 2.0 * t * d_v + t * t - cone.height * cone.height
    inside = radial2 <= cone.radius * cone.radius
    return np.minimum(best, np.where(across & inside & (t > EPSILON), t, np.inf))


def cone_normals(cone: Cone, points: FloatArray) -> FloatArray:
    """Нормаль бока normalize(w − (1 + k²)·m·a), нормаль основания +a.

    Точка относится к той части, уравнению которой она удовлетворяет точнее; в самой
    вершине нормаль не определена, там берётся −a.
    """
    axis = vec3(cone.axis)
    slope = cone.radius / cone.height
    w = points - vec3(cone.apex)
    m = dot(w, axis)
    radial = norm(w - m[:, np.newaxis] * axis)
    on_base = np.abs(m - cone.height) < np.abs(radial - slope * m)
    side = w - ((1.0 + slope * slope) * m)[:, np.newaxis] * axis
    length = norm(side)
    side_normals = np.where(
        (length > 1e-12)[:, np.newaxis],
        side / np.maximum(length, 1e-12)[:, np.newaxis],
        -axis,
    )
    return np.where(on_base[:, np.newaxis], axis, side_normals)


def intersect(shape: Shape, origins: FloatArray, directions: FloatArray) -> FloatArray:
    """Расстояния до объекта любого типа."""
    if isinstance(shape, Sphere):
        return intersect_sphere(origins, directions, shape)
    if isinstance(shape, Plane):
        return intersect_plane(origins, directions, shape)
    if isinstance(shape, Cylinder):
        return intersect_cylinder(origins, directions, shape)
    return intersect_cone(origins, directions, shape)


def surface_normals(shape: Shape, points: FloatArray) -> FloatArray:
    """Единичные нормали объекта в точках его поверхности (без разворота к наблюдателю)."""
    if isinstance(shape, Sphere):
        return sphere_normals(shape, points)
    if isinstance(shape, Plane):
        return plane_normals(shape, points)
    if isinstance(shape, Cylinder):
        return cylinder_normals(shape, points)
    return cone_normals(shape, points)


def closest_hit(
    objects: Sequence[Shape], origins: FloatArray, directions: FloatArray
) -> tuple[FloatArray, IntArray]:
    """Ближайшее попадание перебором объектов: расстояния (N,) и индексы (N,), −1 — мимо."""
    best_t = no_hit(directions)
    best_index = np.full(best_t.shape, -1, dtype=np.intp)
    for index, shape in enumerate(objects):
        t = intersect(shape, origins, directions)
        closer = t < best_t
        best_t = np.where(closer, t, best_t)
        best_index = np.where(closer, index, best_index)
    return best_t, best_index


def occluded(
    objects: Sequence[Shape], origins: FloatArray, directions: FloatArray, distances: FloatArray
) -> BoolArray:
    """Заслонены ли отрезки [O, O + distance·D] каким-либо объектом — для теневых лучей.

    Каждый следующий объект проверяется только на ещё не заслонённых лучах.
    """
    blocked = np.zeros(distances.shape, dtype=np.bool_)
    for shape in objects:
        rest = np.flatnonzero(~blocked)
        if rest.size == 0:
            break
        t = intersect(shape, origins[rest], directions[rest])
        blocked[rest[t < distances[rest]]] = True
    return blocked
