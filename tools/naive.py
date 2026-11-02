"""Эталон для сравнения скорости: тот же алгоритм циклом по пикселям на чистом Python.

Кортежи и math вместо массивов NumPy: правильно, но в десятки раз медленнее.
Поддерживает обязательную часть (сфера, плоскость, цилиндр, один источник, тени).

    python tools/naive.py scenes/three.rt [--width 800 --height 600] [--out naive.png]
"""

import argparse
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

if __package__ in (None, ""):  # запуск как скрипта: python tools/naive.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import numpy.typing as npt

from minirt.image_io import save_png
from minirt.parser import parse_scene_file
from minirt.scene import Cylinder, Plane, Scene, Shape, Sphere

Vec: TypeAlias = tuple[float, float, float]

EPSILON = 1e-4
SHADOW_BIAS = 1e-3


def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def mul(a: Vec, k: float) -> Vec:
    return (a[0] * k, a[1] * k, a[2] * k)


def dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def normalize(a: Vec) -> Vec:
    return mul(a, 1.0 / math.sqrt(dot(a, a)))


def hit_sphere(o: Vec, d: Vec, s: Sphere) -> float:
    oc = sub(o, s.center)
    b = dot(oc, d)
    disc = b * b - (dot(oc, oc) - s.radius * s.radius)
    if disc < 0.0:
        return math.inf
    root = math.sqrt(disc)
    for t in (-b - root, -b + root):
        if t > EPSILON:
            return t
    return math.inf


def hit_plane(o: Vec, d: Vec, p: Plane) -> float:
    denominator = dot(d, p.normal)
    if abs(denominator) <= 1e-12:
        return math.inf
    t = dot(sub(p.point, o), p.normal) / denominator
    return t if t > EPSILON else math.inf


def hit_cylinder(o: Vec, d: Vec, cy: Cylinder) -> float:
    axis, half, r2 = cy.axis, cy.height / 2.0, cy.radius * cy.radius
    v = sub(o, cy.center)
    d_a, v_a = dot(d, axis), dot(v, axis)
    d_p, v_p = sub(d, mul(axis, d_a)), sub(v, mul(axis, v_a))
    a, b, c = dot(d_p, d_p), dot(d_p, v_p), dot(v_p, v_p) - r2
    best = math.inf
    disc = b * b - a * c
    if a > 1e-12 and disc >= 0.0:
        root = math.sqrt(disc)
        for t in ((-b - root) / a, (-b + root) / a):
            if EPSILON < t < best and abs(v_a + t * d_a) <= half:
                best = t
    if abs(d_a) > 1e-12:
        for level in (half, -half):
            t = (level - v_a) / d_a
            if EPSILON < t < best and c + 2.0 * b * t + a * t * t <= 0.0:
                best = t
    return best


def hit(o: Vec, d: Vec, shape: Shape) -> float:
    if isinstance(shape, Sphere):
        return hit_sphere(o, d, shape)
    if isinstance(shape, Plane):
        return hit_plane(o, d, shape)
    if isinstance(shape, Cylinder):
        return hit_cylinder(o, d, shape)
    raise NotImplementedError("эталон покрывает только обязательную часть: sp, pl, cy")


def normal_at(shape: Shape, p: Vec) -> Vec:
    if isinstance(shape, Sphere):
        return mul(sub(p, shape.center), 1.0 / shape.radius)
    if isinstance(shape, Plane):
        return shape.normal
    if not isinstance(shape, Cylinder):
        raise NotImplementedError("эталон покрывает только обязательную часть: sp, pl, cy")
    w = sub(p, shape.center)
    m = dot(w, shape.axis)
    radial = sub(w, mul(shape.axis, m))
    distance = math.sqrt(dot(radial, radial))
    if abs(abs(m) - shape.height / 2.0) < abs(distance - shape.radius):
        return mul(shape.axis, math.copysign(1.0, m))
    return mul(radial, 1.0 / max(distance, 1e-12))


def trace(scene: Scene, o: Vec, d: Vec) -> Vec:
    best, nearest = math.inf, None
    for shape in scene.objects:
        t = hit(o, d, shape)
        if t < best:
            best, nearest = t, shape
    if nearest is None:
        return (0.0, 0.0, 0.0)
    p = add(o, mul(d, best))
    n = normal_at(nearest, p)
    if dot(n, d) > 0.0:
        n = mul(n, -1.0)
    base = mul(nearest.color, 1.0 / 255.0)
    ambient = mul(scene.ambient.color, scene.ambient.ratio / 255.0)
    color = (base[0] * ambient[0], base[1] * ambient[1], base[2] * ambient[2])
    light = scene.lights[0]
    to_light = sub(light.position, p)
    cosine = dot(n, normalize(to_light))
    if cosine > 0.0:
        start = add(p, mul(n, SHADOW_BIAS))
        shadow = sub(light.position, start)
        distance = math.sqrt(dot(shadow, shadow))
        shadow_dir = mul(shadow, 1.0 / distance)
        if all(hit(start, shadow_dir, shape) >= distance for shape in scene.objects):
            k = cosine * light.brightness / 255.0
            lc = light.color
            color = (
                color[0] + base[0] * lc[0] * k,
                color[1] + base[1] * lc[1] * k,
                color[2] + base[2] * lc[2] * k,
            )
    return color


def render_naive(scene: Scene, width: int, height: int) -> list[list[Vec]]:
    """Кадр списком строк, цвета 0..1."""
    camera = scene.camera
    forward = normalize(camera.direction)
    right = cross((0.0, 1.0, 0.0), forward)
    if dot(right, right) < 1e-18:
        right = cross((0.0, 0.0, 1.0), forward)
    right = normalize(right)
    up = cross(forward, right)
    half_w = math.tan(math.radians(camera.fov) / 2.0)
    half_h = half_w * height / width
    rows = []
    for j in range(height):
        v = (1.0 - 2.0 * (j + 0.5) / height) * half_h
        row = []
        for i in range(width):
            u = (2.0 * (i + 0.5) / width - 1.0) * half_w
            d = normalize(add(forward, add(mul(right, u), mul(up, v))))
            row.append(trace(scene, camera.position, d))
        rows.append(row)
    return rows


def to_image(rows: list[list[Vec]]) -> npt.NDArray[np.uint8]:
    colors = np.clip(np.array(rows, dtype=np.float64), 0.0, 1.0)
    return np.rint(colors * 255.0).astype(np.uint8)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Попиксельный трассировщик на чистом Python")
    parser.add_argument("scene")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    scene = parse_scene_file(args.scene)
    start = time.perf_counter()
    rows = render_naive(scene, args.width, args.height)
    elapsed = time.perf_counter() - start
    pixels = args.width * args.height
    print(
        f"{args.scene}: {args.width}x{args.height}, чистый Python — {elapsed:.2f} с "
        f"({elapsed / pixels * 1e6:.1f} мкс на пиксель)",
        file=sys.stderr,
    )
    if args.out:
        save_png(args.out, to_image(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
