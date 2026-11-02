"""Пересобрать снимки для README в каталоге docs/.

python tools/gallery.py
"""

import sys
from pathlib import Path

if __package__ in (None, ""):  # запуск как скрипта: python tools/gallery.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from minirt.image_io import save_png
from minirt.parser import parse_scene_file
from minirt.render import RenderOptions, UInt8Array, render

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

SHOTS = {
    "screenshot.png": ("scenes/three.rt", RenderOptions()),
    "cylinders.png": ("scenes/cylinders.rt", RenderOptions()),
    "phong.png": ("scenes/phong.rt", RenderOptions(specular=True)),
    "lights.png": ("scenes/lights.rt", RenderOptions()),
    "checker.png": ("scenes/checker.rt", RenderOptions(samples=2)),
    "cone.png": ("scenes/cone.rt", RenderOptions(samples=2)),
    "showcase.png": ("scenes/showcase.rt", RenderOptions(specular=True, samples=3, workers=2)),
}


def antialias_comparison() -> UInt8Array:
    """Край красного шара из three.rt: 1 луч на пиксель слева, 9 справа, увеличение ×4."""
    scene = parse_scene_file(ROOT / "scenes/three.rt")
    halves = []
    for samples in (1, 3):
        image = render(scene, 800, 600, RenderOptions(samples=samples))
        crop = image[230:290, 300:380]
        halves.append(np.repeat(np.repeat(crop, 4, axis=0), 4, axis=1))
    gap = np.full((halves[0].shape[0], 8, 3), 255, dtype=np.uint8)
    return np.concatenate([halves[0], gap, halves[1]], axis=1)


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    for name, (scene_path, options) in SHOTS.items():
        save_png(DOCS / name, render(parse_scene_file(ROOT / scene_path), 800, 600, options))
        print(f"docs/{name} ← {scene_path}")
    save_png(DOCS / "antialias.png", antialias_comparison())
    print("docs/antialias.png ← сравнение сглаживания")
    return 0


if __name__ == "__main__":
    sys.exit(main())
