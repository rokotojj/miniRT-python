"""Замеры производительности для таблиц в README.

Каждый случай рендерится repeat раз, в таблицу идут лучшее и медианное время.
Печатает таблицы Markdown и сведения о машине.

    python tools/bench.py [--repeat 5] [--naive] [--parallel]
"""

import argparse
import os
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):  # запуск как скрипта: python tools/bench.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from minirt.parser import parse_scene_file
from minirt.render import RenderOptions, render
from tools.naive import render_naive

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Case:
    scene: str
    width: int = 800
    height: int = 600
    options: RenderOptions = field(default_factory=RenderOptions)
    note: str = ""

    def flags(self) -> str:
        parts = []
        if self.options.specular:
            parts.append("--specular")
        if self.options.samples > 1:
            parts.append(f"--samples {self.options.samples}")
        if self.options.workers > 1:
            parts.append(f"--workers {self.options.workers}")
        return " ".join(parts) or "—"


MANDATORY = (
    Case("scenes/three.rt", note="норматив — 3 с"),
    Case("scenes/three.rt", 400, 300),
    Case("scenes/three.rt", 1600, 1200),
    Case("scenes/cylinders.rt", note="4 цилиндра"),
)

BONUS = (
    Case("scenes/phong.rt", options=RenderOptions(specular=True), note="блики"),
    Case("scenes/lights.rt", note="2 источника — 2 теневых луча"),
    Case("scenes/checker.rt", note="шахматная раскраска"),
    Case("scenes/cone.rt", note="4 конуса"),
    Case("scenes/three.rt", options=RenderOptions(samples=2), note="4 луча на пиксель"),
    Case("scenes/three.rt", options=RenderOptions(samples=3), note="9 лучей на пиксель"),
    Case(
        "scenes/showcase.rt",
        options=RenderOptions(specular=True, samples=2),
        note="все доп. задания",
    ),
)


def parallel_cases() -> tuple[Case, ...]:
    """Тяжёлый кадр с разным числом процессов: 1, 2, 4, … до числа ядер."""
    cores = os.cpu_count() or 1
    counts = sorted({1, 2, 4, cores} | ({8} if cores >= 8 else set()))
    return tuple(
        Case("scenes/three.rt", 1600, 1200, RenderOptions(samples=2, workers=n)) for n in counts
    )


def cpu_name() -> str:
    """Название процессора: platform.processor() пуст в Linux, тогда — /proc/cpuinfo."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def machine() -> str:
    return (
        f"{cpu_name()}, {os.cpu_count()} лог. ядер; {platform.system()} {platform.release()}; "
        f"Python {platform.python_version()}, NumPy {np.__version__}"
    )


def measure(case: Case, repeat: int) -> tuple[float, float]:
    scene = parse_scene_file(ROOT / case.scene)
    render(scene, 64, 48)  # прогрев: импорт и первые выделения памяти
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        render(scene, case.width, case.height, case.options)
        times.append(time.perf_counter() - start)
    return min(times), statistics.median(times)


def table(title: str, cases: Sequence[Case], repeat: int) -> list[float]:
    print(f"\n### {title}\n")
    print("| Сцена | Разрешение | Параметры | Лучшее, с | Медиана, с | Примечание |")
    print("|---|---|---|---|---|---|")
    medians = []
    for case in cases:
        best, median = measure(case, repeat)
        medians.append(median)
        print(
            f"| {case.scene} | {case.width}×{case.height} | {case.flags()} | {best:.2f} "
            f"| {median:.2f} | {case.note} |",
            flush=True,
        )
    return medians


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Замеры времени рендера")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--naive", action="store_true", help="попиксельный эталон")
    parser.add_argument("--bonus", action="store_true", help="дополнительные задания")
    parser.add_argument("--parallel", action="store_true", help="ускорение от --workers")
    args = parser.parse_args(argv)

    print(f"Машина: {machine()}")
    table("Обязательная часть", MANDATORY, args.repeat)
    if args.naive:
        scene = parse_scene_file(ROOT / "scenes/three.rt")
        start = time.perf_counter()
        render_naive(scene, 800, 600)
        elapsed = time.perf_counter() - start
        print(
            f"\nЧистый Python (tools/naive.py), scenes/three.rt 800×600: {elapsed:.2f} с, "
            f"{elapsed / 480_000 * 1e6:.1f} мкс на пиксель"
        )
    if args.bonus:
        table("Дополнительные задания", BONUS, args.repeat)
    if args.parallel:
        cases = parallel_cases()
        medians = table("Многопроцессность", cases, args.repeat)
        print("\n| Процессов | Медиана, с | Ускорение |\n|---|---|---|")
        for case, median in zip(cases, medians, strict=True):
            print(f"| {case.options.workers} | {median:.2f} | ×{medians[0] / median:.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
