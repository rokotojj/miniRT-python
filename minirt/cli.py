"""Точка входа: разбор аргументов, чтение сцены, построение кадра и вывод результата.

Коды возврата: 0 — успех, 1 — ошибка сцены или ввода-вывода (в stderr строка Error
и причина), 2 — неверные аргументы (в stderr справка).
"""

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures.process import BrokenProcessPool
from typing import NoReturn

from minirt.errors import RtError
from minirt.image_io import save_png
from minirt.parser import parse_scene_file
from minirt.render import RenderOptions, render
from minirt.window import show_image

MAX_SIZE = 8192
MAX_SAMPLES = 16
MAX_WORKERS = 64


class _ArgumentParser(argparse.ArgumentParser):
    """argparse, который при ошибке печатает полную справку и завершается с кодом 2."""

    def error(self, message: str) -> NoReturn:
        self.print_help(sys.stderr)
        self.exit(2, f"\nошибка в аргументах: {message}\n")


def _int_range(low: int, high: int) -> Callable[[str], int]:
    """Преобразователь аргумента в целое из диапазона [low, high] для argparse."""

    def convert(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"ожидается целое число, получено '{text}'") from None
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"допустимо {low}..{high}, получено {value}")
        return value

    return convert


def build_parser() -> argparse.ArgumentParser:
    """Описание параметров командной строки."""
    parser = _ArgumentParser(
        prog="python main.py",
        description="miniRT — трассировщик лучей: строит изображение сцены из файла .rt.",
    )
    parser.add_argument("scene", help="файл сцены .rt")
    parser.add_argument(
        "--width", type=_int_range(1, MAX_SIZE), default=800, help="ширина в пикселях (800)"
    )
    parser.add_argument(
        "--height", type=_int_range(1, MAX_SIZE), default=600, help="высота в пикселях (600)"
    )
    parser.add_argument("--out", metavar="ФАЙЛ.png", help="сохранить изображение в PNG")
    parser.add_argument("--no-window", action="store_true", help="не открывать окно")
    parser.add_argument("--specular", action="store_true", help="блики Фонга")
    parser.add_argument(
        "--samples",
        type=_int_range(1, MAX_SAMPLES),
        default=1,
        metavar="N",
        help="сглаживание: N² лучей на пиксель",
    )
    parser.add_argument(
        "--workers",
        type=_int_range(0, MAX_WORKERS),
        default=1,
        metavar="N",
        help="число процессов; 0 — по числу ядер (1)",
    )
    return parser


def _fail(reason: str) -> int:
    print("Error", file=sys.stderr)
    print(reason, file=sys.stderr)
    return 1


def _run(args: argparse.Namespace) -> int:
    try:
        scene = parse_scene_file(args.scene)
    except OSError as exc:
        return _fail(f"не удалось прочитать файл сцены '{args.scene}': {exc.strerror or exc}")
    except RtError as exc:
        return _fail(str(exc))

    start = time.perf_counter()
    try:
        workers = args.workers or os.cpu_count() or 1
        options = RenderOptions(specular=args.specular, samples=args.samples, workers=workers)
        image = render(scene, args.width, args.height, options)
    except MemoryError:
        return _fail(f"недостаточно памяти для кадра {args.width}x{args.height}")
    except BrokenProcessPool:
        return _fail("процесс-обработчик полосы кадра аварийно завершился")
    elapsed = time.perf_counter() - start
    print(
        f"{args.scene}: {args.width}x{args.height}, построено за {elapsed:.2f} с", file=sys.stderr
    )

    if args.out is not None:
        try:
            save_png(args.out, image)
        except OSError as exc:
            return _fail(f"не удалось записать '{args.out}': {exc.strerror or exc}")
    if not args.no_window:
        title = f"miniRT — {args.scene} · {args.width}×{args.height} · {elapsed:.2f} с"
        try:
            show_image(image, title)
        except RtError as exc:
            return _fail(str(exc))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить программу с аргументами argv (по умолчанию — sys.argv[1:])."""
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:  # --help или неверные аргументы
        return exc.code if isinstance(exc.code, int) else 2
    try:
        return _run(args)
    except KeyboardInterrupt:
        return 130
