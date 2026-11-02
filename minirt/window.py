"""Показ готового изображения в окне. Единственный модуль, который импортирует pygame.

Esc и закрытие окна завершают показ. Если изображение больше экрана, в окне
показывается уменьшенная копия; файл PNG сохраняется в исходном размере.
"""

import os

import numpy as np
import numpy.typing as npt

from minirt.errors import WindowError

_SCREEN_SHARE = 0.9  # окно занимает не больше этой доли экрана


def show_image(image: npt.NDArray[np.uint8], title: str) -> None:
    """Открыть окно с изображением (H, W, 3) и ждать Esc или закрытия окна."""
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        import pygame
    except ImportError as exc:
        raise WindowError(
            "не установлен pygame (pip install pygame); запустите с --no-window --out файл.png"
        ) from exc

    try:
        pygame.display.init()
        height, width, _ = image.shape
        surface = pygame.surfarray.make_surface(np.ascontiguousarray(image.swapaxes(0, 1)))
        info = pygame.display.Info()
        scale = 1.0
        if info.current_w > 0 and info.current_h > 0:
            scale = min(
                1.0,
                _SCREEN_SHARE * info.current_w / width,
                _SCREEN_SHARE * info.current_h / height,
            )
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        if size != (width, height):
            surface = pygame.transform.smoothscale(surface, size)
        screen = pygame.display.set_mode(size)
        pygame.display.set_caption(title)
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    return
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(30)
    except pygame.error as exc:
        raise WindowError(f"не удалось открыть окно: {exc}") from exc
    finally:
        pygame.quit()
