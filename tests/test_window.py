"""Дымовой тест окна на виртуальном дисплее SDL: окно открывается и закрывается по Esc."""

import numpy as np
import pytest

from minirt.window import show_image

pygame = pytest.importorskip("pygame")


def test_window_closes_on_escape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    escape = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)
    monkeypatch.setattr(pygame.event, "get", lambda: [escape])
    show_image(np.zeros((6, 8, 3), dtype=np.uint8), "test")
    assert not pygame.display.get_init()
