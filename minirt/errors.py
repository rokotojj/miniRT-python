"""Исключения трассировщика.

Все ошибки, о которых программа сообщает пользователю, — наследники RtError.
Точка входа перехватывает RtError и OSError, печатает «Error» и причину и
возвращает код 1.
"""


class RtError(Exception):
    """Базовая ошибка miniRT."""


class SceneFormatError(RtError):
    """Нарушение формата файла сцены .rt.

    reason — причина без номера строки, line — номер строки (с 1) или None,
    если ошибка относится к сцене целиком (например, нет камеры).
    """

    def __init__(self, reason: str, line: int | None = None) -> None:
        self.reason = reason
        self.line = line
        message = reason if line is None else f"строка {line}: {reason}"
        super().__init__(message)


class WindowError(RtError):
    """Не удалось открыть окно: нет библиотеки окна или дисплея."""
