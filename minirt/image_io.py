"""Сохранение изображения в PNG без сторонних библиотек: zlib + struct.

Формат: сигнатура, блок IHDR (размеры, 8 бит на канал, RGB), блок IDAT со сжатыми
строками (перед каждой — байт фильтра 0), блок IEND. У каждого блока — CRC32.
"""

import os
import struct
import zlib

import numpy as np
import numpy.typing as npt

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def encode_png(image: npt.NDArray[np.uint8]) -> bytes:
    """Закодировать изображение (H, W, 3) uint8 в байты PNG."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"ожидается массив (H, W, 3) uint8, получено {image.shape} {image.dtype}")
    height, width, _ = image.shape
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = np.zeros((height, width * 3 + 1), dtype=np.uint8)
    rows[:, 1:] = image.reshape(height, width * 3)
    data = zlib.compress(rows.tobytes(), 6)
    return PNG_SIGNATURE + _chunk(b"IHDR", header) + _chunk(b"IDAT", data) + _chunk(b"IEND", b"")


def decode_png(data: bytes) -> npt.NDArray[np.uint8]:
    """Прочитать PNG, записанный encode_png (RGB, 8 бит, фильтр 0) — для тестов и сравнения."""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("не PNG: неверная сигнатура")
    position = len(PNG_SIGNATURE)
    width = height = 0
    compressed = b""
    while position < len(data):
        (length,) = struct.unpack(">I", data[position : position + 4])
        kind = data[position + 4 : position + 8]
        body = data[position + 8 : position + 8 + length]
        position += 12 + length
        if kind == b"IHDR":
            width, height, depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", body)
            if (depth, color_type) != (8, 2):
                raise ValueError("поддерживается только RGB с 8 битами на канал")
        elif kind == b"IDAT":
            compressed += body
    raw = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8)
    rows = raw.reshape(height, width * 3 + 1)
    if rows[:, 0].any():
        raise ValueError("поддерживается только фильтр строк 0")
    return rows[:, 1:].reshape(height, width, 3).copy()


def save_png(path: str | os.PathLike[str], image: npt.NDArray[np.uint8]) -> None:
    """Записать изображение в файл PNG."""
    with open(path, "wb") as file:
        file.write(encode_png(image))
