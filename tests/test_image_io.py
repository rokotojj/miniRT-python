"""Тесты записи PNG."""

import struct
import zlib
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from minirt.image_io import PNG_SIGNATURE, decode_png, encode_png, save_png


def sample_image() -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(7, 5, 3), dtype=np.uint8)


def test_png_header_and_chunks() -> None:
    data = encode_png(sample_image())
    assert data.startswith(PNG_SIGNATURE)
    assert data[12:16] == b"IHDR"
    width, height, depth, color_type = struct.unpack(">IIBB", data[16:26])
    assert (width, height, depth, color_type) == (5, 7, 8, 2)
    assert data.endswith(b"IEND\xaeB`\x82")


def test_chunk_crc_is_valid() -> None:
    data = encode_png(sample_image())
    length = struct.unpack(">I", data[8:12])[0]
    crc = struct.unpack(">I", data[16 + length : 20 + length])[0]
    assert crc == zlib.crc32(data[12 : 16 + length])


def test_round_trip(tmp_path: Path) -> None:
    image = sample_image()
    path = tmp_path / "out.png"
    save_png(path, image)
    np.testing.assert_array_equal(decode_png(path.read_bytes()), image)


def test_wrong_shape_rejected() -> None:
    with pytest.raises(ValueError, match="uint8"):
        encode_png(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        encode_png(np.zeros((4, 4, 3), dtype=np.float64))
