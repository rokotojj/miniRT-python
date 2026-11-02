"""Тесты векторной математики."""

import numpy as np
import pytest

from minirt.vec import color01, cross, dot, norm, normalize, vec3


def test_dot_single_and_batched() -> None:
    a = vec3((1.0, 2.0, 3.0))
    assert dot(a, vec3((4.0, 5.0, 6.0))) == pytest.approx(32.0)
    batch = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    np.testing.assert_allclose(dot(batch, vec3((3.0, 4.0, 5.0))), [3.0, 8.0])


def test_norm_and_normalize() -> None:
    v = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, -2.0]])
    np.testing.assert_allclose(norm(v), [5.0, 2.0])
    np.testing.assert_allclose(normalize(v), [[0.6, 0.8, 0.0], [0.0, 0.0, -1.0]])


def test_normalize_keeps_frame_shape() -> None:
    frame = np.ones((4, 5, 3))
    unit = normalize(frame)
    assert unit.shape == (4, 5, 3)
    np.testing.assert_allclose(norm(unit), np.ones((4, 5)))


def test_cross_is_right_handed() -> None:
    np.testing.assert_allclose(cross(vec3((1.0, 0.0, 0.0)), vec3((0.0, 1.0, 0.0))), [0, 0, 1])


def test_color01() -> None:
    np.testing.assert_allclose(color01((255, 0, 51)), [1.0, 0.0, 0.2])
