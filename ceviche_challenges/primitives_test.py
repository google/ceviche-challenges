# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for ceviche_challenges.primitives."""

from typing import Callable
from absl.testing import absltest
import autograd
import autograd.numpy as npa
from ceviche_challenges import primitives
import numpy as np


def _gradient_numerical(func: Callable[[np.ndarray], float],
                        x: np.ndarray,
                        delta: float = 1e-7) -> np.ndarray:
  """Helper for numerically computing the gradient of an R^n -> R^1 function."""
  y = func(x)
  grad = np.zeros_like(x)
  for i in range(x.size):
    x_delta = np.copy(x)
    x_delta.ravel()[i] += delta
    y_delta = func(x_delta)
    grad.ravel()[i] = (y_delta - y) / delta
  return grad


class PrimitivesTest(absltest.TestCase):

  def test_grad_insert_design_variable(self):
    """Test the gradient of the `insert_design_variable` primitive."""

    destination = np.ones((10, 10), dtype=np.float64)
    design_var = np.ones((5, 5), dtype=np.float64)
    coords = (3, 4, 8, 9)

    def func(x):
      y0 = npa.square(x) - 1.0
      y1 = primitives.insert_design_variable(y0, destination, coords)
      return npa.sum(y1)

    self.assertTrue(
        np.allclose(
            autograd.grad(func)(design_var),
            _gradient_numerical(func, design_var)))

  def test_insert_design_variable_invalid(self):
    """Test that a ValueError is thrown for invalid coords or design vars."""
    with self.assertRaises(ValueError):
      destination = np.ones((10, 10), dtype=np.float64)
      design_var = np.ones((5, 5), dtype=np.float64)
      # Negative coordinates
      coords = (-1, 4, 4, 9)
      primitives.insert_design_variable(design_var, destination, coords)
    with self.assertRaises(ValueError):
      destination = np.ones((10, 10), dtype=np.float64)
      design_var = np.ones((5, 5), dtype=np.float64)
      # Incorrect ordering of coordinates
      coords = (8, 4, 3, 9)
      primitives.insert_design_variable(design_var, destination, coords)
    with self.assertRaises(ValueError):
      destination = np.ones((10, 10), dtype=np.float64)
      design_var = np.ones((5, 5), dtype=np.float64)
      # Extending outside of the destination
      coords = (8, 4, 13, 9)
      primitives.insert_design_variable(design_var, destination, coords)
    with self.assertRaises(ValueError):
      destination = np.ones((10, 10), dtype=np.float64)
      design_var = np.ones((15, 5), dtype=np.float64)
      # Too large of a design variable
      coords = (0, 4, 14, 9)
      primitives.insert_design_variable(design_var, destination, coords)


if __name__ == '__main__':
  absltest.main()
