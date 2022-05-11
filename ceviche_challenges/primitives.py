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
"""Definitions for autograd primitives."""

from typing import Tuple

import autograd
import numpy as np


@autograd.primitive
def insert_design_variable(design_var: np.ndarray, destination: np.ndarray,
                           coords: Tuple[int, int, int, int]) -> np.ndarray:
  """Insert 2D design variable a into a larger 2D ndarray at coordinates.

  NOTE: This function only connects gradients flowing through `design_var` and
  not `destination` or `coords`.

  Args:
    design_var: An `np.ndarray` specifying the design variable.
    destination: An `np.ndarray` specifying the destination array for
      `design_var` to be inserted.
    coords: A `Tuple` specifying the coordinates to insert design_var in the
      format `(x_min, y_min, x_max, y_max)`.

  Returns:
    An `np.ndarray` with the same shape as `destination` with the contents of
      `design_var` inserted at the location specified by `coords`.
  """
  (x_min, y_min, x_max, y_max) = coords
  if (design_var.shape[0] > destination.shape[0] or
      design_var.shape[1] > destination.shape[1]):
    raise ValueError(
        'The `design_var` array with shape {} does not fit into the '
        '`destination` array with shape {}'.format(design_var.shape,
                                                   destination.shape))
  if not np.all([coord > 0 for coord in coords]):
    raise ValueError('All values in `coord` must be positive')
  if x_min >= x_max:
    raise ValueError('The min x value must be less than the max x value')
  if y_min >= y_max:
    raise ValueError('The min y value must be less than the max y value')
  if (x_max >= destination.shape[0] or y_max >= destination.shape[1]):
    raise ValueError(
        'Box defined by `coords` extends outside of `destination` array')
  destination_ = np.copy(destination)
  destination_[coords[0]:coords[2], coords[1]:coords[3]] = design_var
  return destination_


def vjp_maker(ans, design_var, destination, coords):
  del ans, design_var, destination
  return lambda x: x[coords[0]:coords[2], coords[1]:coords[3]]


autograd.extend.defvjp(insert_design_variable, vjp_maker, None, None)
