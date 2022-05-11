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

"""Top-level interface for scattering parameter calculation in ceviche.

NOTE: these are currently only implemented for the TM (Ez) polarization and NOT
the TE (Hz) polarization
"""

# pytype: disable=module-attr

from typing import Tuple

import autograd.numpy as npa
from ceviche_challenges import defs
from ceviche_challenges import modes
from ceviche_challenges import ops
import numpy as np


def calculate_amplitudes(
    omega: float,
    dl: float,
    port: modes.Port,
    ez: defs.Field,
    hy: defs.Field,
    hx: defs.Field,
    epsilon_r: defs.Geometry,
) -> Tuple[float, float]:
  """Calculate amplitudes of the forward and backward waves at a port.

  Args:
    omega: `float` specifying the angular frequency of the mode, in units of
      rad/sec.
    dl: `float` specifying the spatial grid cell size, in units of meters.
    port: `Port` specifying details of the mode calculation, e.g. order,
      location, direction, etc.
    ez: `Field` specifying the distribution of the z-component of the electric
      field
    hy: `Field` specifying the distribution of the y-component of the magnetic
      field.
    hx: `Field` specifying the distribution of the x-component of the magnetic
      field.
    epsilon_r: `Geometry` specifying the permitivitty distribution.

  Returns:
    A tuple consisting of the complex-valued scattering parameters, (s_+, s_-)
    at the port.
  """
  coords = port.coords()
  et_m, ht_m, _ = port.field_profiles(epsilon_r[coords], omega, dl)

  if port.dir.is_along_x:
    coords_offset = (coords[0] + port.signed_offset(), coords[1])
    h = (0., npa.ravel(hy[coords_offset]), 0)
    hm = (0., ht_m, 0.)
    # The E-field is not co-located with the H-field in the Yee cell. Therefore,
    # we must sample at two neighboring pixels in the propataion direction and
    # then interpolate:
    coords_e = (coords_offset[0] + np.array([[-1], [0]]), coords_offset[1])
    e_yee_shifted = 0.5 * npa.sum(ez[coords_e], axis=0)
  else:
    coords_offset = (coords[0], coords[1] + port.signed_offset())
    h = (npa.ravel(hx[coords_offset]), 0, 0)
    hm = (-ht_m, 0., 0.)
    # The E-field is not co-located with the H-field in the Yee cell. Therefore,
    # we must sample at two neighboring pixels in the propataion direction and
    # then interpolate:
    coords_e = (coords_offset[0], coords_offset[1] + np.array([[-1], [0]]))
    e_yee_shifted = 0.5 * npa.sum(ez[coords_e], axis=0)

  e = (0., 0., e_yee_shifted)
  em = (0., 0., et_m)

  overlap1 = ops.overlap(em, h, port.dir)
  overlap2 = ops.overlap(hm, e, port.dir)
  normalization = ops.overlap(em, hm, port.dir)

  # Phase convention in ceviche is exp(+jwt-jkz)
  if port.dir.sign > 0:
    s_p = (overlap1 + overlap2) / 2 / np.sqrt(2 * normalization)
    s_m = (overlap1 - overlap2) / 2 / np.sqrt(2 * normalization)
  else:
    s_p = (overlap1 - overlap2) / 2 / np.sqrt(2 * normalization)
    s_m = (overlap1 + overlap2) / 2 / np.sqrt(2 * normalization)

  return s_p, s_m
