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
"""Specification of a WDM structure, using natural units."""

import dataclasses
from typing import Tuple
from ceviche_challenges import units as u

Q = u.Quantity  # pylint: disable=invalid-name
QSequence = u.Sequence  # pylint: disable=invalid-name


@dataclasses.dataclass(frozen=True)
class WdmSpec:
  """Parameters specifying the physical structure of a WDM.

  Attributes:
    extent_ij: the total in-plane extent of the structure, as a tuple (i, j) of
      positions.
    input_wg_j: the center of the input waveguide, along j.
    output_wgs_j: the centers of the output waveguides, along j.
    wg_width: the width (along j) of the waveguides.
    wg_mode_padding: the length we should look around the waveguide in each
      direction in our computation of the modes.
    input_mode_i: location to place the source mode plane (along i).
    output_mode_i: location along i to place the output mode plane.
    variable_region: a box defining the region to be used for the design
      variable.
    cladding_permittivity: the relative permittivity of the cladding surrounding
      the slab.  Also used as the permittivity for design pixels valued `0`.
    slab_permittivity: the relative permittivity within the slab, as well as the
      permittivity value for design pixels valued `1`.
    input_monitor_offset: The offset of the input monitor (if requested via
      `sim.SimParams.monitor_input_port`) from the input source, along i.
    pml_width: the integer number of PML cells we should use within the volume
      at the ends of each axis.
  """
  extent_ij: QSequence
  input_wg_j: Q
  output_wgs_j: QSequence
  wg_width: Q
  wg_mode_padding: Q
  input_mode_i: Q
  output_mode_i: Q
  variable_region: Tuple[QSequence, QSequence]
  cladding_permittivity: float
  slab_permittivity: float
  input_monitor_offset: Q
  pml_width: int
