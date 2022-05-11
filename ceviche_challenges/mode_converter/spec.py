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

"""Specification for a waveguide mode converter, in ceviche.

A sketch of the inverse designed waveguide mode converter is shown below:

                                __________
                                |        |
              port 1  ##########| design |##########  port 2
                      ##########| region |##########
                                |        |
     y ^                        ----------
       |
       -->
         x

The goal of the design in this model is to convert all input signal energy in
a specified transverse mode entering in port 1 to a different transverse mode at
port 2. For example, the device could convert the fundamental waveguide mode to
the second-order waveguide mode. In the specification defined below, a
parameter is provided for selecting the order of the transverse mode.
"""

import dataclasses
from typing import Sequence, Tuple

from ceviche_challenges import units as u

import numpy as np

Q = u.Quantity  # pylint: disable=invalid-name
QSequence = Sequence[Q]  # pylint: disable=invalid-name


@dataclasses.dataclass(frozen=True)
class ModeConverterSpec:
  """Parameters specifying the physical structure of the mode converter.

  Attributes:
    left_wg_width: the width of the waveguide on the left side of the design
      region.
    left_wg_mode_padding: the length we should look around the left waveguide in
      each direction in our computation of the modes.
    left_wg_mode_order: the transverse order of the mode on in the waveguide on
      the left side of the design region.
    right_wg_width: the width of the waveguide on the right side of the design
      region.
    right_wg_mode_padding: the length we should look around the right waveguide
      in each direction in our computation of the modes.
    right_wg_mode_order: the transverse order of the mode on in the waveguide on
      the right side of the design region.
    wg_length: the length of the waveguides entering on the left and right side
      of the design region.
    padding: the padding distance between the top and bottom of the design
      region and the PML.
    port_pml_offset: the offset between the ports and the PML
    variable_region_size: a sequence specifying the size of the design variable
      region.
    cladding_permittivity: the relative permittivity of the cladding surrounding
      the slab.  Also used as the permittivity for design pixels valued `0`.
    slab_permittivity: the relative permittivity within the slab, as well as the
      permittivity value for design pixels valued `1`.
    input_monitor_offset: The offset of the input monitor  from the input
      source, along i.
    pml_width: the integer number of PML cells we should use within the volume
      at the ends of each axis.
  """
  left_wg_width: Q
  left_wg_mode_padding: Q
  left_wg_mode_order: int
  right_wg_width: Q
  right_wg_mode_padding: Q
  right_wg_mode_order: int
  wg_length: Q
  padding: Q
  port_pml_offset: Q
  variable_region_size: QSequence
  cladding_permittivity: float
  slab_permittivity: float
  input_monitor_offset: Q
  pml_width: int

  def __post_init__(self):
    vi, _ = self.variable_region_size
    assert self.left_wg_width <= vi
    assert self.right_wg_width <= vi

  def extent_ij(self, resolution: Q) -> Tuple[Q, Q]:
    """The total in-plane extent of the structure.

    Args:
      resolution: The resolution of the simulation.

    Returns:
      The total in-plane extent of the structure, as a tuple (i, j) of
        positions.
    """
    vi, vj = self.variable_region_size
    pml_thickness = self.pml_width * resolution
    left_port_height = self.left_wg_width + 2 * self.left_wg_mode_padding
    right_port_height = self.right_wg_width + 2 * self.right_wg_mode_padding
    largest_port_height = np.maximum(left_port_height, right_port_height)
    padded_design_height = vj + 2 * self.padding
    extent_i = 2 * pml_thickness + 2 * self.wg_length + vi
    extent_j = 2 * pml_thickness + np.maximum(
        padded_design_height,
        largest_port_height,
    )
    return (extent_i, extent_j)
