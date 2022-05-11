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

"""Specification for a waveguide bend, in ceviche.

A sketch of the inverse designed waveguide bend is shown below:

                                __________
                                |        |
              port 1  ##########| design |
                      ##########| region |
                                |        |
     y ^                        ----------
       |                           ###
       -->                         ###
         x                         ###
                                   ###
                                   ###  port 2

The goal of the design in this model is to route all input signal energy from
port 1 to port 2, and vice-versa.
"""

import dataclasses
from typing import Sequence, Tuple

from ceviche_challenges import units as u

Q = u.Quantity  # pylint: disable=invalid-name
QSequence = Sequence[Q]  # pylint: disable=invalid-name


@dataclasses.dataclass(frozen=True)
class WaveguideBendSpec:
  """Parameters specifying the physical structure of the waveguide bend.

  Attributes:
    wg_width: the width of the waveguides.
    wg_length: the length of the waveguides entering on the left and bottom side
      of the design region.
    wg_mode_padding: the length we should look around the waveguide in each
      direction in our computation of the modes.
    padding: the padding distance between the top and right side of the design
      region and the PML
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
  wg_width: Q
  wg_length: Q
  wg_mode_padding: Q
  padding: Q
  port_pml_offset: Q
  variable_region_size: QSequence
  cladding_permittivity: float
  slab_permittivity: float
  input_monitor_offset: Q
  pml_width: int

  def __post_init__(self):
    vi, vj = self.variable_region_size
    assert self.wg_width <= vi
    assert self.wg_width <= vj
    assert (self.wg_mode_padding + self.wg_width / 2) <= (
        vi / 2 + self.wg_length)
    assert (self.wg_mode_padding + self.wg_width / 2) <= (
        vj / 2 + self.wg_length)

  def extent_ij(self, resolution: Q) -> Tuple[Q, Q]:
    """The total in-plane extent of the structure.

    Args:
      resolution: The resolution of the simulation.

    Returns:
      The total in-plane extent of the structure, as a tuple (i, j) of
        positions.
    """
    vi, vj = self.variable_region_size
    pml_thickness = 2 * self.pml_width * resolution
    extent_i = pml_thickness + self.wg_length + vi + self.padding
    extent_j = pml_thickness + self.wg_length + vj + self.padding
    return (extent_i, extent_j)
