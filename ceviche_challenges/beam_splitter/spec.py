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

"""Specification for a beam splitter with one design region, in ceviche.

A beam splitter is a four port optical device with the layout sketched below:

                                __________
              port 1  ##########|        |##########  port 2
                      ##########|        |##########
                                | design |
                                | region |
                      ##########|        |##########
     y ^      port 4  ##########|        |##########  port 3
       |                        ----------
       -->
         x

The function of a beam splitter (sometimes also referred to as a directional
coupler) is to divide the input power entering *ANY* port equally between the
two ports on the opposite side of the device. For example, an input signal at
port 1 has its power split equally between port 2 and port 3.

The structural symmetry of the beam splitter plays an important role in
determining the symmetry of the resulting scattering matrix of the device. This
symmetry is important for the beam splitter in the context of inverse design
because if no symmetry of the design is enforced, then a total of four
simulations must be included in the objective function to guarantee a 50:50
splitting ratio for each port. Thus, in our beam splitter model we allow for
control over this symmetry via the `design_symmetry` parameter, which can take
a value of 'x', 'xy', or None. An x symmetry enforces mirror symmetry of the
design about a line along the x-axis bisecting the design. In other words, the
size of the design variable is roughly half of the size of the design region. An
xy symmetry imposes an additional mirror symmetry of the design about a line
along the y-axis bisecting the design. In this case, the size of the design
variable is roughly one fourth of the size of the design region. In our
description above, we use the qualifier "roughly" because the size of one or
both of the design region dimensions could be an odd number of grid cells. In
such cases, our implementation takes care of the appropriate mirroring operation
(which can result in the design variable not being exactly half or one fourth
of the design region).

  Illustration of a design with "x" symmetry:

            design variable
             relative to                              full design
            design region
              __________                               __________
 port 1 ######|        |###### port 2     port 1 ######|fffffggg|###### port 2
        ######|        |######                   ######|dddeeeee|######
              |        |                               |aaabbbcc|
              |aaabbbcc|                               |aaabbbcc|
        ######|dddeeeee|######                   ######|dddeeeee|######
 port 4 ######|fffffggg|###### port 3     port 4 ######|fffffggg|###### port 3
              ----------                               ----------

  Illustration of a design with "xy" symmetry:

            design variable
             relative to                              full design
            design region
              __________                               __________
 port 1 ######|        |###### port 2     port 1 ######|qqvvvvqq|###### port 2
        ######|        |######                   ######|zzwwwwzz|######
              |        |                               |xxyyyyxx|
              |xxyy    |                               |xxyyyyxx|
        ######|zzww    |######                   ######|zzwwwwzz|######
 port 4 ######|qqvv    |###### port 3     port 4 ######|qqvvvvqq|###### port 3
              ----------                               ----------
"""

import dataclasses
from typing import Optional, Sequence, Tuple

from ceviche_challenges import units as u

Q = u.Quantity  # pylint: disable=invalid-name
QSequence = Sequence[Q]  # pylint: disable=invalid-name


@dataclasses.dataclass(frozen=True)
class BeamSplitterSpec:
  """Parameters specifying the physical structure of the 2D beam splitter.

  Note: the (i, j) coordinate origin is at the SW corner of the structure
    in the design plane; see the module docstring above for a picture. We use a
    convention where `i` corresponds to `x` and `j` corresponds to `y`.

  Attributes:
    wg_width: the width (along j) of the waveguides.
    wg_length: the length (along i) of the waveguides on either side of the
      design region.
    wg_separation: the separation distance (along j) between the waveguides.
    wg_mode_padding: the length we should look around the waveguide in each
      direction in our computation of the modes.
    port_pml_offset: the offset between the ports and the PML
    variable_region_size: a sequence specifying the size of the design variable
      region.
    cladding_permittivity: the relative permittivity of the cladding surrounding
      the slab.  Also used as the permittivity for design pixels valued `0`.
    slab_permittivity: the relative permittivity within the slab, as well as the
      permittivity value for design pixels valued `1`.
    input_monitor_offset: The offset of the input monitor  from the input
      source, along i.
    design_symmetry: A string specifying the symmetry of the design in the
      design region. Options are 'x', 'xy', or None. See the module docstring
      above for a description of this property.
    pml_width: the integer number of PML cells we should use within the volume
      at the ends of each axis.
  """
  wg_width: Q
  wg_length: Q
  wg_separation: Q
  wg_mode_padding: Q
  port_pml_offset: Q
  variable_region_size: QSequence
  cladding_permittivity: float
  slab_permittivity: float
  input_monitor_offset: Q
  design_symmetry: Optional[str]
  pml_width: int

  def __post_init__(self):
    assert self.wg_separation >= 2 * self.wg_mode_padding
    assert self.variable_region_size[1] >= 2 * self.wg_width + self.wg_separation

  def extent_ij(self, resolution: Q) -> Tuple[Q, Q]:
    """The total in-plane extent of the structure.

    Args:
      resolution: The resolution of the simulation.

    Returns:
      The total in-plane extent of the structure, as a tuple (i, j) of
        positions.
    """
    vi, vj = self.variable_region_size
    extent_i = 2 * self.pml_width * resolution + 2 * self.wg_length + vi
    extent_j = 2 * self.pml_width * resolution + 2 * self.wg_mode_padding + vj
    return (extent_i, extent_j)
