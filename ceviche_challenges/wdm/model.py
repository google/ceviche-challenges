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
"""A planar WDM device with one design region, in ceviche."""

from typing import Tuple, List

from ceviche_challenges import defs
from ceviche_challenges import model_base
from ceviche_challenges import modes
from ceviche_challenges import params as _params
from ceviche_challenges import units as u
from ceviche_challenges.wdm import spec as _spec

import numpy as np


class WdmModel(model_base.Model):
  """A planar WDM device with one design region, in ceviche."""

  def __init__(self, params: _params.CevicheSimParams, spec: _spec.WdmSpec):
    """Initializes a new model using `SimParams` and `WdmSpec`.

    Args:
      params: `SimParams` specifying the simulation parameters.
      spec: `WdmSpec` specifying the geometry of the WDM.
    """
    super().__init__()
    self.params = params
    self.spec = spec
    extent_i, extent_j = spec.extent_ij
    self._shape = (
        u.resolve(extent_i, params.resolution),
        u.resolve(extent_j, params.resolution),
    )
    self._make_bg_density_and_ports()

  def _make_bg_density_and_ports(self, init_design_region: bool = True):
    """Initializes background density and ports for the model.


    Args:
      init_design_region: `bool` specifying whether the pixels in the background
        density distribution that lie within the design region should be
        initialized to a non-zero value. If `True`, the pixels are initialized
        to a value of `0.5`.
    Side effects: Initializes `_density_bg`, an `np.ndarray` specifying the
      background material density distribution of the WDM. Initalizes `ports`, a
      `List[Port]` that specifies the ports of the WDM.
    """
    p = self.params
    s = self.spec

    density = np.zeros(self.shape)

    monitor_offset = 5
    ports = []

    design_region_x0, _, design_region_x1, _ = self.design_region_coords

    # Input waveguide
    y1 = u.resolve(s.input_wg_j - s.wg_width / 2, p.resolution)
    y2 = u.resolve(s.input_wg_j + s.wg_width / 2, p.resolution)
    density[:design_region_x0, y1:y2] = 1.0

    # Input port
    ports.append(
        modes.WaveguidePort(
            x=u.resolve(s.input_mode_i, p.resolution),
            y=u.resolve(s.input_wg_j, p.resolution),
            width=u.resolve(s.wg_width + 2 * s.wg_mode_padding, p.resolution),
            order=1,
            dir=defs.Direction.X_POS,
            offset=monitor_offset))

    # Output waveguides
    for output_wg_j in s.output_wgs_j:
      y1 = u.resolve(output_wg_j - s.wg_width / 2, p.resolution)
      y2 = u.resolve(output_wg_j + s.wg_width / 2, p.resolution)
      density[design_region_x1:, y1:y2] = 1.0
      ports.append(
          modes.WaveguidePort(
              x=u.resolve(s.output_mode_i, p.resolution),
              y=u.resolve(output_wg_j, p.resolution),
              width=u.resolve(s.wg_width + 2 * s.wg_mode_padding, p.resolution),
              order=1,
              dir=defs.Direction.X_NEG,
              offset=monitor_offset))

    if init_design_region:
      density = density + self.design_region.astype(np.float64)

    self._density_bg = density
    self._ports = ports

  @property
  def design_region_coords(self) -> Tuple[int, int, int, int]:
    """The coordinates of the design region as (x_min, y_min, x_max, y_max)."""
    s = self.spec
    p = self.params
    x_min, y_min = [u.resolve(v, p.resolution) for v in s.variable_region[0]]
    x_max, y_max = [u.resolve(v,p.resolution) for v in s.variable_region[1]]
    return (x_min, y_min, x_max, y_max)

  @property
  def shape(self) -> Tuple[int, int]:
    """Shape of the simulation domain, in grid units."""
    return self._shape

  @property
  def density_bg(self) -> np.ndarray:
    """The background density distribution of the model."""
    return self._density_bg

  @property
  def slab_permittivity(self) -> float:
    """The slab permittivity of the model."""
    s = self.spec
    return s.slab_permittivity

  @property
  def cladding_permittivity(self) -> float:
    """The cladding permittivity of the model."""
    s = self.spec
    return s.cladding_permittivity

  @property
  def dl(self) -> float:
    """The grid resolution of the model."""
    p = self.params
    return p.resolution.to_value('m')

  @property
  def pml_width(self) -> int:
    """The width of the PML region, in grid units."""
    s = self.spec
    return s.pml_width

  @property
  def ports(self) -> List[modes.Port]:
    """A list of the device ports."""
    return self._ports

  @property
  def output_wavelengths(self) -> List[float]:
    """A list of the wavelengths, in nm, to output fields and s-parameters."""
    return u.Array(self.params.wavelengths).to_value(u.nm)
