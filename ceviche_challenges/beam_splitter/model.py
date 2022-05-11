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
"""A model of a waveguide beam splitter with one design region, in ceviche."""

from typing import Tuple, List

from ceviche_challenges import units as u
from ceviche_challenges import defs
from ceviche_challenges import model_base
from ceviche_challenges import modes
from ceviche_challenges import params as _params
from ceviche_challenges.beam_splitter import spec as _spec

# Note: autograd.numpy does not play nicely with pytype, which is the reason for
# disabling module-attr below.
# pytype: disable=module-attr
import autograd.numpy as npa
import numpy as np


class BeamSplitterModel(model_base.Model):
  """A planar beam splitter with one design region, in ceviche."""

  def __init__(
      self,
      params: _params.CevicheSimParams,
      spec: _spec.BeamSplitterSpec,
  ):
    """Initializes a new beam splitter model.

    See the module docstring in spec.py for more details on the specification
    of the beam splitter model.

    Args:
      params: Parameters for the ceviche simulation.
      spec: Specification of the beam splitter geometry.
    """
    self.params = params
    self.spec = spec
    extent_i, extent_j = spec.extent_ij(params.resolution)
    self._shape = (
        u.resolve(extent_i, params.resolution),
        u.resolve(extent_j, params.resolution),
    )
    self._make_bg_density_and_ports()

  def _make_bg_density_and_ports(self, init_design_region: bool = False):
    """Initializes background density and ports for the model.

    Args:
      init_design_region: `bool` specifying whether the pixels in the background
        density distribution that lie within the design region should be
        initialized to a non-zero value. If `True`, the pixels are initialized
        to a value of `1.0`.
    Side effects: Initializes `_density_bg`, an `np.ndarray` specifying the
      background material density distribution of the WDM. Initalizes `ports`, a
      `List[Port]` that specifies the ports of the WDM.
    """
    p = self.params
    s = self.spec

    density = np.zeros(self.shape)

    monitor_offset = u.resolve(s.input_monitor_offset, p.resolution)

    wgs_j = [
        s.extent_ij(p.resolution)[1] / 2 - s.wg_separation / 2 - s.wg_width / 2,
        s.extent_ij(p.resolution)[1] / 2 + s.wg_separation / 2 + s.wg_width / 2,
    ]

    for wg_j in wgs_j:
      y1 = u.resolve(wg_j - s.wg_width / 2, p.resolution)
      y2 = u.resolve(wg_j + s.wg_width / 2, p.resolution)
      density[:, y1:y2] = 1.0

    port_i1 = s.pml_width + u.resolve(s.port_pml_offset, p.resolution)
    port_i2 = u.resolve(
        s.extent_ij(p.resolution)[0] - s.port_pml_offset,
        p.resolution) - s.pml_width
    port_js = [
        u.resolve(j, p.resolution)
        for j in [wgs_j[1], wgs_j[1], wgs_j[0], wgs_j[0]]
    ]
    port_is = [port_i1, port_i2, port_i2, port_i1]
    port_dirs = [
        defs.Direction.X_POS,
        defs.Direction.X_NEG,
        defs.Direction.X_NEG,
        defs.Direction.X_POS,
    ]

    ports = []
    for port_i, port_j, port_dir in zip(port_is, port_js, port_dirs):
      ports.append(
          modes.WaveguidePort(
              x=port_i,
              y=port_j,
              width=u.resolve(s.wg_width + 2 * s.wg_mode_padding, p.resolution),
              order=1,
              dir=port_dir,
              offset=monitor_offset))

    if init_design_region:
      density[self.design_region] = 1.0

    self._density_bg = density
    self._ports = ports

  def transform_design_variable(
      self,
      design_variable: np.ndarray,
  ) -> np.ndarray:
    """Transformation of design variable before inlaying to the design region.

    The transformation applied here depends on the symmetry of the design
    defined in the specification. If no symmetry is specified, then this
    transformation applies an identity operation. If x symmetry is specified
    then the transformation consists of a single mirroring operation about the
    x-axis. If xy symmetry is specified then the transformation consists of two
    mirroring operations: one about the x-axis and one about the y-axis. For a
    more detailed description of these symmetries see the module docstring in
    params.py.

    Args:
      design_variable: An `np.ndarray` of shape `self.design_variable_shape`
        that corresponds to the design variable.

    Returns:
      An `np.ndarray` of shape `self.design_region_shape` that can be inlaid to
        the design region.
    """
    s = self.spec
    if s.design_symmetry == 'x':
      if self.design_region_shape[1] % 2:
        reflected = design_variable[:, :-1]
      else:
        reflected = design_variable
      transformed_design_variable = npa.hstack((
          design_variable,
          npa.fliplr(reflected),
      ))
    elif s.design_symmetry == 'xy':
      if self.design_region_shape[1] % 2:
        reflected = design_variable[:, :-1]
      else:
        reflected = design_variable
      transformed_design_variable = npa.hstack((
          design_variable,
          npa.fliplr(reflected),
      ))
      if self.design_region_shape[0] % 2:
        reflected = transformed_design_variable[:-1,]
      else:
        reflected = transformed_design_variable
      transformed_design_variable = npa.vstack((
          transformed_design_variable,
          npa.flipud(reflected),
      ))
    else:
      transformed_design_variable = design_variable
    return transformed_design_variable

  @property
  def design_variable_shape(self) -> Tuple[int, int]:
    """Shape of the design variable, in grid units."""
    s = self.spec
    if s.design_symmetry == 'x':
      i = self.design_region_shape[0]
      j = np.ceil(self.design_region_shape[1] / 2)
    elif s.design_symmetry == 'xy':
      i = np.ceil(self.design_region_shape[0] / 2)
      j = np.ceil(self.design_region_shape[1] / 2)
    else:
      i, j = self.design_region_shape
    return (int(i), int(j))

  @property
  def design_region_coords(self) -> Tuple[int, int, int, int]:
    """The coordinates of the design region as (x_min, y_min, x_max, y_max)."""
    s = self.spec
    p = self.params
    x_min = s.pml_width + u.resolve(s.wg_length, p.resolution)
    x_max = x_min + u.resolve(s.variable_region_size[0], p.resolution)
    y_min = s.pml_width + u.resolve(s.wg_mode_padding, p.resolution)
    y_max = y_min + u.resolve(s.variable_region_size[1], p.resolution)
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
