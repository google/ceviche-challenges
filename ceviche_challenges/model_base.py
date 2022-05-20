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
"""The base model for planar devices in ceviche with a single design region."""

import abc
import concurrent.futures
import itertools
from typing import Tuple, Optional, List, Sequence

import autograd.numpy as npa
import ceviche

from ceviche_challenges import modes
from ceviche_challenges import primitives
from ceviche_challenges import units as u
from ceviche_challenges.scattering import calculate_amplitudes

import numpy as np


def _wavelengths_nm_to_omegas(wavelengths_nm: np.ndarray) -> np.ndarray:
  """Convert array of wavelengths to array of angular frequencies for ceviche."""
  return 2 * np.pi * u.c.to_value('nm/s') / wavelengths_nm


class Model(abc.ABC):
  """The base class for planar devices in ceviche with a single design region.

  This class uses a background material density distribution to define features
  such as input and output waveguides, and the design variable is provided
  upon simulating the model. It is up to child classes to define the simulation
  routine.

  No up- or down-sampling of the design variable is performed while inlaying the
  design variable. Instead, the design variable shape must match that of the
  design variable size specified by the model. Often the size of the design
  variable will be identical to that of the design region, however this isn't a
  requirement, e.g. if a model wishes to impose some symmetry. A rough
  illustration of the inlaying for a planar WDM device is shown below:

                                ___________________
                                |                 |
                                |            #####|
               background       |#####            |
                 density        |            #####|
                                |_________________|
                                ___________________
                                |                 |
                                |     *******     |
                 design         |     *******     |
                 region         |     *******     |
                                |_________________|
                                     _________
                 design              |8877733|
                 variable            |8177733|
                                     |1177733|
                                ___________________
                                |                 |
                                |     8877733#####|
                 simulated      |#####8177733     |
                 density        |     1177733#####|
                                |_________________|

  """

  # Axes for model outputs
  SPARAMS_FREQ_AXIS = -3  # pylint: disable=invalid-name
  SPARAMS_INPUT_PORT_AXIS = -2  # pylint: disable=invalid-name
  SPARAMS_OUTPUT_PORT_AXIS = -1  # pylint: disable=invalid-name
  FIELDS_FREQ_AXIS = -4  # pylint: disable=invalid-name
  FIELDS_INPUT_PORT_AXIS = -3  # pylint: disable=invalid-name
  FIELDS_OUTPUT_PORT_AXIS = -2  # pylint: disable=invalid-name

  def __init__(self):
    """Initializes a new model."""
    pass

  def simulate(
      self,
      design_variable: np.ndarray,
      excite_port_idxs: Sequence[int] = (0,),
      wavelengths_nm: Optional[np.ndarray] = None,
      max_parallelizm: Optional[int] = None,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the device described by the model on a design variable.

    Args:
      design_variable: `np.ndarray` specifying the topology of the design.
      excite_port_idxs: A sorted sequence of ints specifying the ports to
        excite. These values are indexed starting from 0 for port 1. The default
        behavior is to excite only port 1.
      wavelengths_nm: an (optional) ndarray describing the free-space
        wavelengths for which we will perform simulations.  The ndarray must be
        rank-1. If None, the resulting simulation will be as if the user had
        passed `self.output_wavelengths`--a rank-1 array.
      max_parallelizm: Maximum number of parallel threads executing simulations.
        Parallelizm is applied across excitation ports and wavelengths. If None,
        we set the number of workers to number of excitation ports times the
        number of wavelengths.

    Returns:
      s_params: A complex-valued `np.ndarray` of shape `wavelengths_nm.shape +
        (number of excited ports, number of ports,)` corresponding to several
        columns of a complex scattering matrix.
      fields: A complex-valued `np.ndarray` of shape `wavelengths_nm.shape +
        len(excite_port_idxs) + ez.shape`,  containing the steady-state Ez field
        per frequency.
    """
    if np.max(excite_port_idxs) > len(self.ports) - 1:
      raise ValueError('Invalid port index, {}, which exceeds the number of '
                       'ports in the device, {}.'.format(
                           np.max(excite_port_idxs),
                           len(self.ports),
                       ))
    if np.min(excite_port_idxs) < 0:
      raise ValueError('Invalid port index, {}, which below the minimum port '
                       'index of 0.'.format(np.min(excite_port_idxs),))
    if len(np.unique(excite_port_idxs)) != len(excite_port_idxs):
      raise ValueError('Duplicate port index specified in `excite_port_idxs`.')
    if not np.all(np.sort(excite_port_idxs) == np.asarray(excite_port_idxs)):
      raise ValueError('Ports specified in `excite_port_idxs` are not sorted.')

    if wavelengths_nm is None:
      wavelengths_nm = self.output_wavelengths
    else:
      wavelengths_nm = np.asarray(wavelengths_nm)
      if wavelengths_nm.ndim != 1:
        raise ValueError('`wavelengths_nm` arg must be rank-1.')

    omegas = _wavelengths_nm_to_omegas(wavelengths_nm)

    pml_width = self.pml_width
    dl = self.dl
    epsilon_r = self.epsilon_r(design_variable)

    # We use the background epsilon_r for the modal functions because we do not
    # want autograd to attempt to track gradients through the eigensolver. Such
    # tracking would likely fail as the eigensolver is not differentiable.
    # Moreover, the modal sources should be located well away from the design
    # region.
    epsilon_r_bg = self.epsilon_r_bg()

    num_excite_ports = len(excite_port_idxs)
    flat_omegas = list(omegas.ravel(order='C'))
    num_omegas = len(flat_omegas)
    sparams = [[None] * num_omegas for _ in range(num_excite_ports)]
    efields = [[None] * num_omegas for _ in range(num_excite_ports)]

    def _simulate(excite_port_idx_and_omega):
      excite_port_idx, omega = excite_port_idx_and_omega

      sim = ceviche.fdfd_ez(
          omega,
          dl,
          epsilon_r_bg,
          [pml_width, pml_width],
      )
      sim.eps_r = epsilon_r
      source = self.ports[excite_port_idx].source_fdfd(
          omega,
          dl,
          epsilon_r_bg,
      )
      hx, hy, ez = sim.solve(source)

      sm = []
      sp = []
      for j, port in enumerate(self.ports):
        a, b = calculate_amplitudes(
            omega,
            dl,
            port,
            ez,
            hy,
            hx,
            epsilon_r_bg,
        )
        if j == excite_port_idx:
          sp = a
        sm.append(b)
      return excite_port_idx, omega, [smi / sp for smi in sm], ez

    # Run simulations in parallel across excitation ports and omegas.
    num_workers = max_parallelizm
    if not num_workers:
      num_workers = num_excite_ports * num_omegas
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers) as executor:
      simute_results = list(
          executor.map(_simulate,
                       itertools.product(excite_port_idxs, flat_omegas)))

    # Collect results from multiple threads.
    for port, omega, sps, ez in simute_results:
      port_idx = excite_port_idxs.index(port)
      omega_idx = flat_omegas.index(omega)
      sparams[port_idx][omega_idx] = sps
      efields[port_idx][omega_idx] = ez

    # Stack into arrays and reshape for output
    sparams = npa.stack(sparams, axis=self.SPARAMS_INPUT_PORT_AXIS)
    efields = npa.stack(efields, axis=self.FIELDS_INPUT_PORT_AXIS)
    return sparams, efields

  def density(self, design_variable: np.ndarray) -> np.ndarray:
    """The combined (design + background) density distribution of the model."""
    if design_variable.shape != self.design_variable_shape:
      raise ValueError(
          'Invalid design variable shape. Got ({}, {},) but expected ({}, {},)'
          .format(
              design_variable.shape[0],
              design_variable.shape[1],
              self.design_variable_shape[0],
              self.design_variable_shape[1],
          ))
    return primitives.insert_design_variable(
        self.transform_design_variable(design_variable),
        self.density_bg,
        self.design_region_coords,
    )

  def epsilon_r(self, design_variable: np.ndarray) -> np.ndarray:
    """The combined permittivity distribution of the model."""
    return self._epsilon_r(self.density(design_variable))

  def epsilon_r_bg(self) -> np.ndarray:
    """The background permittivity distribution of the model."""
    return self._epsilon_r(self.density_bg)

  def _epsilon_r(self, density: np.ndarray) -> np.ndarray:
    """Helper function for mapping density values to permittivity values."""
    return self.cladding_permittivity + (self.slab_permittivity -
                                         self.cladding_permittivity) * density

  def transform_design_variable(self,
                                design_variable: np.ndarray) -> np.ndarray:
    """Transformation of design variable before inlaying to the design region.

    By default this transformation is the identity operation and
    `self.design_variable_shape == self.design_region_shape`. However, models
    may override this operation to perform transformations on the design
    variable such as resampling or mirroring.

    Args:
      design_variable: An `np.ndarray` of shape `self.design_variable_shape`
        that corresponds to the design variable.

    Returns:
      An `np.ndarray` of shape `self.design_region_shape` that can be inlaid to
        the design region.
    """
    return design_variable

  @property
  def design_region(self) -> np.ndarray:
    """A boolean mask for the design region."""
    x0, y0, x1, y1 = self.design_region_coords
    mask = np.zeros(self.shape, dtype=bool)
    mask[x0:x1, y0:y1] = True
    return mask

  @property
  def design_variable_shape(self) -> Tuple[int, int]:
    """Shape of the design variable, in grid units."""
    return self.design_region_shape

  @property
  def design_region_shape(self) -> Tuple[int, int]:
    """Shape of the design region, in grid units."""
    (x_min, y_min, x_max, y_max) = self.design_region_coords
    return (x_max - x_min, y_max - y_min)

  @property
  @abc.abstractmethod
  def shape(self) -> Tuple[int, int]:
    """Shape of the simulation domain, in grid units."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def design_region_coords(self) -> Tuple[int, int, int, int]:
    """The coordinates of the design region as (x_min, y_min, x_max, y_max)."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def slab_permittivity(self) -> float:
    """The slab permittivity of the model."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def cladding_permittivity(self) -> float:
    """The cladding permittivity of the model."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def density_bg(self) -> np.ndarray:
    """The background density distribution of the model."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def dl(self) -> float:
    """The grid resolution of the model."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def pml_width(self) -> int:
    """The width of the PML region, in grid units."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def ports(self) -> List[modes.Port]:
    """A list of the device ports."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def output_wavelengths(self) -> List[float]:
    """A list of the wavelengths, in nm, to output fields and s-parameters."""
    raise NotImplementedError
