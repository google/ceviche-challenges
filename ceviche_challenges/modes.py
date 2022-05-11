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
"""Port definition and low level waveguide eigenmode solver implementation.

This module defines a class for describing a port in ceviche. It also implements
a waveguide eigenmode solver for solving the eigenmodes of such ports. Note that
we limit ourselves to the polarization with E along the z-direction (out of the
2D geometry plane). Below is an illustration of the port slice concept.

  ILLUSTRATION OF A WAVEGUIDE PORT SLICE IN 2D DOMAIN
  .______________________________________.
  |                                      |
  |                1D                    |
  |                port                  |
  |                slice                 |
  |                  :                   |
  |   waveguide      :                   |
  |   cladding       :                   |
  |                  :                   |
  |##################:###################|
  |##################:###################|
  |## waveguide  ####:###################|
  |####  core  ######:###################|
  |##################:###################|
  |##################:###################|
  |                  :                   |
  |   waveguide      :                   |      y
  |   cladding       :                   |
  |                  :----> propagation  |      ^
  |                          direction   |      |
  |                                      |      |
  .______________________________________.      .--->  x

                                                z
  NOTE: a port could also be defined for a
  waveguide propagating in the y-direction

"""

import dataclasses
from typing import Tuple

from ceviche import constants
from ceviche import derivatives
from ceviche_challenges import defs

import matplotlib as mpl
import numpy as np
import scipy


@dataclasses.dataclass
class Port:
  """A generic port, represented as a slice (a line of pixels) in a 2D domain.

  Attributes:
    x: `int` specifying the x-coordinate of the center of the mode slice, in
      pixel units.
    y: `int` specifying the y-coordinate of the center of the mode slice, in
      pixel units.
    width: `int` specifying the transverse width of the mode slice, in pixel
      units.
    dir: `Direction` specifying the direction perpendicular to the mode slice.
    offset: `int` specfying distance from the source slice to perform the
      decomposition of the forward and backward mode amplitudes.
  """
  x: int
  y: int
  width: int
  dir: defs.Direction
  offset: int

  def __post_init__(self):
    if self.width % 2 == 1:
      raise ValueError(
          'Odd values for the port width are currently not supported')

  def coords(self) -> defs.Slice:
    """Generate coordinate vectors for slicing in/out of 2D arrays."""
    # pylint:disable=g-bad-todo
    # TODO: correctly handle an odd width value, rather than round off
    if self.dir.is_along_x:
      x_coords = np.ones((self.width,), dtype=int) * self.x
      y_coords = np.arange(
          self.y - self.width // 2, self.y + self.width // 2, dtype=int)
    else:
      y_coords = np.ones((self.width,), dtype=int) * self.y
      x_coords = np.arange(
          self.x - self.width // 2, self.x + self.width // 2, dtype=int)

    return (x_coords, y_coords)

  def source_fdfd(
      self,
      omega: float,
      dl: float,
      epsilon_r: defs.Geometry,
  ) -> defs.Field:
    """Generate a source array to supply to a ceviche simulation.

    In ceviche (and FDFD methods in general) we need a source to define the RHS
    of our `Ax = b` problem, where `A` is the operator defining Maxwell's
    equations, `x` is the unknown field distribuion, and `b` is an electric or
    magnetic current source. This function provides an interface for generating
    the `b` for a `Port`.

    Args:
      omega: `float` specifying the angular frequency of the mode, in units of
        rad/sec.
      dl: `float` specifying the spatial grid cell size, in units of meters.
      epsilon_r: `Geometry` specifying the permitivitty distribution.

    Returns:
      A source `Field` with the same shape as the input geometry, `epsilon_r`.
      The source is zero at all points except those within the port slice, where
      the source takes on the complex values of the mode field.
    """
    coords = self.coords()
    source = np.zeros(epsilon_r.shape, dtype=complex)
    e, _, _ = self.field_profiles(epsilon_r[coords], omega, dl)
    source[coords] = e
    return source

  def plot(self, ax: mpl.axes.Axes, c: str = 'k', alpha=0.5):
    """Plot the mode slice into a matplotlib axes."""
    coords = self.coords()
    ax.plot(coords[0], coords[1], '-', c=c, alpha=alpha)
    if self.dir.sign > 0:
      offset = self.offset
    else:
      offset = -self.offset
    if self.dir.is_along_x:
      ax.plot(coords[0] + offset, coords[1], ':', c=c, alpha=alpha)
    else:
      ax.plot(coords[0], coords[1] + offset, ':', c=c, alpha=alpha)

  def signed_offset(self):
    """The signed offset for computing the fwd/bkwd mode amplitudes."""
    return self.offset * self.dir.sign

  def field_profiles(self, epsilon_r: defs.Geometry, omega: float, dl: float):
    """Computes electric field, magnetic field, and wave vector of the port.

    Args:
      epsilon_r: `Geometry` specifying the permitivitty distribution of the
        waveguide slice.
      omega: `float` specifying the angular frequency of the mode, in units of
        rad/sec.
      dl: `float` specifying the spatial grid cell size, in units of meters.

    Returns:
      The electric field, the magnetic field, and the wave vector.
    """
    raise NotImplementedError


@dataclasses.dataclass
class WaveguidePort(Port):
  """A waveguide port.

  Attributes:
    order: integer specifying the transverse order of the mode, with the
      fundamental mode corresponding to `order=1`.
  """
  order: int

  def field_profiles(self, epsilon_r: defs.Geometry, omega: float, dl: float):
    """Computes electric and magnetic field profile of the waveguide mode."""
    return solve_modes(epsilon_r, omega, dl, self.order)


def solve_modes(
    epsilon_r: defs.Geometry,
    omega: float,
    dl: float,
    order: int = 1,
) -> Tuple[defs.Field, defs.Field, float]:
  """Low level eigenmode solver to compute mode for a geometry slice.

  Args:
    epsilon_r: `Geometry` specifying the permitivitty distribution of the
      waveguide slice.
    omega: `float` specifying the angular frequency of the mode, in units of
      rad/sec.
    dl: `float` specifying the spatial grid cell size, in units of meters.
    order: `int` specifying the transverse order of the mode, with the
      fundamental mode corresponding to `order=1`. Defaults to the fundamental
      mode.

  Returns:
    The electric field, the magnetic field, and the real part of the mode
    wave vector.
  """
  k0 = omega / constants.C_0

  dxf, dxb, _, _ = derivatives.compute_derivative_matrices(
      omega, (epsilon_r.size, 1), [0, 0], dL=dl)

  diag_eps_r = scipy.sparse.spdiags(epsilon_r.ravel(), [0], epsilon_r.size,
                                    epsilon_r.size)

  # Solves the eigenvalue problem:
  #
  #    [ ∂²/∂x² + εr k₀² ] E = β² E
  #
  # where E is the transverse electric field component of the eigenmode and β²
  # is the eigenvalue. β corresponds to the guided wavevector of the eigenmode.
  vals, vecs = scipy.sparse.linalg.eigs(
      dxf.dot(dxb) + k0**2 * diag_eps_r,
      k=order,
      v0=epsilon_r.ravel(),
      which='LR')

  # Sort the eigenmodes because apparently scipy does not guarantee this
  betas = np.real(np.sqrt(vals, dtype=complex))
  inds_sorted = np.argsort(betas)
  e = vecs[:, inds_sorted[0]]
  beta = betas[inds_sorted[0]]

  # Compute transverse magnetic field as:
  #
  #    H = β / (μ₀ ω) E
  #
  # where the β term originates from the spatial derivative in the propagation
  # direction.
  h = beta / omega / constants.MU_0 * e

  return e, h, beta
