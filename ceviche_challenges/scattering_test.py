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
"""Tests for ceviche_challenges.scattering."""

from typing import List, Tuple

from absl.testing import absltest
from absl.testing import parameterized

import ceviche
from ceviche_challenges import units as u
from ceviche_challenges import defs
from ceviche_challenges import modes
from ceviche_challenges import scattering

import numpy as np

# Maximum reflection from a straight waveguide
_THRESHOLD_R_DB = -40  # dB

# Minimum transmission through a straight waveguide
_THRESHOLD_T_DB = -1e-3  # dB

# Minimum cross-mode transmission through a straight waveguide
_THRESHOLD_T_CROSS_DB = -200  # dB

_WG_PERMITTIVITY = 12

_REL_ERROR_TOL = 1e-4


def power_dB(x):  # pylint: disable=invalid-name
  return 20 * np.log10(np.abs(x))


class StraightWaveguideScatteringTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.npml = 20
    self.omega = 2 * np.pi * 200e12
    self.dl = 25e-9

  def build_model(self,
                  shape: Tuple[int, int] = (100, 100),
                  wg_width: int = 10,
                  wg_padding: int = 20,
                  direction: defs.Direction = defs.Direction.X_POS,
                  order: int = 1,
                  offset: int = 5) -> Tuple[defs.Geometry, List[modes.Port]]:
    """Build a straight waveguide model.

    The straight waveguide can be oriented along either the x or y direction

    Args:
      shape: The size of the domain, in pixel units
      wg_width: The width of the waveguide core, in pixel units
      wg_padding: The padding on either side of the waveguide core, in pixel
        units
      direction: The directin of the waveguide
      order: The order of the waveguide mode
      offset: The offset from the port, in pixel units, to decompose fwd/bwd
        modes

    Returns:
      The geometry and a list of ports
    """
    epsr = np.ones(shape)
    if direction.is_along_x:
      epsr[:, shape[1] // 2 - wg_width // 2:shape[1] // 2 +
           wg_width // 2] = _WG_PERMITTIVITY
      port1 = modes.WaveguidePort(
          x=self.npml + 1,
          y=shape[1] // 2,
          width=wg_width + 2 * wg_padding,
          order=order,
          dir=defs.Direction.X_POS,
          offset=offset)
      port2 = modes.WaveguidePort(
          x=shape[0] - self.npml - 1,
          y=shape[1] // 2,
          width=wg_width + 2 * wg_padding,
          order=order,
          dir=defs.Direction.X_NEG,
          offset=offset)
    else:
      epsr[shape[0] // 2 - wg_width // 2:shape[0] // 2 +
           wg_width // 2, :] = _WG_PERMITTIVITY
      port1 = modes.WaveguidePort(
          y=self.npml + 1,
          x=shape[0] // 2,
          width=wg_width + 2 * wg_padding,
          order=order,
          dir=defs.Direction.Y_POS,
          offset=offset)
      port2 = modes.WaveguidePort(
          y=shape[1] - self.npml - 1,
          x=shape[0] // 2,
          width=wg_width + 2 * wg_padding,
          order=order,
          dir=defs.Direction.Y_NEG,
          offset=offset)

    return epsr, [port1, port2]

  def run_model(self, epsr, ports,
                which_port) -> Tuple[complex, complex, complex, complex]:
    """Run a ceviche simulation on a model.

    Args:
      epsr: the permittivity distribution of the model to simulate
      ports: a list containing the two waveguide ports
      which_port: the index of the port in `ports` to excite

    Returns:
      the complex-valued scattering parameters: s_1+, s_1-, s_2+, s_2-
    """
    simulation = ceviche.fdfd_ez(self.omega, self.dl, epsr,
                                 [self.npml, self.npml])

    hx, hy, ez = simulation.solve(ports[which_port].source_fdfd(
        self.omega, self.dl, epsr))
    s1p, s1m = scattering.calculate_amplitudes(self.omega, self.dl, ports[0],
                                               ez, hy, hx, epsr)
    s2p, s2m = scattering.calculate_amplitudes(self.omega, self.dl, ports[1],
                                               ez, hy, hx, epsr)
    return s1p, s1m, s2p, s2m

  def test_xplus(self):
    """Measure transmission and reflection from a straight waveguide.

    This test excites the fundamental mode in the +x direction
    """
    epsr, ports = self.build_model(
        offset=5,
        direction=defs.Direction.X_POS,
        order=1,
        shape=(100, 100),
        wg_width=10)
    s1p, s1m, _, s2m = self.run_model(epsr, ports, 0)
    r_db = power_dB(s1m / s1p)
    t_db = power_dB(s2m / s1p)
    self.assertLess(r_db, _THRESHOLD_R_DB)
    self.assertGreater(t_db, _THRESHOLD_T_DB)

  def test_xplus_2nd_order_mode(self):
    """Measure transmission and reflection from a straight waveguide.

    This test excites the second-order mode in the +x direction
    """
    epsr, ports = self.build_model(
        offset=5,
        direction=defs.Direction.X_POS,
        order=2,
        shape=(130, 130),
        wg_width=20)
    s1p, s1m, _, s2m = self.run_model(epsr, ports, 0)
    r_db = power_dB(s1m / s1p)
    t_db = power_dB(s2m / s1p)
    self.assertLess(r_db, _THRESHOLD_R_DB)
    self.assertGreater(t_db, _THRESHOLD_T_DB)

  def test_xminus(self):
    """Measure transmission and reflection from a straight waveguide.

    This test excites the fundamental mode in the -x direction
    """
    epsr, ports = self.build_model(
        offset=5,
        direction=defs.Direction.X_NEG,
        order=1,
        shape=(100, 100),
        wg_width=10)
    _, s1m, s2p, s2m = self.run_model(epsr, ports, 1)
    r_db = power_dB(s2m / s2p)
    t_db = power_dB(s1m / s2p)
    self.assertLess(r_db, _THRESHOLD_R_DB)
    self.assertGreater(t_db, _THRESHOLD_T_DB)

  def test_xminus_2nd_order_mode(self):
    """Measure transmission and reflection from a straight waveguide.

    This test excites the second-order mode in the -x direction
    """
    epsr, ports = self.build_model(
        offset=5,
        direction=defs.Direction.X_NEG,
        order=2,
        shape=(130, 130),
        wg_width=20)
    _, s1m, s2p, s2m = self.run_model(epsr, ports, 1)
    r_db = power_dB(s2m / s2p)
    t_db = power_dB(s1m / s2p)
    self.assertLess(r_db, _THRESHOLD_R_DB)
    self.assertGreater(t_db, _THRESHOLD_T_DB)

  def test_yplus(self):
    """Measure transmission and reflection from a straight waveguide.

    This test excites the fundamental mode in the +y direction
    """
    epsr, ports = self.build_model(
        offset=5,
        direction=defs.Direction.Y_POS,
        order=1,
        shape=(100, 100),
        wg_width=10)
    s1p, s1m, _, s2m = self.run_model(epsr, ports, 0)
    r_db = power_dB(s1m / s1p)
    t_db = power_dB(s2m / s1p)
    self.assertLess(r_db, _THRESHOLD_R_DB)
    self.assertGreater(t_db, _THRESHOLD_T_DB)

  def test_yplus_2nd_order_mode(self):
    """Measure transmission and reflection from a straight waveguide.

    This test excites the second-order mode in the +y direction
    """
    epsr, ports = self.build_model(
        offset=5,
        direction=defs.Direction.Y_POS,
        order=2,
        shape=(130, 130),
        wg_width=20)
    s1p, s1m, _, s2m = self.run_model(epsr, ports, 0)
    r_db = power_dB(s1m / s1p)
    t_db = power_dB(s2m / s1p)
    self.assertLess(r_db, _THRESHOLD_R_DB)
    self.assertGreater(t_db, _THRESHOLD_T_DB)

  def test_yminus(self):
    """Measure transmission and reflection from a straight waveguide.

    This test excites the fundamental mode in the -y direction
    """
    epsr, ports = self.build_model(
        offset=5,
        direction=defs.Direction.Y_NEG,
        order=1,
        shape=(100, 100),
        wg_width=10)
    _, s1m, s2p, s2m = self.run_model(epsr, ports, 1)
    r_db = power_dB(s2m / s2p)
    t_db = power_dB(s1m / s2p)
    self.assertLess(r_db, _THRESHOLD_R_DB)
    self.assertGreater(t_db, _THRESHOLD_T_DB)

  def test_yminus_2nd_order_mode(self):
    """Measure transmission and reflection from a straight waveguide.

    This test excites the second-order mode in the -y direction
    """
    epsr, ports = self.build_model(
        offset=5,
        direction=defs.Direction.Y_NEG,
        order=2,
        shape=(130, 130),
        wg_width=20)
    _, s1m, s2p, s2m = self.run_model(epsr, ports, 1)
    r_db = power_dB(s2m / s2p)
    t_db = power_dB(s1m / s2p)
    self.assertLess(r_db, _THRESHOLD_R_DB)
    self.assertGreater(t_db, _THRESHOLD_T_DB)

  def test_x_1st_to_2nd(self):
    """Measure transmission and reflection from a straight waveguide.

    This test excites the fundamental mode in the +x direction and measures
    transmission into the second-order mode (which should be very small)
    """
    epsr, ports = self.build_model(
        offset=5,
        direction=defs.Direction.X_POS,
        order=1,
        shape=(130, 130),
        wg_width=20)
    ports[1].order = 2
    s1p, s1m, _, s2m = self.run_model(epsr, ports, 0)
    r_db = power_dB(s1m / s1p)
    t_db = power_dB(s2m / s1p)
    self.assertLess(r_db, _THRESHOLD_R_DB)
    self.assertLess(t_db, _THRESHOLD_T_CROSS_DB)

  def test_x_2nd_to_1st(self):
    """Measure transmission and reflection from a straight waveguide.

    This test excites the second-order mode in the +x direction and measures
    transmission into the fundamental mode (which should be very small)
    """
    epsr, ports = self.build_model(
        offset=5,
        direction=defs.Direction.X_POS,
        order=1,
        shape=(130, 130),
        wg_width=20)
    ports[1].order = 2
    _, s1m, s2p, s2m = self.run_model(epsr, ports, 1)
    r_db = power_dB(s2m / s2p)
    t_db = power_dB(s1m / s2p)
    self.assertLess(r_db, _THRESHOLD_R_DB)
    self.assertLess(t_db, _THRESHOLD_T_CROSS_DB)


class TransmissionComparisonTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'kink_0nm',
          'transpose': False,
          'offset_kink': 0 * u.nm,
      },
      {
          'testcase_name': 'kink_0nm_transposed',
          'transpose': True,
          'offset_kink': 0 * u.nm,
      },
      {
          'testcase_name': 'kink_200nm',
          'transpose': False,
          'offset_kink': 200 * u.nm,
      },
      {
          'testcase_name': 'kink_200nm_transposed',
          'transpose': True,
          'offset_kink': 200 * u.nm,
      },
      {
          'testcase_name': 'kink_400nm',
          'transpose': False,
          'offset_kink': 400 * u.nm,
      },
      {
          'testcase_name': 'kink_400nm_transposed',
          'transpose': True,
          'offset_kink': 400 * u.nm,
      },
      {
          'testcase_name': 'kink_600nm',
          'transpose': False,
          'offset_kink': 600 * u.nm,
      },
      {
          'testcase_name': 'kink_600nm_transposed',
          'transpose': True,
          'offset_kink': 600 * u.nm,
      },
      {
          'testcase_name': 'kink_800nm',
          'transpose': False,
          'offset_kink': 800 * u.nm,
      },
      {
          'testcase_name': 'kink_800nm_transposed',
          'transpose': True,
          'offset_kink': 800 * u.nm,
      },
  )
  def test_scattering_symmetry(
      self,
      permittivity: float = 12.,
      wg_width: u.Quantity = 300 * u.nm,
      wg_padding: u.Quantity = 600 * u.nm,
      mode_order: int = 1,
      monitor_offset: int = 1,
      source_pml_distance: u.Quantity = 50 * u.nm,
      omega: float = 2 * np.pi * 200e12,
      dl=25 * u.nm,
      npml: int = 10,
      length_feed: u.Quantity = 600 * u.nm,
      length_kink: u.Quantity = 300 * u.nm,
      offset_kink: u.Quantity = 400 * u.nm,
      transpose: bool = False,
  ):
    """Measure the asymmetry of the scattering matrix for a kinked waveguide.

    Asymmetry is quantified via the calculation ||S-S^T||/||S||, where S is the
    amplitude of the scattering matrix.

    Args:
      permittivity: Permittivity of the waveguide core region.
      wg_width: The width of the waveguide, in real units.
      wg_padding: The padding around the waveguide for the port, in real units.
      mode_order: The order of the modes.
      monitor_offset: The offset distance between the source and the monitor, in
        grid cells.
      source_pml_distance: The distance between the pml and the source, in real
        units.
      omega: A `float` specifying the angular frequency for the simulation.
      dl: The resolution of the simulation, in real units.
      npml: The number of grid cells to use for the pml.
      length_feed: The length of the straight feed waveguide before and after
        the kink, in real units.
      length_kink: The length of the kink in the direction of propagation, in
        real units.
      offset_kink: The offset of the input and output waveguide in the
        transverse direction, specifying the size of the kink, in real units.
      transpose: A `bool` specifying whether or not the simulation should be
        transposed, allowing it to run in the y-direction.
    """

    shape = (
        2 * npml + u.resolve(length_kink + 2 * length_feed, dl),
        2 * npml + u.resolve(offset_kink + wg_width + 2 * wg_padding, dl),
    )

    epsr = np.ones(shape)
    epsr[:npml + u.resolve(length_feed, dl),
         npml + u.resolve(wg_padding, dl):npml +
         u.resolve(wg_padding + wg_width, dl),] = permittivity
    epsr[-npml - u.resolve(length_feed, dl):,
         npml + u.resolve(wg_padding + offset_kink, dl):npml +
         u.resolve(wg_padding + offset_kink + wg_width, dl),] = permittivity
    epsr[npml + u.resolve(length_feed, dl):npml +
         u.resolve(length_feed + length_kink, dl),
         npml + u.resolve(wg_padding, dl):npml +
         u.resolve(wg_padding + offset_kink + wg_width, dl),] = permittivity

    p1_x = npml + u.resolve(source_pml_distance, dl)
    p1_y = npml + u.resolve(wg_padding + wg_width / 2, dl)
    p2_x = shape[0] - npml - u.resolve(source_pml_distance, dl)
    p2_y = npml + u.resolve(wg_padding + wg_width / 2 + offset_kink, dl)
    p1_dir = defs.Direction.X_POS
    p2_dir = defs.Direction.X_NEG
    if transpose:
      tmp = p1_x
      p1_x = p1_y
      p1_y = tmp
      tmp = p2_x
      p2_x = p2_y
      p2_y = tmp
      p1_dir = defs.Direction.Y_POS
      p2_dir = defs.Direction.Y_NEG
      epsr = epsr.T

    port1 = modes.WaveguidePort(
        x=p1_x,
        y=p1_y,
        width=u.resolve(wg_width + 2 * wg_padding, dl),
        order=mode_order,
        dir=p1_dir,
        offset=monitor_offset)
    port2 = modes.WaveguidePort(
        x=p2_x,
        y=p2_y,
        width=u.resolve(wg_width + 2 * wg_padding, dl),
        order=mode_order,
        dir=p2_dir,
        offset=monitor_offset)

    ports = [port1, port2]

    simulation = ceviche.fdfd_ez(omega, dl.to_value('m'), epsr, [npml, npml])

    s = []
    fields = []
    hx, hy, ez = simulation.solve(ports[0].source_fdfd(omega, dl.to_value('m'),
                                                       epsr))
    s1p, s1m = scattering.calculate_amplitudes(
        omega,
        dl.to_value('m'),
        ports[0],
        ez,
        hy,
        hx,
        epsr,
    )
    s2p, s2m = scattering.calculate_amplitudes(
        omega,
        dl.to_value('m'),
        ports[1],
        ez,
        hy,
        hx,
        epsr,
    )
    s.append([s1m / s1p, s2m / s1p])
    fields.append(ez)

    ###

    hx, hy, ez = simulation.solve(ports[1].source_fdfd(omega, dl.to_value('m'),
                                                       epsr))
    s1p, s1m = scattering.calculate_amplitudes(
        omega,
        dl.to_value('m'),
        ports[0],
        ez,
        hy,
        hx,
        epsr,
    )
    s2p, s2m = scattering.calculate_amplitudes(
        omega,
        dl.to_value('m'),
        ports[1],
        ez,
        hy,
        hx,
        epsr,
    )
    s.append([s1m / s2p, s2m / s2p])
    s = np.array(s)
    rerr = np.linalg.norm(np.abs(s) - np.abs(s).T) / np.linalg.norm(np.abs(s))
    self.assertLess(rerr, _REL_ERROR_TOL)


if __name__ == '__main__':
  absltest.main()
