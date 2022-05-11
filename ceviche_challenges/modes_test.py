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
"""Tests for ceviche_challenges.modes."""

from absl.testing import absltest
from ceviche import constants

from ceviche_challenges import defs
from ceviche_challenges import modes

import numpy as np

class ModesTest(absltest.TestCase):

  def test_port(self):
    """Test the port definition."""
    port = modes.WaveguidePort(
        x=10,
        y=20,
        width=20,
        order=1,
        dir=defs.Direction.X_POS,
        offset=1,
    )
    np.testing.assert_array_equal(port.coords()[0], port.x)

    self.assertEqual(port.coords()[0].size, port.width)
    self.assertEqual(port.coords()[1].size, port.width)

    port = modes.WaveguidePort(
        x=10,
        y=20,
        width=20,
        order=1,
        dir=defs.Direction.X_NEG,
        offset=1,
    )
    np.testing.assert_array_equal(port.coords()[0], port.x)

    self.assertEqual(port.coords()[0].size, port.width)
    self.assertEqual(port.coords()[1].size, port.width)

    port = modes.WaveguidePort(
        x=10,
        y=20,
        width=20,
        order=1,
        dir=defs.Direction.Y_POS,
        offset=1,
    )
    np.testing.assert_array_equal(port.coords()[1], port.y)

    self.assertEqual(port.coords()[0].size, port.width)
    self.assertEqual(port.coords()[1].size, port.width)

    port = modes.WaveguidePort(
        x=10,
        y=20,
        width=20,
        order=1,
        dir=defs.Direction.Y_NEG,
        offset=1,
    )
    np.testing.assert_array_equal(port.coords()[1], port.y)

    self.assertEqual(port.coords()[0].size, port.width)
    self.assertEqual(port.coords()[1].size, port.width)

    with self.assertRaises(ValueError):
      modes.WaveguidePort(
          x=10,
          y=20,
          width=21,
          order=1,
          dir=defs.Direction.X_POS,
          offset=1,
      )

  def test_solver(self):
    """Test the eigenmode solver."""
    # Create a silicon waveguide cross section
    n = 150
    width = 20
    omega = 200e12 * 2 * np.pi
    dl = 25e-9
    k0 = omega / constants.C_0
    epsilon_r = np.ones((n,))
    epsilon_r[n // 2 - width // 2:n // 2 + width // 2] = 12.25

    # Solve for modes of different transverse order
    _, _, beta1 = modes.solve_modes(epsilon_r, omega, dl, order=1)
    _, _, beta2 = modes.solve_modes(epsilon_r, omega, dl, order=2)
    _, _, beta3 = modes.solve_modes(epsilon_r, omega, dl, order=3)
    _, _, beta4 = modes.solve_modes(epsilon_r, omega, dl, order=4)

    # Higher order modes should have successively smaller wave vectors.
    self.assertGreater(beta1, beta2)
    self.assertGreater(beta2, beta3)
    self.assertGreater(beta3, beta4)

    # Modes should always be guided, meaning that they are below the light line.
    self.assertGreater(beta1 / k0, 1.)
    self.assertGreater(beta2 / k0, 1.)
    self.assertGreater(beta3 / k0, 1.)

    # For the above waveguide design, the fourth order mode should be above the
    # light line, meaning that beta4 / k0 < 1.0
    self.assertLess(beta4 / k0, 1.)


if __name__ == '__main__':
  absltest.main()
