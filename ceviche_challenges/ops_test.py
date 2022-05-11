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
"""Tests for ceviche_challenges.ops."""

from absl.testing import absltest
from ceviche_challenges import defs
from ceviche_challenges import ops
import numpy as np

_TOL_OVERLAP_MAG = 1e-6


class OpsTestCross(absltest.TestCase):

  def test_unitvectors(self):
    """Test the cross product operation on scalar unit vectors."""

    # X x X = 0
    self.assertEqual(ops.cross((1., 0., 0.), (1., 0., 0.)), (0., 0., 0.))

    # Y x Y = 0
    self.assertEqual(ops.cross((0., 1., 0.), (0., 1., 0.)), (0., 0., 0.))

    # Z x Z = 0
    self.assertEqual(ops.cross((0., 0., 1.), (0., 0., 1.)), (0., 0., 0.))

    # X x Y = Z
    self.assertEqual(ops.cross((1., 0., 0.), (0., 1., 0.)), (0., 0., 1.))

    # -X x Y = -Z
    self.assertEqual(ops.cross((-1., 0., 0.), (0., 1., 0.)), (0., 0., -1.))

    # Y x X = -Z
    self.assertEqual(ops.cross((0., 1., 0.), (1., 0., 0.)), (0., 0., -1.))

    # -Y x X = Z
    self.assertEqual(ops.cross((0., -1., 0.), (1., 0., 0.)), (0., 0., 1.))

    # X x Z = -Y
    self.assertEqual(ops.cross((1., 0., 0.), (0., 0., 1.)), (0., -1., 0.))

    # Z x X = Y
    self.assertEqual(ops.cross((0., 0., 1.), (1., 0., 0.)), (0., 1., 0.))

    # Z x Y = -X
    self.assertEqual(ops.cross((0., 0., 1.), (0., 1., 0.)), (-1., 0., 0.))

    # Y x Z = X
    self.assertEqual(ops.cross((0., 1., 0.), (0., 0., 1.)), (1., 0., 0.))


class OpsTestOverlap(absltest.TestCase):

  def setUp(self):
    super().setUp()
    x = np.linspace(0., 2 * np.pi, 99)
    self.a_sym = np.sin(x)
    self.a_asym = np.cos(x)

  def test_nonorthogonal(self):
    """Test the overlap integral operation on non-othogonal fields.

    e.g.   ∫ symmetric . (symmetric)*  ≠  0.
    """

    ans = ops.overlap(
        (0.0, 0.0, self.a_sym),
        (0.0, -self.a_sym, 0.0),
        defs.Direction.X_POS,
    )
    self.assertGreater(np.abs(ans), _TOL_OVERLAP_MAG)

  def test_orthogonal(self):
    """Test the overlap integral operation on non-othogonal fields.

    e.g.   ∫ symmetric . (anti-symmetric)*  =  0.
    """

    ans = ops.overlap(
        (0.0, 0.0, self.a_sym),
        (0.0, -self.a_asym, 0.0),
        defs.Direction.X_POS,
    )
    self.assertLess(np.abs(ans), _TOL_OVERLAP_MAG)

    ans = ops.overlap(
        (0.0, 0.0, self.a_sym),
        (0.0, self.a_asym, 0.0),
        defs.Direction.X_POS,
    )
    self.assertLess(np.abs(ans), _TOL_OVERLAP_MAG)


if __name__ == '__main__':
  absltest.main()
