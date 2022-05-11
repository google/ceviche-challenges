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
"""Tests for ceviche_challenges.defs."""

from absl.testing import absltest
from ceviche_challenges import defs


class DefsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.dir_xp = defs.Direction.X_POS
    self.dir_xn = defs.Direction.X_NEG
    self.dir_yp = defs.Direction.Y_POS
    self.dir_yn = defs.Direction.Y_NEG

  def test_direction_axis(self):
    """Test the `Direction` along axis functionality."""
    self.assertTrue(self.dir_xp.is_along_x)
    self.assertTrue(self.dir_xn.is_along_x)
    self.assertFalse(self.dir_xp.is_along_y)
    self.assertFalse(self.dir_xn.is_along_y)

    self.assertTrue(self.dir_yp.is_along_y)
    self.assertTrue(self.dir_yn.is_along_y)
    self.assertFalse(self.dir_yp.is_along_x)
    self.assertFalse(self.dir_yn.is_along_x)

  def test_direction_sign(self):
    """Test the `Direction` sign functionality."""
    self.assertGreater(self.dir_yp.sign, 0)
    self.assertGreater(self.dir_xp.sign, 0)
    self.assertLess(self.dir_yn.sign, 0)
    self.assertLess(self.dir_xn.sign, 0)

  def test_direction_index(self):
    """Test the `Direction` index generation functionality."""
    self.assertEqual(self.dir_yp.index, 1)
    self.assertEqual(self.dir_yn.index, 1)
    self.assertEqual(self.dir_xp.index, 0)
    self.assertEqual(self.dir_xn.index, 0)


if __name__ == '__main__':
  absltest.main()
