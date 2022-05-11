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
"""Tests for ceviche_challenges.mode_converter.model."""

import numpy as np

from ceviche_challenges import units as u
from ceviche_challenges.mode_converter import model as _model
from ceviche_challenges.mode_converter import prefabs

from absl.testing import absltest
from absl.testing import parameterized

_ATOL_SCATTERING_PARAMETERS = 1e-2


class ModelScatteringTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': '12_resolution_50nm',
          'resolution': 50 * u.nm,
          'spec_prefab_fn': prefabs.mode_converter_spec_12,
      },
      {
          'testcase_name': '13_resolution_50nm',
          'resolution': 50 * u.nm,
          'spec_prefab_fn': prefabs.mode_converter_spec_13,
      },
      {
          'testcase_name': '23_resolution_50nm',
          'resolution': 50 * u.nm,
          'spec_prefab_fn': prefabs.mode_converter_spec_23,
      },
      {
          'testcase_name': '12_resolution_25nm',
          'resolution': 25 * u.nm,
          'spec_prefab_fn': prefabs.mode_converter_spec_12,
      },
      {
          'testcase_name': '13_resolution_25nm',
          'resolution': 25 * u.nm,
          'spec_prefab_fn': prefabs.mode_converter_spec_13,
      },
      {
          'testcase_name': '23_resolution_25nm',
          'resolution': 25 * u.nm,
          'spec_prefab_fn': prefabs.mode_converter_spec_23,
      },
  )
  def test_scattering_matrix_symmetry(
      self,
      resolution=50 * u.nm,
      spec_prefab_fn=prefabs.mode_converter_spec_12,
  ):
    """Test that the scattering matrix is symmetric (reciprocal)."""
    params = prefabs.mode_converter_sim_params(resolution=resolution)
    spec = spec_prefab_fn()
    model = _model.ModeConverterModel(params, spec)

    design_var = np.ones(model.design_variable_shape)
    s_params, _ = model.simulate(
        design_var,
        excite_port_idxs=(0, 1),
    )
    s = np.abs(s_params.squeeze())
    # Check overall scattering matrix symmetry
    np.testing.assert_allclose(
        s,
        s.T,
        atol=_ATOL_SCATTERING_PARAMETERS,
    )


if __name__ == '__main__':
  absltest.main()
