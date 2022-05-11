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
"""Tests for ceviche_challenges.wdm.model."""

from absl.testing import absltest

from ceviche_challenges import units as u
from ceviche_challenges.wdm import model
from ceviche_challenges.wdm import prefabs
from ceviche_challenges.wdm import spec as _spec

import numpy as np

# Expected scattering matrix rows for the WDM device with a "constant 1"
# design variable, at wavelengths 1270 and 1290 (the default simulation
# wavelengths) and 1310.

WDM_SPARAM_1270NM = np.array([
    0.203882 - 0.116885j,  # S11
    0.092964 + 0.326927j,  # S21
    0.092963 + 0.326926j,  # S31
])

WDM_SPARAM_1290NM = np.array([
    -0.043613 + 0.164152j,  # S11
    -0.247385 + 0.169515j,  # S21
    -0.247388 + 0.169516j,  # S31
])

WDM_SPARAM_1310NM = np.array([
    -0.004384 - 0.058996j,  # S11
    -0.222785 - 0.093600j,  # S21
    -0.222784 - 0.093600j,  # S31
])

h = 40 * u.nm
TINY_SPEC = _spec.WdmSpec(
    extent_ij=(20 * h, 16 * h),
    input_wg_j=7 * h,
    output_wgs_j=[5 * h, 9 * h],
    wg_width=2 * h,
    wg_mode_padding=h,
    input_mode_i=2 * h,
    output_mode_i=18 * h,
    variable_region=(u.Array([6 * h, 2 * h],
                             u.nm), u.Array([14 * h, 14 * h], u.nm)),
    cladding_permittivity=2,
    slab_permittivity=4,
    input_monitor_offset=1 * h,
    pml_width=1,
)
TINY_PARAMS = prefabs.wdm_sim_params(resolution=h)


class SimulationsGoldenTest(absltest.TestCase):

  def test_wdm(self):
    """Test of the model running the wdm prefab to produce known values."""
    params = prefabs.wdm_sim_params()
    spec = prefabs.wdm_spec()
    m = model.WdmModel(params, spec)
    design_var = np.ones(m.design_variable_shape)
    s_params, _ = m.simulate(design_var)
    np.testing.assert_allclose(
        s_params,
        np.asarray([[WDM_SPARAM_1270NM], [WDM_SPARAM_1290NM]]),
        atol=1e-5,
    )

  def test_wdm_with_injected_wavelenghts(self):
    """Tests running the wdm with explicitly specified wavelengths."""
    params = prefabs.wdm_sim_params()
    spec = prefabs.wdm_spec()
    m = model.WdmModel(params, spec)
    design_var = np.ones(m.design_variable_shape)

    # Test explicitly passing 1270, 1290, 1310nm.
    s_params, _ = m.simulate(design_var, wavelengths_nm=[1270., 1290., 1310.])
    np.testing.assert_allclose(
        s_params,
        np.asarray([[WDM_SPARAM_1270NM], [WDM_SPARAM_1290NM],
                    [WDM_SPARAM_1310NM]]),
        atol=1e-5,
    )


class ModelTest(absltest.TestCase):

  def test_model_design_var(self):
    """Test of incorrectly sized design variable at various model inputs."""
    params = prefabs.wdm_sim_params()
    spec = prefabs.wdm_spec()
    m = model.WdmModel(params, spec)
    with self.assertRaisesRegex(ValueError, 'Invalid design variable shape'):
      design_var = np.ones((10, 10))
      m.simulate(design_var)
    with self.assertRaisesRegex(ValueError, 'Invalid design variable shape'):
      design_var = np.ones((10, 10))
      m.density(design_var)
    with self.assertRaisesRegex(ValueError, 'Invalid design variable shape'):
      design_var = np.ones((10, 10))
      m.epsilon_r(design_var)

  def test_model_invalid_excitations(self):
    """Test of various invalid port excitations."""
    params = prefabs.wdm_sim_params()
    spec = prefabs.wdm_spec()
    m = model.WdmModel(params, spec)
    with self.assertRaisesRegex(ValueError, 'Invalid port index'):
      design_var = np.ones(m.design_variable_shape)
      m.simulate(design_var, excite_port_idxs=(-1, 0))
    with self.assertRaisesRegex(ValueError, 'Invalid port index'):
      design_var = np.ones(m.design_variable_shape)
      m.simulate(design_var, excite_port_idxs=(-1, 0, 1, 2, 3, 4, 5))
    with self.assertRaisesRegex(ValueError, 'Duplicate port index specified'):
      design_var = np.ones(m.design_variable_shape)
      m.simulate(design_var, excite_port_idxs=(0, 1, 1))
    with self.assertRaisesRegex(ValueError, 'Ports specified in'):
      design_var = np.ones(m.design_variable_shape)
      m.simulate(design_var, excite_port_idxs=(2, 1, 0))

  def test_model_shape(self):
    """Test of consistently sized outputs w.r.t. inputs."""
    params = prefabs.wdm_sim_params()
    spec = prefabs.wdm_spec()
    m = model.WdmModel(params, spec)
    design_var = np.ones(m.design_variable_shape)
    s_params, fields = m.simulate(design_var)
    self.assertLen(params.wavelengths, fields.shape[0])
    self.assertLen(params.wavelengths, s_params.shape[0])
    self.assertLen(m.ports, s_params.shape[2])
    self.assertEqual(fields.shape[2:], m.shape)
    # The number of elements in mask should sum to the design region area
    self.assertEqual(m.design_region.sum(), np.prod(m.design_variable_shape))
    excitation_ports = (0, 1)
    s_params, fields = m.simulate(design_var, excite_port_idxs=excitation_ports)
    self.assertLen(excitation_ports, s_params.shape[1])
    self.assertLen(excitation_ports, fields.shape[1])
    excitation_ports = (0, 1, 2)
    s_params, fields = m.simulate(design_var, excite_port_idxs=excitation_ports)
    self.assertLen(excitation_ports, s_params.shape[1])
    self.assertLen(excitation_ports, fields.shape[1])

  def test_model_epsilon_r_bounds(self):
    """Test of model epsilon_r output value bounds, e.g. min and max."""
    params = prefabs.wdm_sim_params()
    spec = prefabs.wdm_spec()
    m = model.WdmModel(params, spec)
    self.assertEqual(m.epsilon_r_bg().min(), spec.cladding_permittivity)
    self.assertEqual(m.epsilon_r_bg().max(), spec.slab_permittivity)
    design_var = np.ones(m.design_variable_shape)
    self.assertEqual(m.epsilon_r(design_var).min(), spec.cladding_permittivity)
    self.assertEqual(m.epsilon_r(design_var).max(), spec.slab_permittivity)

  def test_model_epsilon_r_full(self):
    """Test of model epsilon_r output distribution on a tiny model."""
    params = TINY_PARAMS
    spec = TINY_SPEC
    m = model.WdmModel(params, spec)
    density_tgt = np.array([
        [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
        [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
        [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
        [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
        [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
        [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
        [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
        [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
        [0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
    ])
    np.testing.assert_equal(density_tgt, m.density_bg)
    epsilon_r_bg_tgt = np.array([
        [2., 2., 2., 2., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 2., 2.],
        [2., 2., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 2., 2.],
        [2., 2., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 2., 2.],
        [2., 2., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 2., 2.],
        [2., 2., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 2., 2.],
        [2., 2., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 2., 2.],
        [2., 2., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 2., 2.],
        [2., 2., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 2., 2.],
        [2., 2., 2., 2., 4., 4., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 4., 4., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 4., 4., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 4., 4., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 4., 4., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 4., 4., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2.],
    ])
    np.testing.assert_equal(epsilon_r_bg_tgt, m.epsilon_r_bg())
    design_var = np.ones(m.design_variable_shape) * 0.5
    epsilon_r_tgt = np.array([
        [2., 2., 2., 2., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 2., 2.],
        [2., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 2., 2.],
        [2., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 2., 2.],
        [2., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 2., 2.],
        [2., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 2., 2.],
        [2., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 2., 2.],
        [2., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 2., 2.],
        [2., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 2., 2.],
        [2., 2., 2., 2., 4., 4., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 4., 4., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 4., 4., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 4., 4., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 4., 4., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 4., 4., 2., 2., 4., 4., 2., 2., 2., 2., 2., 2.],
    ])
    np.testing.assert_equal(epsilon_r_tgt, m.epsilon_r(design_var))


if __name__ == '__main__':
  absltest.main()
