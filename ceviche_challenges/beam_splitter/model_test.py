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
"""Tests for ceviche_challenges.beam_splitter.model."""

from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from ceviche_challenges import units as u
from ceviche_challenges.beam_splitter import model as _model
from ceviche_challenges.beam_splitter import prefabs


_ATOL_SCATTERING_PARAMETERS = 1e-2


def init_design_var_feature(
    radius_scale: float,
    shape: Tuple[int, int],
) -> np.ndarray:
  """Helper for initializing a design var with an embedded circular feature.

  Args:
    radius_scale: a `float` specifying the radius of the circular feature
      relative to the minimum dimension in `shape`.
    shape: the shape of the design variable to create.

  Returns:
    An `np.ndarray` corresponding to the initialized design variable.
  """
  design_var = np.ones(shape)
  r0 = radius_scale * np.min(shape)
  j, i = np.meshgrid(
      np.arange(0, shape[1]),
      np.arange(0, shape[0]),
  )
  # Position the origin off center to break symmetry
  r = (i - shape[0] / 4)**2 + (j - shape[1] / 4)**2
  design_var[r < r0] = 0.0
  return design_var


class ModelScatteringTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'radius_scale_1_resolution_20nm',
          'radius_scale': 1,
          'resolution': 20 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_2_resolution_20nm',
          'radius_scale': 2,
          'resolution': 20 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_3_resolution_20nm',
          'radius_scale': 3,
          'resolution': 20 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_4_resolution_20nm',
          'radius_scale': 4,
          'resolution': 20 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_1_resolution_40nm',
          'radius_scale': 1,
          'resolution': 40 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_2_resolution_40nm',
          'radius_scale': 2,
          'resolution': 40 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_3_resolution_40nm',
          'radius_scale': 3,
          'resolution': 40 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_4_resolution_40nm',
          'radius_scale': 4,
          'resolution': 40 * u.nm,
      },
  )
  def test_scattering_matrix_symmetry(
      self,
      resolution=20 * u.nm,
      radius_scale=2,
  ):
    """Test that the scattering matrix is symmetric (reciprocal) for a non-symmetric device."""
    spec = prefabs.pico_splitter_spec(design_symmetry=None)
    params = prefabs.pico_splitter_sim_params(resolution=resolution)
    model = _model.BeamSplitterModel(params, spec)

    design_var = init_design_var_feature(
        radius_scale,
        model.design_variable_shape,
    )
    s_params, _ = model.simulate(
        design_var,
        excite_port_idxs=(0, 1, 2, 3),
    )
    s = np.abs(s_params.squeeze())
    # Check overall scattering matrix symmetry
    np.testing.assert_allclose(
        s,
        s.T,
        atol=_ATOL_SCATTERING_PARAMETERS,
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'radius_scale_1_resolution_20nm',
          'radius_scale': 1,
          'resolution': 20 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_2_resolution_20nm',
          'radius_scale': 2,
          'resolution': 20 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_3_resolution_20nm',
          'radius_scale': 3,
          'resolution': 20 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_4_resolution_20nm',
          'radius_scale': 4,
          'resolution': 20 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_1_resolution_40nm',
          'radius_scale': 1,
          'resolution': 40 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_2_resolution_40nm',
          'radius_scale': 2,
          'resolution': 40 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_3_resolution_40nm',
          'radius_scale': 3,
          'resolution': 40 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_4_resolution_40nm',
          'radius_scale': 4,
          'resolution': 40 * u.nm,
      },
  )
  def test_scattering_matrix_symmetry_x(
      self,
      resolution=20 * u.nm,
      radius_scale=2,
  ):
    """Test symmetry properties of the scattering matrix for an x-symmetric device."""
    spec = prefabs.pico_splitter_spec(design_symmetry='x')
    params = prefabs.pico_splitter_sim_params(resolution=resolution)
    model = _model.BeamSplitterModel(params, spec)

    design_var = init_design_var_feature(radius_scale,
                                         model.design_variable_shape)
    s_params, _ = model.simulate(
        design_var,
        excite_port_idxs=(0, 1, 2, 3),
    )
    s = np.abs(s_params.squeeze())
    # Check overall scattering matrix symmetry
    np.testing.assert_allclose(
        s,
        s.T,
        atol=_ATOL_SCATTERING_PARAMETERS,
    )
    # Break up the scattering matrix as:
    #     S = [A. ,  B;
    #          B^T,  C]
    a = s[0:2, 0:2]
    c = s[2:4, 2:4]
    # Verify that the scattering between port 1 and port 2 is equivalent to the
    # scattering between port 4 and port 3. The flipud and fliplr are used to
    # line up the elements of the sub-matrices.
    np.testing.assert_allclose(
        a,
        np.flipud(np.fliplr(c)),
        atol=_ATOL_SCATTERING_PARAMETERS,
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'radius_scale_1_resolution_20nm',
          'radius_scale': 1,
          'resolution': 20 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_2_resolution_20nm',
          'radius_scale': 2,
          'resolution': 20 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_3_resolution_20nm',
          'radius_scale': 3,
          'resolution': 20 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_4_resolution_20nm',
          'radius_scale': 4,
          'resolution': 20 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_1_resolution_40nm',
          'radius_scale': 1,
          'resolution': 40 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_2_resolution_40nm',
          'radius_scale': 2,
          'resolution': 40 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_3_resolution_40nm',
          'radius_scale': 3,
          'resolution': 40 * u.nm,
      },
      {
          'testcase_name': 'radius_scale_4_resolution_40nm',
          'radius_scale': 4,
          'resolution': 40 * u.nm,
      },
  )
  def test_scattering_matrix_symmetry_xy(
      self,
      resolution=20 * u.nm,
      radius_scale=2,
  ):
    """Test symmetry properties of the scattering matrix for an yx-symmetric device."""
    spec = prefabs.pico_splitter_spec(design_symmetry='xy')
    params = prefabs.pico_splitter_sim_params(resolution=resolution)
    model = _model.BeamSplitterModel(params, spec)

    design_var = init_design_var_feature(radius_scale,
                                         model.design_variable_shape)
    s_params, _ = model.simulate(
        design_var,
        excite_port_idxs=(0, 1, 2, 3),
    )
    s = np.abs(s_params.squeeze())
    # Check overall scattering matrix symmetry
    np.testing.assert_allclose(
        s,
        s.T,
        atol=_ATOL_SCATTERING_PARAMETERS,
    )
    # Break up the scattering matrix as:
    #     S = [A. ,  B;
    #          B^T,  C]
    a = s[0:2, 0:2]
    b = s[0:2, 2:4]
    c = s[2:4, 2:4]
    # Verify that the scattering between port 1 and port 2 is equivalent to the
    # scattering between port 4 and port 3. The flipud and fliplr are used to
    # line up the elements of the sub-matrices.
    np.testing.assert_allclose(
        a,
        np.flipud(np.fliplr(c)),
        atol=_ATOL_SCATTERING_PARAMETERS,
    )
    # Verify that the diagonal scattering pathways are equivalent in both
    # directions:
    np.testing.assert_allclose(
        b,
        b.T,
        atol=_ATOL_SCATTERING_PARAMETERS,
    )


class ModelDesignTest(parameterized.TestCase):
  """Test various physical symmetries of the model's design region.

  These test cases consider various combinations of design symmetries (i.e. 'x',
  'xy', and None) with different design region sizes, both with even and odd
  sizes.
  """

  @parameterized.named_parameters(
      {
          'testcase_name': 'x-symmetry_11x9_separation_5',
          'design_symmetry': 'x',
          'variable_region_size': (11, 9),
          'wg_separation': 5,
      },
      {
          'testcase_name': 'xy-symmetry_11x9_separation_5',
          'design_symmetry': 'xy',
          'variable_region_size': (11, 9),
          'wg_separation': 5,
      },
      {
          'testcase_name': 'x-symmetry_9x9_separation_5',
          'design_symmetry': 'x',
          'variable_region_size': (9, 9),
          'wg_separation': 5,
      },
      {
          'testcase_name': 'xy-symmetry_9x9_separation_5',
          'design_symmetry': 'xy',
          'variable_region_size': (9, 9),
          'wg_separation': 5,
      },
      {
          'testcase_name': 'x-symmetry_8x9_separation_5',
          'design_symmetry': 'x',
          'variable_region_size': (8, 9),
          'wg_separation': 5,
      },
      {
          'testcase_name': 'xy-symmetry_8x9_separation_5',
          'design_symmetry': 'xy',
          'variable_region_size': (8, 9),
          'wg_separation': 5,
      },
      {
          'testcase_name': 'x-symmetry_8x11_separation_5',
          'design_symmetry': 'x',
          'variable_region_size': (8, 11),
          'wg_separation': 5,
      },
      {
          'testcase_name': 'xy-symmetry_8x11_separation_5',
          'design_symmetry': 'xy',
          'variable_region_size': (8, 11),
          'wg_separation': 5,
      },
      {
          'testcase_name': 'x-symmetry_8x12_separation_6',
          'design_symmetry': 'x',
          'variable_region_size': (8, 12),
          'wg_separation': 6,
      },
      {
          'testcase_name': 'xy-symmetry_8x12_separation_6',
          'design_symmetry': 'xy',
          'variable_region_size': (8, 12),
          'wg_separation': 6,
      },
      {
          'testcase_name': 'x-symmetry_7x12_separation_6',
          'design_symmetry': 'x',
          'variable_region_size': (7, 12),
          'wg_separation': 6,
      },
      {
          'testcase_name': 'xy-symmetry_7x12_separation_6',
          'design_symmetry': 'xy',
          'variable_region_size': (7, 12),
          'wg_separation': 6,
      },
      {
          'testcase_name': 'x-symmetry_9x12_separation_6',
          'design_symmetry': 'x',
          'variable_region_size': (9, 12),
          'wg_separation': 6,
      },
      {
          'testcase_name': 'xy-symmetry_9x12_separation_6',
          'design_symmetry': 'xy',
          'variable_region_size': (9, 12),
          'wg_separation': 6,
      },
  )
  def test_design_sizes(
      self,
      design_symmetry,
      variable_region_size,
      wg_separation,
  ):
    """Tests that the transform routine produces the correctly sized design."""
    dl = 40 * u.nm
    spec = prefabs.pico_splitter_spec(
        wg_width=2 * dl,
        wg_length=2 * dl,
        wg_separation=wg_separation * dl,
        wg_mode_padding=1 * dl,
        port_pml_offset=1 * dl,
        variable_region_size=(
            variable_region_size[0] * dl,
            variable_region_size[1] * dl,
        ),
        cladding_permittivity=2.25,
        slab_permittivity=12.25,
        input_monitor_offset=dl,
        design_symmetry=design_symmetry,
        pml_width=1,
    )
    params = prefabs.pico_splitter_sim_params(resolution=dl)
    model = _model.BeamSplitterModel(params, spec)
    design_var = np.ones(model.design_variable_shape)
    design_var_transformed = model.transform_design_variable(design_var)
    self.assertEqual(model.design_region_shape, design_var_transformed.shape)

  @parameterized.named_parameters(
      {
          'testcase_name': 'even_x_even_y_sym_x',
          'design_symmetry': 'x',
          'variable_region_size': (2000 * u.nm, 1440 * u.nm),
          'wg_separation': 1040 * u.nm,
      },
      {
          'testcase_name': 'even_x_even_y_sym_xy',
          'design_symmetry': 'xy',
          'variable_region_size': (2000 * u.nm, 1440 * u.nm),
          'wg_separation': 1040 * u.nm,
      },
      {
          'testcase_name': 'even_x_even_y_sym_none',
          'design_symmetry': None,
          'variable_region_size': (2000 * u.nm, 1440 * u.nm),
          'wg_separation': 1040 * u.nm,
      },
      {
          'testcase_name': 'odd_x_even_y_sym_x',
          'design_symmetry': 'x',
          'variable_region_size': (2040 * u.nm, 1440 * u.nm),
          'wg_separation': 1040 * u.nm,
      },
      {
          'testcase_name': 'odd_x_even_y_sym_xy',
          'design_symmetry': 'xy',
          'variable_region_size': (2040 * u.nm, 1440 * u.nm),
          'wg_separation': 1040 * u.nm,
      },
      {
          'testcase_name': 'odd_x_even_y_sym_none',
          'design_symmetry': None,
          'variable_region_size': (2040 * u.nm, 1440 * u.nm),
          'wg_separation': 1040 * u.nm,
      },
      {
          'testcase_name': 'even_x_odd_y_sym_x',
          'design_symmetry': 'x',
          'variable_region_size': (2000 * u.nm, 1480 * u.nm),
          'wg_separation': 1000 * u.nm,
      },
      {
          'testcase_name': 'even_x_odd_y_sym_xy',
          'design_symmetry': 'xy',
          'variable_region_size': (2000 * u.nm, 1480 * u.nm),
          'wg_separation': 1000 * u.nm,
      },
      {
          'testcase_name': 'even_x_odd_y_sym_none',
          'design_symmetry': None,
          'variable_region_size': (2000 * u.nm, 1480 * u.nm),
          'wg_separation': 1000 * u.nm,
      },
      {
          'testcase_name': 'odd_x_odd_y_sym_x',
          'design_symmetry': 'x',
          'variable_region_size': (2040 * u.nm, 1480 * u.nm),
          'wg_separation': 1000 * u.nm,
      },
      {
          'testcase_name': 'odd_x_odd_y_sym_xy',
          'design_symmetry': 'xy',
          'variable_region_size': (2040 * u.nm, 1480 * u.nm),
          'wg_separation': 1000 * u.nm,
      },
      {
          'testcase_name': 'odd_x_odd_y_sym_none',
          'design_symmetry': None,
          'variable_region_size': (2040 * u.nm, 1480 * u.nm),
          'wg_separation': 1000 * u.nm,
      },
  )
  def test_design_symmetry(
      self,
      design_symmetry,
      variable_region_size,
      wg_separation,
  ):
    """Tests that the design obeys the expected symmetry."""
    spec = prefabs.pico_splitter_spec(
        design_symmetry=design_symmetry,
        variable_region_size=variable_region_size,
        wg_separation=wg_separation,
    )
    params = prefabs.pico_splitter_sim_params(resolution=40 * u.nm)
    model = _model.BeamSplitterModel(params, spec)

    design_var = init_design_var_feature(
        2,
        model.design_variable_shape,
    )
    model_density = model.density(design_var)
    coords = model.design_region_coords
    design_density = model_density[coords[0]:coords[2], coords[1]:coords[3]]
    if design_symmetry == 'x' or design_symmetry == 'xy':
      np.testing.assert_equal(
          design_density,
          np.fliplr(design_density),
      )
    if design_symmetry == 'xy':
      np.testing.assert_equal(
          design_density,
          np.flipud(design_density),
      )
    if design_symmetry != 'x' and design_symmetry != 'xy':
      np.testing.assert_equal(
          design_density,
          design_var,
      )


if __name__ == '__main__':
  absltest.main()
