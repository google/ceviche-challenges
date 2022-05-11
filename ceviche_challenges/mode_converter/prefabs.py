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
"""Model prefabs for a waveguide mode converter, in ceviche."""

import dataclasses
from typing import Sequence

from ceviche_challenges import units as u
from ceviche_challenges import params
from ceviche_challenges.mode_converter import spec

Q = u.Quantity  # pylint: disable=invalid-name
QSequence = Sequence[Q]  # pylint: disable=invalid-name


def mode_converter_spec_12(**overrides) -> spec.ModeConverterSpec:
  """Design spec for a first-order to second-order mode converter."""
  defaults = spec.ModeConverterSpec(
      left_wg_width=200 * u.nm,
      left_wg_mode_padding=750 * u.nm,
      left_wg_mode_order=1,
      right_wg_width=400 * u.nm,
      right_wg_mode_padding=750 * u.nm,
      right_wg_mode_order=2,
      wg_length=750 * u.nm,
      padding=500 * u.nm,
      port_pml_offset=50 * u.nm,
      variable_region_size=(1500 * u.nm, 1500 * u.nm),
      cladding_permittivity=1.0,
      slab_permittivity=12.25,
      input_monitor_offset=50 * u.nm,
      pml_width=20,
  )
  return dataclasses.replace(defaults, **overrides)


def mode_converter_spec_13(**overrides) -> spec.ModeConverterSpec:
  """Design spec for a first-order to second-order mode converter."""
  defaults = spec.ModeConverterSpec(
      left_wg_width=200 * u.nm,
      left_wg_mode_padding=750 * u.nm,
      left_wg_mode_order=1,
      right_wg_width=700 * u.nm,
      right_wg_mode_padding=750 * u.nm,
      right_wg_mode_order=3,
      wg_length=750 * u.nm,
      padding=500 * u.nm,
      port_pml_offset=50 * u.nm,
      variable_region_size=(1500 * u.nm, 1500 * u.nm),
      cladding_permittivity=1.0,
      slab_permittivity=12.25,
      input_monitor_offset=50 * u.nm,
      pml_width=20,
  )
  return dataclasses.replace(defaults, **overrides)


def mode_converter_spec_23(**overrides) -> spec.ModeConverterSpec:
  """Design spec for a first-order to second-order mode converter."""
  defaults = spec.ModeConverterSpec(
      left_wg_width=400 * u.nm,
      left_wg_mode_padding=750 * u.nm,
      left_wg_mode_order=2,
      right_wg_width=700 * u.nm,
      right_wg_mode_padding=750 * u.nm,
      right_wg_mode_order=3,
      wg_length=750 * u.nm,
      padding=500 * u.nm,
      port_pml_offset=50 * u.nm,
      variable_region_size=(1500 * u.nm, 1500 * u.nm),
      cladding_permittivity=1.0,
      slab_permittivity=12.25,
      input_monitor_offset=50 * u.nm,
      pml_width=20,
  )
  return dataclasses.replace(defaults, **overrides)


def mode_converter_sim_params(**overrides) -> params.CevicheSimParams:
  """Simulation parameters appropriate for the mode converter."""
  defaults = params.CevicheSimParams(
      resolution=50 * u.nm,
      wavelengths=u.Array([1550.], u.nm),
  )
  return dataclasses.replace(defaults, **overrides)
