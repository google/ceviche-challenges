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
"""Model prefabs for a waveguide bend, in ceviche."""

import dataclasses
from typing import Sequence

from ceviche_challenges import params
from ceviche_challenges import units as u
from ceviche_challenges.waveguide_bend import spec

Q = u.Quantity  # pylint: disable=invalid-name
QSequence = Sequence[Q]  # pylint: disable=invalid-name


def waveguide_bend_1umx1um_spec(**overrides) -> spec.WaveguideBendSpec:
  """Design spec for a waveguide bend with a 1 um x 1 um design region."""
  defaults = spec.WaveguideBendSpec(
      wg_width=200 * u.nm,
      wg_length=750 * u.nm,
      wg_mode_padding=750 * u.nm,
      padding=400 * u.nm,
      port_pml_offset=50 * u.nm,
      variable_region_size=(1000 * u.nm, 1000 * u.nm),
      cladding_permittivity=1.0,
      slab_permittivity=12.25,
      input_monitor_offset=50 * u.nm,
      pml_width=20,
  )
  return dataclasses.replace(defaults, **overrides)


def waveguide_bend_2umx2um_spec(**overrides) -> spec.WaveguideBendSpec:
  """Design spec for a waveguide bend with a 2 um x 2 um design region."""
  defaults = spec.WaveguideBendSpec(
      wg_width=200 * u.nm,
      wg_length=750 * u.nm,
      wg_mode_padding=750 * u.nm,
      padding=400 * u.nm,
      port_pml_offset=50 * u.nm,
      variable_region_size=(2000 * u.nm, 2000 * u.nm),
      cladding_permittivity=1.0,
      slab_permittivity=12.25,
      input_monitor_offset=50 * u.nm,
      pml_width=20,
  )
  return dataclasses.replace(defaults, **overrides)


def waveguide_bend_3umx3um_spec(**overrides) -> spec.WaveguideBendSpec:
  """Design spec for a waveguide bend with a 3 um x 3 um design region."""
  defaults = spec.WaveguideBendSpec(
      wg_width=200 * u.nm,
      wg_length=750 * u.nm,
      wg_mode_padding=750 * u.nm,
      padding=400 * u.nm,
      port_pml_offset=50 * u.nm,
      variable_region_size=(3000 * u.nm, 3000 * u.nm),
      cladding_permittivity=1.0,
      slab_permittivity=12.25,
      input_monitor_offset=50 * u.nm,
      pml_width=20,
  )
  return dataclasses.replace(defaults, **overrides)


def waveguide_bend_sim_params(**overrides) -> params.CevicheSimParams:
  """Simulation parameters appropriate for the waveguide bend."""
  defaults = params.CevicheSimParams(
      resolution=50 * u.nm,
      wavelengths=u.Array([1550.], u.nm),
  )
  return dataclasses.replace(defaults, **overrides)
