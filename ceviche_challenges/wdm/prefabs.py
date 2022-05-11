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
"""Model prefabs for a WDM, in ceviche."""

import dataclasses

from ceviche_challenges import params
from ceviche_challenges import units as u
from ceviche_challenges.wdm import spec as _spec


def wdm_spec(**overrides) -> _spec.WdmSpec:
  """Design spec for a WDM."""
  spec = _spec.WdmSpec(
      extent_ij=u.Array([5120, 5120], u.nm),
      input_wg_j=2560 * u.nm,
      output_wgs_j=u.Array([1800, 3320], u.nm),
      wg_width=400 * u.nm,
      wg_mode_padding=400 * u.nm,
      input_mode_i=440 * u.nm,
      output_mode_i=4680 * u.nm,
      variable_region=(u.Array([1000, 960], u.nm), u.Array([4200, 4160], u.nm)),
      cladding_permittivity=2.25,
      slab_permittivity=12.25,
      input_monitor_offset=40 * u.nm,
      pml_width=10,
  )
  return dataclasses.replace(spec, **overrides)


def wdm_sim_params(**overrides) -> params.CevicheSimParams:
  """Simulation parameters appropriate for the WDM."""
  defaults = params.CevicheSimParams(
      resolution=40 * u.nm,
      wavelengths=u.Array([1270, 1290], u.nm),
  )
  return dataclasses.replace(defaults, **overrides)
