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
"""Model prefabs for beam splitter with one design region, in ceviche."""

import dataclasses
from typing import Sequence

from ceviche_challenges import params
from ceviche_challenges import units as u
from ceviche_challenges.beam_splitter import spec

Q = u.Quantity  # pylint: disable=invalid-name
QSequence = Sequence[Q]  # pylint: disable=invalid-name


def pico_splitter_spec(**overrides) -> spec.BeamSplitterSpec:
  """Design spec for the pico beam splitter with a 2.0 by 1.4 um design region."""
  defaults = spec.BeamSplitterSpec(
      wg_width=160 * u.nm,
      wg_length=400 * u.nm,
      wg_separation=1080 * u.nm,
      wg_mode_padding=480 * u.nm,
      port_pml_offset=40 * u.nm,
      variable_region_size=(2000 * u.nm, 1400 * u.nm),
      cladding_permittivity=1.0,
      slab_permittivity=12.25,
      input_monitor_offset=40 * u.nm,
      design_symmetry=None,
      pml_width=20,
  )
  return dataclasses.replace(defaults, **overrides)


def pico_splitter_sim_params(**overrides) -> params.CevicheSimParams:
  """Simulation parameters appropriate for the pico beam splitter."""
  defaults = params.CevicheSimParams(
      resolution=40 * u.nm,
      wavelengths=u.Array([1550.], u.nm),
  )
  return dataclasses.replace(defaults, **overrides)
