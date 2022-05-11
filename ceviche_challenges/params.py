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
"""Generial simulation parameters for ceviche models."""

import dataclasses
from typing import Sequence

from ceviche_challenges import units as u

Q = u.Quantity  # pylint: disable=invalid-name
QSequence = Sequence[Q]  # pylint: disable=invalid-name


@dataclasses.dataclass(frozen=True)
class CevicheSimParams:
  """Parameters for a Ceviche simulation.

  Attributes:
    resolution: the spatial resolution of the simulation grid, i.e. the side
      length of a grid cell
    wavelengths: a list of the wavelengths that will be simulated
  """
  resolution: Q
  wavelengths: QSequence
