#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from nomad.config.models.plugins import NormalizerEntryPoint


class BandStructureNormalizerEntryPoint(NormalizerEntryPoint):
    def load(self):
        import bandstructurenormalizer
        from .normalizer import BandStructureNormalizer

        bandstructurenormalizer.BandStructureNormalizer = BandStructureNormalizer

        return BandStructureNormalizer(**self.dict())


bandstructure_normalizer_entry_point = BandStructureNormalizerEntryPoint(
    name='BandStructureNormalizer',
    description='Normalizer for the band structure data.',
)
