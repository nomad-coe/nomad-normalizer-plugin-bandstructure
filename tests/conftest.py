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
import numpy as np
import pytest
from typing import List
import json

from nomad.utils import get_logger
from nomad.datamodel import EntryArchive
from nomad.metainfo import Section, SubSection, MSection, Quantity
from nomad.normalizing import normalizers
import runschema


LOGGER = get_logger(__name__)


# TODO remove fhiiams-specifix metainfo usage from method normalizer.
# We do not want to import the electronicparsers project for this!


class x_fhi_aims_section_controlIn_basis_set(MSection):
    m_def = Section(validate=False)

    x_fhi_aims_controlIn_species_name = Quantity(
        type=str,
        shape=[],
    )


class Method(runschema.method.Method):
    m_def = Section(extends_base_section=True)

    x_fhi_aims_section_controlIn_basis_set = SubSection(
        sub_section=x_fhi_aims_section_controlIn_basis_set, repeats=True
    )


def get_template_computation() -> EntryArchive:
    """Returns a basic archive template for a computational calculation"""
    template = EntryArchive()
    run = runschema.run.Run()
    template.run.append(run)
    run.program = runschema.run.Program(name='VASP', version='4.6.35')
    system = runschema.system.System()
    run.system.append(system)
    system.atoms = runschema.system.Atoms(
        lattice_vectors=[
            [5.76372622e-10, 0.0, 0.0],
            [0.0, 5.76372622e-10, 0.0],
            [0.0, 0.0, 4.0755698899999997e-10],
        ],
        positions=[
            [2.88186311e-10, 0.0, 2.0377849449999999e-10],
            [0.0, 2.88186311e-10, 2.0377849449999999e-10],
            [0.0, 0.0, 0.0],
            [2.88186311e-10, 2.88186311e-10, 0.0],
        ],
        labels=['Br', 'K', 'Si', 'Si'],
        periodic=[True, True, True],
    )
    scc = runschema.calculation.Calculation()
    run.calculation.append(scc)
    scc.system_ref = system
    scc.energy = runschema.calculation.Energy(
        free=runschema.calculation.EnergyEntry(value=-1.5936767191492225e-18),
        total=runschema.calculation.EnergyEntry(value=-1.5935696296699573e-18),
        total_t0=runschema.calculation.EnergyEntry(value=-3.2126683561907e-22),
    )
    return template


def get_template_dft() -> EntryArchive:
    """Returns a basic archive template for a DFT calculation."""
    template = get_template_computation()
    run = template.run[-1]
    method = runschema.method.Method()
    run.method.append(method)
    method.electrons_representation = [
        runschema.method.BasisSetContainer(
            type='plane waves',
            scope=['wavefunction'],
            basis_set=[
                runschema.method.BasisSet(
                    type='plane waves',
                    scope=['valence'],
                )
            ],
        )
    ]
    method.electronic = runschema.method.Electronic(method='DFT')
    xc_functional = runschema.method.XCFunctional(
        exchange=[runschema.method.Functional(name='GGA_X_PBE')]
    )
    method.dft = runschema.method.DFT(xc_functional=xc_functional)
    scc = run.calculation[-1]
    scc.method_ref = method
    return template


def add_template_band_structure(
    template: EntryArchive,
    band_gaps: List = None,
    type: str = 'electronic',
    has_references: bool = True,
    has_reciprocal_cell: bool = True,
) -> EntryArchive:
    """Used to create a test data for band structures.

    Args:
        band_gaps: List containing the band gap value and band gap type as a
            tuple, e.g. [(1, 'direct'), (0.5, 'indirect)]. Band gap values are
            in eV. Use a list of Nones if you don't want a gap for a specific
            channel.
        type: 'electronic' or 'vibrational'
        has_references: Whether the band structure has energy references or not.
        has_reciprocal_cell: Whether the reciprocal cell is available or not.
    """

    if band_gaps is None:
        band_gaps = [None]
    if not has_reciprocal_cell:
        template.run[0].system[0].atoms = None
    scc = template.run[0].calculation[0]
    if type == 'electronic':
        bs = runschema.calculation.BandStructure()
        scc.band_structure_electronic.append(bs)
        n_spin_channels = len(band_gaps)
        fermi: List[float] = []
        highest: List[float] = []
        lowest: List[float] = []
        for gap in band_gaps:
            if gap is None:
                highest.append(0)
                lowest.append(0)
                fermi.append(0)
            else:
                fermi.append(1 * 1.60218e-19)

        if has_references:
            scc.energy = runschema.calculation.Energy(fermi=fermi[0])
            if len(highest) > 0:
                scc.energy.highest_occupied = highest[0]
            if len(lowest) > 0:
                scc.energy.lowest_unoccupied = lowest[0]
    else:
        bs = runschema.calculation.BandStructure()
        scc.band_structure_phonon.append(bs)
        n_spin_channels = 1
    n_segments = 2
    full_space = np.linspace(0, 2 * np.pi, 200)
    k, m = divmod(len(full_space), n_segments)
    space = list(
        (
            full_space[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
            for i in range(n_segments)
        )
    )
    for i_seg in range(n_segments):
        krange = space[i_seg]
        n_points = len(krange)
        seg = runschema.calculation.BandEnergies()
        bs.segment.append(seg)
        energies = np.zeros((n_spin_channels, n_points, 2))
        k_points = np.zeros((n_points, 3))
        k_points[:, 0] = np.linspace(0, 1, n_points)
        if type == 'electronic':
            for i_spin in range(n_spin_channels):
                if band_gaps[i_spin] is not None:
                    if band_gaps[i_spin][1] == 'direct':
                        energies[i_spin, :, 0] = -np.cos(krange)
                        energies[i_spin, :, 1] = np.cos(krange)
                    elif band_gaps[i_spin][1] == 'indirect':
                        energies[i_spin, :, 0] = -np.cos(krange)
                        energies[i_spin, :, 1] = np.sin(krange)
                    else:
                        raise ValueError('Invalid band gap type')
                    energies[i_spin, :, 1] += 2 + band_gaps[i_spin][0]
                else:
                    energies[i_spin, :, 0] = -np.cos(krange)
                    energies[i_spin, :, 1] = np.cos(krange)
        else:
            energies[0, :, 0] = -np.cos(krange)
            energies[0, :, 1] = np.cos(krange)
        seg.energies = energies * 1.60218e-19
        seg.kpoints = k_points
    return template


def get_template_band_structure(
    band_gaps: List = None,
    type: str = 'electronic',
    has_references: bool = True,
    has_reciprocal_cell: bool = True,
    normalize: bool = True,
) -> EntryArchive:
    archive = get_template_dft()
    archive = add_template_band_structure(
        archive, band_gaps, type, has_references, has_reciprocal_cell
    )
    if normalize:
        for normalizer in normalizers:
            normalizer(archive).normalize()
    return archive


def parse(filepath, parser_class):
    archive = EntryArchive()
    parser_class().parse(filepath, archive, LOGGER)
    for normalizer in normalizers:
        normalizer(archive).normalize()
    return archive


def load_archive(filepath: str):
    archive = EntryArchive.m_from_dict(json.load(open(filepath)))
    # TODO investigate: for some reason, ref are not updated when loading archive from file
    for calc in archive.run[0].calculation:
        calc.system_ref = calc.system_ref.m_resolved()
        calc.method_ref = calc.method_ref.m_resolved()
    for normalizer in normalizers:
        normalizer(archive).normalize()
    return archive


@pytest.fixture(scope='session')
def band_path_cP() -> EntryArchive:
    """Band structure calculation for a cP Bravais lattice."""
    return load_archive('tests/data/cP.archive.json')


@pytest.fixture(scope='session')
def band_path_cF() -> EntryArchive:
    """Band structure calculation for a cF Bravais lattice."""
    return load_archive('tests/data/cF.archive.json')


@pytest.fixture(scope='session')
def band_path_tP() -> EntryArchive:
    """Band structure calculation for a tP Bravais lattice."""
    return load_archive('tests/data/tP.archive.json')


@pytest.fixture(scope='session')
def band_path_oP() -> EntryArchive:
    """Band structure calculation for a oP Bravais lattice."""
    return load_archive('tests/data/oP.archive.json')


@pytest.fixture(scope='session')
def band_path_oF() -> EntryArchive:
    """Band structure calculation for a oF Bravais lattice."""
    return load_archive('tests/data/oF.archive.json')


@pytest.fixture(scope='session')
def band_path_hP() -> EntryArchive:
    """Band structure calculation for a hP Bravais lattice."""
    return load_archive('tests/data/hP.archive.json')


@pytest.fixture(scope='session')
def band_path_oI() -> EntryArchive:
    """Band structure calculation for a oI Bravais lattice."""
    return load_archive('tests/data/oI.archive.json')


@pytest.fixture(scope='session')
def band_path_mP() -> EntryArchive:
    """Band structure calculation for a mP Bravais lattice."""
    return load_archive('tests/data/mP.archive.json')


@pytest.fixture(scope='session')
def band_path_aP() -> EntryArchive:
    """Band structure calculation for a aP Bravais lattice."""
    return load_archive('tests/data/aP.archive.json')


@pytest.fixture(scope='session')
def band_path_cF_nonstandard() -> EntryArchive:
    """Band structure calculation for a cF Bravais lattice with non-standard k points."""
    return load_archive('tests/data/cF_nonstandard.archive.json')


@pytest.fixture(scope='session')
def band_path_cI_nonstandard() -> EntryArchive:
    """Band structure calculation for a cI Bravais lattice with non-standard k points."""
    return load_archive('tests/data/cI_nonstandard.archive.json')


@pytest.fixture(scope='session')
def band_path_tI_nonstandard() -> EntryArchive:
    """Band structure calculation for a tI Bravais lattice with non-standard k points."""
    return load_archive('tests/data/tI_nonstandard.archive.json')


@pytest.fixture(scope='session')
def band_path_oS_nonstandard() -> EntryArchive:
    """Band structure calculation for a oS Bravais lattice with non-standard k points."""
    return load_archive('tests/data/oS_nonstandard.archive.json')


@pytest.fixture(scope='session')
def band_path_hR_nonstandard() -> EntryArchive:
    """Band structure calculation for a hR Bravais lattice with non-standard k points."""
    return load_archive('tests/data/hR_nonstandard.archive.json')


@pytest.fixture(scope='session')
def band_path_mP_nonstandard() -> EntryArchive:
    """Band structure calculation for a mP Bravais lattice with a non-standard
    lattice ordering.
    """
    return load_archive('tests/data/mP_nonstandard.archive.json')


@pytest.fixture(scope='session')
def band_path_mS_nonstandard() -> EntryArchive:
    """Band structure calculation for a mS Bravais lattice with non-standard k points.
    lattice ordering.
    """
    return load_archive('tests/data/mS_nonstandard.archive.json')


@pytest.fixture(scope='session')
def phonon() -> EntryArchive:
    return load_archive('tests/data/phonon.archive.json')
