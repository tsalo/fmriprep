# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
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
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Denoising of BOLD images
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_dwidenoise_wf

"""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from ... import config
from ...interfaces.denoise import (
    ComplexToMagnitude,
    ComplexToPhase,
    DWIDenoise,
    NoiseEstimate,
    PhaseToRad,
    PolarToComplex,
    ValidateComplex,
)

LOGGER = config.loggers.workflow


def init_bold_dwidenoise_wf(
    *,
    mem_gb: dict,
    has_phase: bool = False,
    has_norf: bool = False,
    name='bold_dwidenoise_wf',
):
    """Create a workflow for the removal of thermal noise with dwidenoise.

    This workflow applies MP-PCA or NORDIC to the input
    :abbr:`BOLD (blood-oxygen-level dependent)` image.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.bold import init_bold_dwidenoise_wf
            wf = init_bold_dwidenoise_wf(
                has_phase=True,
                has_norf=True,
                mem_gb={'filesize': 1},
            )

    Parameters
    ----------
    has_phase : :obj:`bool`
        True if phase data is available. False if not.
    has_norf : :obj:`bool`
        True if noRF data is available. False if not.
    metadata : :obj:`dict`
        BIDS metadata for BOLD file
    name : :obj:`str`
        Name of workflow (default: ``bold_stc_wf``)

    Inputs
    ------
    bold_file
        BOLD series NIfTI file

    Outputs
    -------
    mag_file
        Denoised BOLD series NIfTI file
    phase_file
        Denoised phase series NIfTI file
    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
NORDIC or MP-PCA was applied to the BOLD data.
"""
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['mag_file', 'norf_file', 'phase_file', 'phase_norf_file'],
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['mag_file', 'phase_file'],
        ),
        name='outputnode',
    )

    if has_norf:
        # Calculate noise map from noise volumes
        # TODO: Figure out how to estimate the noise map from the noRF data
        noise_estimate = pe.Node(
            NoiseEstimate(),
            name='noise_estimate',
            mem_gb=mem_gb['filesize'],
        )
        if has_phase:
            validate_complex_norf = pe.Node(ValidateComplex(), name='validate_complex_norf')
            workflow.connect([
                (inputnode, validate_complex_norf, [
                    ('mag_norf_file', 'mag_file'),
                    ('phase_norf_file', 'phase_file'),
                ]),
            ])  # fmt:skip

            # Combine magnitude and phase data into complex-valued data
            phase_to_radians_norf = pe.Node(
                PhaseToRad(),
                name='phase_to_radians_norf',
            )
            workflow.connect([
                (validate_complex_norf, phase_to_radians_norf, [('phase_file', 'phase_file')]),
            ])  # fmt:skip

            combine_complex_norf = pe.Node(
                PolarToComplex(),
                name='combine_complex_norf',
            )
            workflow.connect([
                (validate_complex_norf, combine_complex_norf, [('mag_file', 'mag_file')]),
                (phase_to_radians_norf, combine_complex_norf, [('phase_file', 'phase_file')]),
                (combine_complex_norf, noise_estimate, [('out_file', 'in_file')]),
            ])  # fmt:skip

        else:
            workflow.connect([(inputnode, noise_estimate, [('mag_norf_file', 'in_file')])])

    complex_buffer = pe.Node(niu.IdentityInterface(fields=['bold_file']), name='complex_buffer')
    if has_phase:
        validate_complex = pe.Node(ValidateComplex(), name='validate_complex')

        # Combine magnitude and phase data into complex-valued data
        phase_to_radians = pe.Node(
            PhaseToRad(),
            name='phase_to_radians',
        )
        workflow.connect([(validate_complex, phase_to_radians, [('phase_file', 'phase_file')])])

        combine_complex = pe.Node(
            PolarToComplex(),
            name='combine_complex',
        )
        workflow.connect([
            (validate_complex, combine_complex, [('mag_file', 'mag_file')]),
            (phase_to_radians, combine_complex, [('phase_file', 'phase_file')]),
            (combine_complex, complex_buffer, [('out_file', 'bold_file')]),
        ])  # fmt:skip
    else:
        workflow.connect([(inputnode, complex_buffer, [('mag_file', 'bold_file')])])

    # Run NORDIC
    dwidenoise = pe.Node(
        DWIDenoise(),
        mem_gb=mem_gb['filesize'] * 2,
        name='dwidenoise',
    )
    workflow.connect([(complex_buffer, dwidenoise, [('bold_file', 'in_file')])])

    if has_norf:
        workflow.connect([(noise_estimate, dwidenoise, [('noise_map', 'noise_map')])])

    if has_phase:
        # Split the denoised complex-valued data into magnitude and phase
        split_magnitude = pe.Node(
            ComplexToMagnitude(),
            name='split_complex',
        )
        workflow.connect([
            (dwidenoise, split_magnitude, [('out_file', 'complex_file')]),
            (split_magnitude, outputnode, [('out_file', 'mag_file')]),
        ])  # fmt:skip

        split_phase = pe.Node(
            ComplexToPhase(),
            name='split_phase',
        )
        workflow.connect([
            (dwidenoise, split_phase, [('out_file', 'complex_file')]),
            (split_phase, outputnode, [('out_file', 'phase_file')]),
        ])  # fmt:skip
    else:
        workflow.connect([(dwidenoise, outputnode, [('out_file', 'mag_file')])])

    return workflow