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
NORDIC denoising of BOLD images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_nordic_wf

"""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from ... import config

LOGGER = config.loggers.workflow


def init_bold_nordic_wf(
    *,
    mem_gb: dict,
    phase: bool = False,
    name='bold_nordic_wf',
):
    """Create a workflow for NORDIC denoising.

    This workflow applies NORDIC to the input
    :abbr:`BOLD (blood-oxygen-level dependent)` image.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.bold import init_bold_nordic_wf
            wf = init_bold_nordic_wf(
                phase=True,
                mem_gb={'filesize': 1},
            )

    Parameters
    ----------
    phase : :obj:`bool`
        True if phase data is available. False if not.
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
    nordic_file
        NORDIC-denoised BOLD series NIfTI file
    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
NORDIC was applied to the BOLD data.
"""
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['bold_file', 'norf_file', 'phase_file', 'phase_norf_file'],
        ),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['nordic_file']), name='outputnode')

    # Add noRF file to end of bold_file if available
    add_noise = pe.Node(niu.IdentityInterface(fields=['bold_file']), name='add_noise')

    if phase:
        # Do the same for the phase data if available
        add_phase_noise = pe.Node(
            niu.IdentityInterface(fields=['bold_file', 'norf_file']),
            name='add_phase_noise',
        )
        validate_complex = pe.Node(
            niu.IdentityInterface(
                fields=['mag_file', 'phase_file', 'n_mag_noise_volumes', 'n_phase_noise_volumes'],
            ),
            name='validate_complex',
        )

    # Run NORDIC
    nordic = pe.Node(
        niu.IdentityInterface(fields=['mag_file', 'phase_file']),
        mem_gb=mem_gb['filesize'] * 2,
        name='nordic',
    )

    # Split noise volumes out of denoised bold_file
    split_noise = pe.Node(
        niu.IdentityInterface(fields=['bold_file', 'n_noise_volumes']),
        name='split_noise',
    )

    workflow.connect([
        (inputnode, add_noise, [('bold_file', 'in_file')]),
        (nordic, split_noise, [('mag_file', 'bold_file')]),
        (split_noise, outputnode, [('bold_file', 'nordic_file')]),
    ])  # fmt:skip

    if phase:
        workflow.connect([
            (inputnode, add_phase_noise, [
                ('phase_file', 'bold_file'),
                ('phase_norf_file', 'norf_file'),
            ]),
            (add_noise, validate_complex, [
                ('bold_file', 'mag_file'),
                ('n_noise_volumes', 'n_mag_noise_volumes'),
            ]),
            (add_phase_noise, validate_complex, [
                ('bold_file', 'phase_file'),
                ('n_noise_volumes', 'n_phase_noise_volumes'),
            ]),
            (validate_complex, nordic, [
                ('mag_file', 'mag_file'),
                ('phase_file', 'phase_file'),
                ('n_noise_volumes', 'n_noise_volumes'),
            ]),
            (validate_complex, split_noise, [('n_noise_volumes', 'n_noise_volumes')]),
        ])  # fmt:skip
    else:
        workflow.connect([
            (add_noise, nordic, [
                ('bold_file', 'mag_file'),
                ('n_noise_volumes', 'n_noise_volumes'),
            ]),
            (add_noise, split_noise, [('n_noise_volumes', 'n_noise_volumes')]),
        ])

    return workflow
