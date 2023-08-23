# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
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
Orchestrating the BOLD-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_func_preproc_wf
.. autofunction:: init_func_derivatives_wf

"""
import os
import typing as ty

import bids
import nibabel as nb
import numpy as np
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.func.util import init_enhance_and_skullstrip_bold_wf
from niworkflows.interfaces.header import ValidateImage
from niworkflows.interfaces.nitransforms import ConcatenateXFMs
from niworkflows.utils.connections import listify
from sdcflows.workflows.apply.correction import init_unwarp_wf
from sdcflows.workflows.apply.registration import init_coeff2epi_wf

from ... import config

# BOLD workflows
from .hmc import init_bold_hmc_wf
from .outputs import init_ds_boldref_wf, init_ds_hmc_wf, init_ds_registration_wf
from .reference import init_raw_boldref_wf
from .registration import init_bold_reg_wf


def get_sbrefs(
    bold_files: ty.List[str],
    entity_overrides: ty.Dict[str, ty.Any],
    layout: bids.BIDSLayout,
) -> ty.List[str]:
    """Find single-band reference(s) associated with BOLD file(s)

    Parameters
    ----------
    bold_files
        List of absolute paths to BOLD files
    entity_overrides
        Query parameters to override defaults
    layout
        :class:`~bids.layout.BIDSLayout` to query

    Returns
    -------
    sbref_files
        List of absolute paths to sbref files associated with input BOLD files,
        sorted by EchoTime
    """
    entities = extract_entities(bold_files)
    entities.pop("echo", None)
    entities.update(suffix="sbref", extension=[".nii", ".nii.gz"], **entity_overrides)

    return sorted(
        layout.get(return_type="file", **entities),
        key=lambda fname: layout.get_metadata(fname).get("EchoTime"),
    )


def init_bold_fit_wf(
    *,
    bold_series: ty.Union[str, ty.List[str]],
    precomputed: dict,
    fieldmap_id: ty.Optional[str] = None,
    omp_nthreads: int = 1,
) -> pe.Workflow:
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.utility import KeySelect

    layout = config.execution.layout

    # Collect bold and sbref files, sorted by EchoTime
    bold_files = sorted(
        listify(bold_series),
        key=lambda fname: layout.get_metadata(fname).get("EchoTime"),
    )
    sbref_files = get_sbrefs(
        bold_files,
        entity_overrides=config.execution.get().get('bids_filters', {}).get('sbref', {}),
        layout=layout,
    )

    # Fitting operates on the shortest echo
    # This could become more complicated in the future
    bold_file = bold_files[0]

    if os.path.isfile(bold_file):
        bold_tlen, mem_gb = _create_mem_gb(bold_file)

    # Boolean used to update workflow self-descriptions
    multiecho = len(bold_files) > 1

    have_hmcref = "hmc_boldref" in precomputed
    have_hmc = "hmc_xforms" in precomputed
    have_coregref = "coreg_boldref" in precomputed
    # Can contain
    #  1) boldref2fmap
    #  2) boldref2anat
    #  3) hmc
    transforms = precomputed.get("transforms", {})
    # have_regref = "coreg_boldref" in precomputed
    # XXX This may need a better name
    coreg_xfm = precomputed.get("transforms", {}).get("boldref", {})

    workflow = Workflow(name=_get_wf_name(bold_file))

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_file",
                # Fieldmap registration
                "fmap",
                "fmap_ref",
                "fmap_coeff",
                "fmap_mask",
                "fmap_id",
                "sdc_method",
                # Anatomical coregistration
                "t1w_preproc",
                "t1w_mask",
                "t1w_dseg",
                "subjects_dir",
                "subject_id",
                "fsnative2t1w_xfm",
            ],
        ),
        name="inputnode",
    )
    inputnode.inputs.bold_file = bold_series

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "hmc_boldref",
                "coreg_boldref",
                "bold_mask",
                "motion_xfm",
                "boldref2anat_xfm",
            ],
        ),
        name="outputnode",
    )

    # If all derivatives exist, inputnode could go unconnected, so add explicitly
    workflow.add_nodes([inputnode])

    hmcref_buffer = pe.Node(
        niu.IdentityInterface(fields=["boldref", "bold_file"]), name="hmcref_buffer"
    )
    fmapref_buffer = pe.Node(niu.Function(function=_select_ref), name="fmapref_buffer")
    hmc_buffer = pe.Node(niu.IdentityInterface(fields=["hmc_xforms"]), name="hmc_buffer")
    fmapreg_buffer = pe.Node(
        niu.IdentityInterface(fields=["boldref2fmap_xform"]), name="fmapreg_buffer"
    )
    regref_buffer = pe.Node(
        niu.IdentityInterface(fields=["boldref", "boldmask"]), name="regref_buffer"
    )

    # fmt:off
    workflow.connect([
        (hmcref_buffer, outputnode, [("boldref", "hmc_boldref")]),
        (regref_buffer, outputnode, [("boldref", "coreg_boldref"),
                                     ("boldmask", "bold_mask")]),
        (hmc_buffer, outputnode, [("hmc_xforms", "motion_xfm")]),
    ])
    # fmt:on

    # Stage 1: Generate motion correction boldref
    if not have_hmcref:
        config.loggers.workflow.info("Stage 1: Adding HMC boldref workflow")
        hmc_boldref_wf = init_raw_boldref_wf(
            name="hmc_boldref_wf",
            bold_file=bold_file,
            multiecho=multiecho,
        )
        hmc_boldref_wf.inputs.inputnode.dummy_scans = config.workflow.dummy_scans

        ds_hmc_boldref_wf = init_ds_boldref_wf(
            bids_root=layout.root,
            output_dir=config.execution.output_dir,
            desc='hmc',
            name='ds_hmc_boldref_wf',
        )
        ds_hmc_boldref_wf.inputs.inputnode.source_files = [bold_file]

        # fmt:off
        workflow.connect([
            (hmc_boldref_wf, hmcref_buffer, [("outputnode.bold_file", "bold_file")]),
            (hmc_boldref_wf, ds_hmc_boldref_wf, [("outputnode.boldref", "inputnode.boldref")]),
            (ds_hmc_boldref_wf, hmcref_buffer, [("outputnode.boldref", "boldref")]),
        ])
        # fmt:on
    else:
        config.loggers.workflow.info("Found HMC boldref - skipping Stage 1")

        validate_bold = pe.Node(ValidateImage(), name="validate_bold")
        validate_bold.inputs.in_file = bold_file

        workflow.connect(validate_bold, "out_file", hmcref_buffer, "bold_file")
        hmcref_buffer.inputs.boldref = precomputed["hmc_boldref"]

    # Stage 2: Estimate head motion
    if not have_hmc:
        config.loggers.workflow.info("Stage 2: Adding motion correction workflow")
        bold_hmc_wf = init_bold_hmc_wf(
            name="bold_hmc_wf", mem_gb=mem_gb["filesize"], omp_nthreads=omp_nthreads
        )

        ds_hmc_wf = init_ds_hmc_wf(
            bids_root=layout.root,
            output_dir=config.execution.output_dir,
        )
        ds_hmc_wf.inputs.inputnode.source_files = [bold_file]

        # fmt:off
        workflow.connect([
            (hmcref_buffer, bold_hmc_wf, [
                ("boldref", "inputnode.raw_ref_image"),
                ("bold_file", "inputnode.bold_file"),
            ]),
            (bold_hmc_wf, ds_hmc_wf, [("outputnode.xforms", "inputnode.xforms")]),
            (ds_hmc_wf, hmc_buffer, [("outputnode.xforms", "hmc_xforms")]),
        ])
        # fmt:on
    else:
        config.loggers.workflow.info("Found motion correction transforms - skipping Stage 2")
        hmc_buffer.inputs.hmc_xforms = precomputed["hmc_xforms"]

    # Stage 3: Create coregistration reference
    # Fieldmap correction only happens during fit if this stage is needed
    if not have_coregref:
        config.loggers.workflow.info("Stage 3: Adding coregistration boldref workflow")

        # Select initial boldref, enhance contrast, and generate mask
        fmapref_buffer.inputs.sbref_files = sbref_files
        enhance_boldref_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads)

        ds_coreg_boldref_wf = init_ds_boldref_wf(
            bids_root=layout.root,
            output_dir=config.execution.output_dir,
            desc='coreg',
            name='ds_coreg_boldref_wf',
        )

        # fmt:off
        workflow.connect([
            (hmcref_buffer, fmapref_buffer, [("boldref", "boldref_files")]),
            (fmapref_buffer, enhance_boldref_wf, [("out", "inputnode.in_file")]),
            (fmapref_buffer, ds_coreg_boldref_wf, [("out", "inputnode.source_files")]),
            (ds_coreg_boldref_wf, regref_buffer, [("outputnode.boldref", "boldref")]),
        ])
        # fmt:on

        if fieldmap_id:
            coeff2epi_wf = init_coeff2epi_wf(
                debug="fieldmaps" in config.execution.debug,
                omp_nthreads=config.nipype.omp_nthreads,
                sloppy=config.execution.sloppy,
            )
            unwarp_wf = init_unwarp_wf(
                free_mem=config.environment.free_mem,
                debug="fieldmaps" in config.execution.debug,
                omp_nthreads=config.nipype.omp_nthreads,
            )
            unwarp_wf.inputs.inputnode.metadata = layout.get_metadata(bold_file)

            fmap_select = pe.Node(
                KeySelect(
                    fields=["fmap", "fmap_ref", "fmap_coeff", "fmap_mask", "sdc_method"],
                    key=fieldmap_id,
                ),
                name="fmap_select",
                run_without_submitting=True,
            )

            itk_mat2txt = pe.Node(ConcatenateXFMs(out_fmt="itk"), name="itk_text")

            ds_fmapreg_wf = init_ds_registration_wf(
                bids_root=layout.root,
                output_dir=config.execution.output_dir,
                source="boldref",
                dest=fieldmap_id.replace('_', ''),
                name="ds_fmapreg_wf",
            )

            # fmt:off
            workflow.connect([
                (inputnode, fmap_select, [
                    ("fmap", "fmap"),
                    ("fmap_ref", "fmap_ref"),
                    ("fmap_coeff", "fmap_coeff"),
                    ("fmap_mask", "fmap_mask"),
                    ("sdc_method", "sdc_method"),
                    ("fmap_id", "keys"),
                ]),
                (fmap_select, coeff2epi_wf, [
                    ("fmap_ref", "inputnode.fmap_ref"),
                    ("fmap_coeff", "inputnode.fmap_coeff"),
                    ("fmap_mask", "inputnode.fmap_mask"),
                ]),
                (coeff2epi_wf, itk_mat2txt, [('outputnode.target2fmap_xfm', 'in_xfms')]),
                (itk_mat2txt, ds_fmapreg_wf, [('out_xfm', 'inputnode.xform')]),
                # XXX Incomplete
                (fmapref_buffer, ds_fmapreg_wf, [('out', 'inputnode.source_files')]),
                (ds_fmapreg_wf, fmapreg_buffer, [('outputnode.xform', 'boldref2fmap_xfm')]),
                (enhance_boldref_wf, coeff2epi_wf, [
                    ('outputnode.bias_corrected_file', 'inputnode.target_ref'),
                    ('outputnode.mask_file', 'inputnode.target_mask'),
                ]),
                (coeff2epi_wf, unwarp_wf, [
                    ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
                ]),
                (enhance_boldref_wf, unwarp_wf, [
                    ('outputnode.bias_corrected_file', 'inputnode.distorted'),
                ]),
                (unwarp_wf, ds_coreg_boldref_wf, [
                    ('outputnode.corrected', 'inputnode.boldref'),
                ]),
                (unwarp_wf, regref_buffer, [
                    ('outputnode.corrected_mask', 'boldmask'),
                ]),
            ])
            # fmt:on
        else:
            # fmt:off
            workflow.connect([
                (enhance_boldref_wf, ds_coreg_boldref_wf, [
                    ('outputnode.bias_corrected_file', 'inputnode.boldref'),
                ]),
                (enhance_boldref_wf, regref_buffer, [
                    ('outputnode.mask_file', 'boldmask'),
                ]),
            ])
            # fmt:on
    else:
        config.loggers.workflow.info("Found coregistration reference - skipping Stage 3")
        regref_buffer.inputs.boldref = precomputed["coreg_boldref"]

    if "boldref2anat" not in transforms:
        # calculate BOLD registration to T1w
        bold_reg_wf = init_bold_reg_wf(
            bold2t1w_dof=config.workflow.bold2t1w_dof,
            bold2t1w_init=config.workflow.bold2t1w_init,
            freesurfer=config.workflow.run_reconall,
            mem_gb=mem_gb["resampled"],
            name="bold_reg_wf",
            omp_nthreads=omp_nthreads,
            sloppy=config.execution.sloppy,
            use_bbr=config.workflow.use_bbr,
            use_compression=False,
            write_report=False,
        )

        ds_boldreg_wf = init_ds_registration_wf(
            bids_root=layout.root,
            output_dir=config.execution.output_dir,
            source="boldref",
            dest="T1w",
            name="ds_boldreg_wf",
        )

        # fmt:off
        workflow.connect([
            (inputnode, bold_reg_wf, [
                ("t1w_dseg", "inputnode.t1w_dseg"),
                # Undefined if --fs-no-reconall, but this is safe
                ("subjects_dir", "inputnode.subjects_dir"),
                ("subject_id", "inputnode.subject_id"),
                ("fsnative2t1w_xfm", "inputnode.fsnative2t1w_xfm"),
            ]),
            (regref_buffer, bold_reg_wf, [("boldref", "inputnode.ref_bold_brain")]),
            # Incomplete sources
            (regref_buffer, ds_boldreg_wf, [("boldref", "inputnode.source_files")]),
            (bold_reg_wf, ds_boldreg_wf, [("outputnode.itk_bold_to_t1", "inputnode.xform")]),
            (ds_boldreg_wf, outputnode, [("outputnode.xform", "boldref2anat_xfm")]),
        ])
        # fmt:on

    return workflow


def _create_mem_gb(bold_fname):
    img = nb.load(bold_fname)
    nvox = int(np.prod(img.shape, dtype='u8'))
    # Assume tools will coerce to 8-byte floats to be safe
    bold_size_gb = 8 * nvox / (1024**3)
    bold_tlen = img.shape[-1]
    mem_gb = {
        "filesize": bold_size_gb,
        "resampled": bold_size_gb * 4,
        "largemem": bold_size_gb * (max(bold_tlen / 100, 1.0) + 4),
    }

    return bold_tlen, mem_gb


def _get_wf_name(bold_fname):
    """
    Derive the workflow name for supplied BOLD file.

    >>> _get_wf_name("/completely/made/up/path/sub-01_task-nback_bold.nii.gz")
    'func_preproc_task_nback_wf'
    >>> _get_wf_name("/completely/made/up/path/sub-01_task-nback_run-01_echo-1_bold.nii.gz")
    'func_preproc_task_nback_run_01_echo_1_wf'

    """
    from nipype.utils.filemanip import split_filename

    fname = split_filename(bold_fname)[1]
    fname_nosub = "_".join(fname.split("_")[1:])
    name = "func_preproc_" + fname_nosub.replace(".", "_").replace(" ", "").replace(
        "-", "_"
    ).replace("_bold", "_wf")

    return name


def _to_join(in_file, join_file):
    """Join two tsv files if the join_file is not ``None``."""
    from niworkflows.interfaces.utility import JoinTSVColumns

    if join_file is None:
        return in_file
    res = JoinTSVColumns(in_file=in_file, join_file=join_file).run()
    return res.outputs.out_file


def extract_entities(file_list):
    """
    Return a dictionary of common entities given a list of files.

    Examples
    --------
    >>> extract_entities("sub-01/anat/sub-01_T1w.nii.gz")
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    >>> extract_entities(["sub-01/anat/sub-01_T1w.nii.gz"] * 2)
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    >>> extract_entities(["sub-01/anat/sub-01_run-1_T1w.nii.gz",
    ...                   "sub-01/anat/sub-01_run-2_T1w.nii.gz"])
    {'subject': '01', 'run': [1, 2], 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}

    """
    from collections import defaultdict

    from bids.layout import parse_file_entities

    entities = defaultdict(list)
    for e, v in [
        ev_pair for f in listify(file_list) for ev_pair in parse_file_entities(f).items()
    ]:
        entities[e].append(v)

    def _unique(inlist):
        inlist = sorted(set(inlist))
        if len(inlist) == 1:
            return inlist[0]
        return inlist

    return {k: _unique(v) for k, v in entities.items()}


def get_img_orientation(imgf):
    """Return the image orientation as a string"""
    img = nb.load(imgf)
    return "".join(nb.aff2axcodes(img.affine))


def get_estimator(layout, fname):
    field_source = layout.get_metadata(fname).get("B0FieldSource")
    if isinstance(field_source, str):
        field_source = (field_source,)

    if field_source is None:
        import re
        from pathlib import Path

        from sdcflows.fieldmaps import get_identifier

        # Fallback to IntendedFor
        intended_rel = re.sub(r"^sub-[a-zA-Z0-9]*/", "", str(Path(fname).relative_to(layout.root)))
        field_source = get_identifier(intended_rel)

    return field_source


def _select_ref(sbref_files, boldref_files):
    """Select first sbref or boldref file, preferring sbref if available"""
    from niworkflows.utils.connections import listify

    refs = sbref_files or boldref_files
    return listify(refs)[0]
