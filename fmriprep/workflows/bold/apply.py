from __future__ import annotations

import os
import typing as ty

import nibabel as nb
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from niworkflows.interfaces.header import ValidateImage
from niworkflows.interfaces.utility import KeySelect
from niworkflows.utils.connections import listify

from fmriprep import config

from ...interfaces.resampling import (
    DistortionParameters,
    ReconstructFieldmap,
    ResampleSeries,
)
from ...utils.misc import estimate_bold_mem_usage
from .stc import init_bold_stc_wf
from .t2s import init_bold_t2s_wf, init_t2s_reporting_wf

if ty.TYPE_CHECKING:
    from niworkflows.utils.spaces import SpatialReferences


def init_bold_volumetric_resample_wf(
    *,
    metadata: dict,
    fieldmap_id: ty.Optional[str] = None,
    omp_nthreads: int = 1,
    name: str = 'bold_volumetric_resample_wf',
) -> pe.Workflow:
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_file",
                "ref_file",
                # HMC
                "motion_xfm",
                # SDC
                "boldref2fmap_xfm",
                "fmap_ref",
                "fmap_coeff",
                "fmap_id",
                # Anatomical
                "boldref2anat_xfm",
                # Template
                "anat2std_xfm",
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(niu.IdentityInterface(fields=["bold_file"]), name='outputnode')

    boldref2target = pe.Node(niu.Merge(2), name='boldref2target')
    bold2target = pe.Node(niu.Merge(2), name='bold2target')
    resample = pe.Node(ResampleSeries(), name="resample", n_procs=omp_nthreads)

    workflow.connect([
        (inputnode, boldref2target, [
            ('boldref2anat_xfm', 'in1'),
            ('anat2std_xfm', 'in2'),
        ]),
        (inputnode, bold2target, [('motion_xfm', 'in1')]),
        (inputnode, resample, [
            ('bold_file', 'in_file'),
            ('ref_file', 'ref_file'),
        ]),
        (boldref2target, bold2target, [('out', 'in2')]),
        (bold2target, resample, [('out', 'transforms')]),
        (resample, outputnode, [('out_file', 'bold_file')]),
    ])  # fmt:skip

    if fieldmap_id:
        fmap_select = pe.Node(
            KeySelect(fields=["fmap_ref", "fmap_coeff"], key=fieldmap_id),
            name="fmap_select",
            run_without_submitting=True,
        )
        distortion_params = pe.Node(
            DistortionParameters(metadata=metadata),
            name="distortion_params",
            run_without_submitting=True,
        )
        fmap2target = pe.Node(niu.Merge(2), name='fmap2target')
        inverses = pe.Node(niu.Function(function=_gen_inverses), name='inverses')

        fmap_recon = pe.Node(ReconstructFieldmap(), name="fmap_recon")

        workflow.connect([
            (inputnode, fmap_select, [
                ("fmap_ref", "fmap_ref"),
                ("fmap_coeff", "fmap_coeff"),
                ("fmap_id", "keys"),
            ]),
            (inputnode, distortion_params, [('bold_file', 'in_file')]),
            (inputnode, fmap2target, [('boldref2fmap_xfm', 'in1')]),
            (boldref2target, fmap2target, [('out', 'in2')]),
            (boldref2target, inverses, [('out', 'inlist')]),
            (fmap_select, fmap_recon, [
                ("fmap_coeff", "in_coeffs"),
                ("fmap_ref", "fmap_ref_file"),
            ]),
            (fmap2target, fmap_recon, [('out', 'transforms')]),
            (inverses, fmap_recon, [('out', 'inverse')]),
            # Inject fieldmap correction into resample node
            (distortion_params, resample, [
                ("readout_time", "ro_time"),
                ("pe_direction", "pe_dir"),
            ]),
            (fmap_recon, resample, [('out_file', 'fieldmap')]),
        ])  # fmt:skip

    return workflow


def init_bold_apply_wf(
    *,
    spaces: SpatialReferences,
    name: str = 'bold_apply_wf',
) -> pe.Workflow:
    """TODO: Docstring"""
    from smriprep.workflows.outputs import init_template_iterator_wf

    workflow = pe.Workflow(name=name)

    if spaces.is_cached() and spaces.cached.references:
        template_iterator_wf = init_template_iterator_wf(spaces=spaces)
        # TODO: Refactor bold_std_trans_wf
        # bold_std_trans_wf = init_bold_std_trans_wf(
        #     freesurfer=config.workflow.run_reconall,
        #     mem_gb=mem_gb["resampled"],
        #     omp_nthreads=config.nipype.omp_nthreads,
        #     spaces=spaces,
        #     multiecho=multiecho,
        #     use_compression=not config.execution.low_mem,
        #     name="bold_std_trans_wf",
        # )

    return workflow


def _gen_inverses(inlist: list) -> list[bool]:
    """Create a list indicating the first transform should be inverted.

    The input list is the collection of transforms that follow the
    inverted one.
    """
    from niworkflows.utils.connections import listify

    if not inlist:
        return [True]
    return [True] + [False] * len(listify(inlist))
