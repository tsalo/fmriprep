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
"""Utilities for confounds manipulation."""


def acompcor_masks(in_files):
    """
    Generate aCompCor masks.

    This function selects the CSF partial volume map from the input,
    and generates the WM and combined CSF+WM masks for aCompCor.

    The implementation deviates from Behzadi et al.
    Their original implementation thresholded the CSF and the WM partial-volume
    masks at 0.99 (i.e., 99% of the voxel volume is filled with a particular tissue),
    and then binary eroded that 2 voxels:

    > Anatomical data were segmented into gray matter, white matter,
    > and CSF partial volume maps using the FAST algorithm available
    > in the FSL software package (Smith et al., 2004). Tissue partial
    > volume maps were linearly interpolated to the resolution of the
    > functional data series using AFNI (Cox, 1996). In order to form
    > white matter ROIs, the white matter partial volume maps were
    > thresholded at a partial volume fraction of 0.99 and then eroded by
    > two voxels in each direction to further minimize partial voluming
    > with gray matter. CSF voxels were determined by first thresholding
    > the CSF partial volume maps at 0.99 and then applying a threedimensional
    > nearest neighbor criteria to minimize multiple tissue
    > partial voluming. Since CSF regions are typically small compared
    > to white matter regions mask, erosion was not applied.

    This particular procedure is not generalizable to BOLD data with different voxel zooms
    as the mathematical morphology operations will be scaled by those.
    Also, from reading the excerpt above and the tCompCor description, I (@oesteban)
    believe that they always operated slice-wise given the large slice-thickness of
    their functional data.

    Instead, *fMRIPrep*'s implementation deviates from Behzadi's implementation on two
    aspects:

      * the masks are prepared in high-resolution, anatomical space and then
        projected into BOLD space; and,
      * instead of using binary erosion, a dilated GM map is generated after thresholding
        the corresponding PV map at >0.05 (i.e., pixels containing more than 5% of GM tissue)
        and then subtracting that map from the CSF, WM and CSF+WM (combined) masks.
        This should be equivalent to eroding the masks, except that the erosion
        only happens at direct interfaces with GM.

    """
    from pathlib import Path

    import nibabel as nb
    from scipy.ndimage import binary_dilation
    from skimage.morphology import ball

    csf_file = in_files[2]  # BIDS labeling (CSF=2; last of list)
    # Load PV maps (fast) or segments (recon-all)
    gm_vf = nb.load(in_files[0])
    wm_vf = nb.load(in_files[1])
    csf_vf = nb.load(csf_file)

    gm_data = gm_vf.get_fdata() > 0.05
    wm_data = wm_vf.get_fdata()
    csf_data = csf_vf.get_fdata()

    # Dilate the GM mask
    gm_data = binary_dilation(gm_data, structure=ball(3))

    # Output filenames
    wm_file = str(Path('acompcor_wm.nii.gz').absolute())
    combined_file = str(Path('acompcor_wmcsf.nii.gz').absolute())

    # Prepare WM mask
    wm_data[gm_data] = 0  # Make sure voxel does not contain GM
    nb.Nifti1Image(wm_data, gm_vf.affine, gm_vf.header).to_filename(wm_file)

    # Prepare combined CSF+WM mask
    comb_data = csf_data + wm_data
    comb_data[gm_data] = 0  # Make sure voxel does not contain GM
    nb.Nifti1Image(comb_data, gm_vf.affine, gm_vf.header).to_filename(combined_file)
    return [csf_file, wm_file, combined_file]
