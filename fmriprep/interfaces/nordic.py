"""NORDIC-related interfaces."""

import os
from pathlib import Path
from string import Template

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.matlab import MatlabCommand


class _ValidateComplexInputSpec(BaseInterfaceInputSpec):
    mag_file = File(
        exists=True,
        mandatory=True,
        desc='Magnitude BOLD EPI',
    )
    phase_file = File(
        exists=True,
        mandatory=False,
        desc='Phase BOLD EPI',
    )
    n_mag_noise_volumes = traits.Int(
        mandatory=False,
        default=0,
        usedefault=True,
        desc='Number of volumes in the magnitude noise scan',
    )
    n_phase_noise_volumes = traits.Int(
        mandatory=False,
        default=0,
        usedefault=True,
        desc='Number of volumes in the phase noise scan',
    )


class _ValidateComplexOutputSpec(TraitedSpec):
    mag_file = File(exists=True, desc='Validated magnitude file')
    phase_file = File(exists=True, desc='Validated phase file')
    n_noise_volumes = traits.Int(desc='Number of noise volumes')


class ValidateComplex(SimpleInterface):
    """Validate complex-valued BOLD data."""

    input_spec = _ValidateComplexInputSpec
    output_spec = _ValidateComplexOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb

        if not isdefined(self.inputs.phase_file):
            self._results['mag_file'] = self.inputs.mag_file
            self._results['n_noise_volumes'] = self.inputs.n_mag_noise_volumes
            return runtime

        if self.inputs.n_mag_noise_volumes != self.inputs.n_phase_noise_volumes:
            raise ValueError(
                f'Number of noise volumes in magnitude file ({self.inputs.n_mag_noise_volumes}) '
                f'!= number of noise volumes in phase file ({self.inputs.n_phase_noise_volumes})'
            )

        mag_img = nb.load(self.inputs.mag_file)
        phase_img = nb.load(self.inputs.phase_file)
        n_mag_vols = mag_img.shape[3]
        n_phase_vols = phase_img.shape[3]

        if n_mag_vols != n_phase_vols:
            raise ValueError(
                f'Number of volumes in magnitude file ({n_mag_vols}) '
                f'!= number of volumes in phase file ({n_phase_vols})'
            )

        self._results['mag_file'] = self.inputs.mag_file
        self._results['phase_file'] = self.inputs.phase_file
        self._results['n_noise_volumes'] = self.inputs.n_mag_noise_volumes

        return runtime


class _AppendNoiseInputSpec(BaseInterfaceInputSpec):
    bold_file = File(
        exists=True,
        mandatory=True,
        desc='BOLD file without noise volumes',
    )
    norf_file = File(
        exists=True,
        mandatory=False,
        desc='No radio-frequency pulse noise file',
    )


class _AppendNoiseOutputSpec(TraitedSpec):
    bold_file = File(exists=True, desc='BOLD file with noise volumes appended')
    n_noise_volumes = traits.Int(desc='Number of noise volumes')


class AppendNoise(SimpleInterface):
    """Validate complex-valued BOLD data."""

    input_spec = _AppendNoiseInputSpec
    output_spec = _AppendNoiseOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb
        from nilearn.image import concat_imgs

        if not isdefined(self.inputs.norf_file):
            self._results['bold_file'] = self.inputs.bold_file
            self._results['n_noise_volumes'] = 0
            return runtime

        norf_img = nb.load(self.inputs.norf_file)
        concat_img = concat_imgs([self.inputs.bold_file, norf_img])

        out_file = Path(runtime.cwd) / 'appended_noise.nii.gz'
        concat_img.to_filename(str(out_file))
        self._results['n_noise_volumes'] = norf_img.shape[3]
        self._results['bold_file'] = str(out_file)

        return runtime


class _RemoveNoiseInputSpec(BaseInterfaceInputSpec):
    bold_file = File(
        exists=True,
        mandatory=True,
        desc='BOLD file without noise volumes',
    )
    n_noise_volumes = traits.Int(
        mandatory=False,
        default=0,
        usedefault=True,
        desc='Number of noise volumes in the BOLD file',
    )


class _RemoveNoiseOutputSpec(TraitedSpec):
    bold_file = File(exists=True, desc='BOLD file with noise volumes removed')


class RemoveNoise(SimpleInterface):
    """Validate complex-valued BOLD data."""

    input_spec = _RemoveNoiseInputSpec
    output_spec = _RemoveNoiseOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb

        if self.inputs.n_noise_volumes == 0:
            self._results['bold_file'] = self.inputs.bold_file
            return runtime

        bold_img = nb.load(self.inputs.bold_file)
        bold_img = bold_img.slicer[..., :-self.inputs.n_noise_volumes]

        out_file = Path(runtime.cwd) / 'noise_removed.nii.gz'
        bold_img.to_filename(str(out_file))
        self._results['bold_file'] = str(out_file)

        return runtime


class _NORDICInputSpec(BaseInterfaceInputSpec):
    mag_file = File(exists=True, mandatory=True)
    phase_file = File(exists=True, mandatory=False)
    n_noise_volumes = traits.Int(mandatory=False, default=0, usedefault=True)
    out_prefix = traits.Str('denoised', usedefault=True)


class _NORDICOutputSpec(TraitedSpec):
    mag_file = File(exists=True)
    phase_file = File(exists=True)


class NORDIC(BaseInterface):
    input_spec = _NORDICInputSpec
    output_spec = _NORDICOutputSpec

    def _run_interface(self, runtime):
        d = {
            'mag_file': self.inputs.mag_file,
            'out_dir': os.getcwd(),
            'out_prefix': self.inputs.out_prefix,
            'n_noise_vols': self.inputs.n_noise_vols,
        }
        if isdefined(self.inputs.phase_file):
            d['phase_file'] = self.inputs.phase_file
            d['magnitude_only'] = 0
        else:
            d['phase_file'] = ''
            d['magnitude_only'] = 1

        # This is your MATLAB code template
        script = Template(
            """
% A template MATLAB script to run NORDIC on a magnitude+phase file pair.
% Settings come from Thomas Madison.

% Set args as recommended for fMRI
% Set to 0 if input includes both magnitude + phase timeseries
ARG.magnitude_only = $magnitude_only;
% Save out the phase data too
ARG.make_complex_nii = 1;
% Set to 1 for fMRI
ARG.temporal_phase = 1;
% Set to 1 to enable NORDIC denoising
ARG.NORDIC = 1;
% Use 10 for fMRI
ARG.phase_filter_width = 10;
% Set equal to number of noise frames at end of scan, if present
ARG.noise_volume_last = $n_noise_vols;
% DIROUT may need to be separate from fn_out
ARG.DIROUT = '$out_dir';

fn_magn_in = '$mag_file';
fn_phase_in = '$phase_file';
fn_out = '$out_prefix'

% Add the NORDIC code
addpath('/path/to/nifti/library/')
addpath('/path/to/NORDIC_Raw/')

% Call NORDIC on the input files
NIFTI_NORDIC(fn_magn_in, fn_phase_in, fn_out, ARG)
exit;
"""
        ).substitute(d)

        # mfile = True  will create an .m file with your script and executed.
        # Alternatively
        # mfile can be set to False which will cause the matlab code to be
        # passed
        # as a commandline argument to the matlab executable
        # (without creating any files).
        # This, however, is less reliable and harder to debug
        # (code will be reduced to
        # a single line and stripped of any comments).
        mlab = MatlabCommand(script=script, mfile=True)
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        prefix = self.inputs.out_prefix
        outputs['mag_file'] = os.path.join(os.getcwd(), f'{prefix}magn.nii')
        if isdefined(self.inputs.phase_file):
            outputs['phase_file'] = os.path.join(os.getcwd(), f'{prefix}phase.nii')
        return outputs
