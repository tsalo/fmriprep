"""BIDS-related interfaces."""

from nipype.interfaces.base import File, SimpleInterface, TraitedSpec, traits

from pathlib import Path

from .. import config
from ..utils.bids import _find_nearest_path


class _BIDSURIInputSpec(TraitedSpec):
    in_file = traits.Either(
        File(exists=False),
        traits.List(File(exists=False)),
        mandatory=True,
        desc='Input imaging file(s)',
    )


class _BIDSURIOutputSpec(TraitedSpec):
    uri = traits.Either(
        traits.Str,
        traits.List(traits.Str),
        desc='BIDS URI(s) for file',
    )


class BIDSURI(SimpleInterface):
    """Simple clipping interface that clips values to specified minimum/maximum

    If no values are outside the bounds, nothing is done and the in_file is passed
    as the out_file without copying.
    """

    input_spec = _BIDSURIInputSpec
    output_spec = _BIDSURIOutputSpec

    def _run_interface(self, runtime):
        updated_keys = {f'bids:{k}:': v for k, v in config.execution.dataset_links.items()}
        updated_keys['bids::'] = config.execution.fmriprep_dir
        if isinstance(self.inputs.in_file, list):
            uri = [_find_nearest_path(f, updated_keys) for f in self.inputs.in_file]
        else:
            uri = _find_nearest_path(updated_keys, Path(self.inputs.in_file))

        self._results['uri'] = uri

        return runtime
