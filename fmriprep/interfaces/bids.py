"""BIDS-related interfaces."""

from pathlib import Path

from bids.utils import listify
from nipype.interfaces.base import File, SimpleInterface, TraitedSpec, traits

from .. import config
from ..utils.bids import _find_nearest_path


class _BIDSURIInputSpec(TraitedSpec):
    in_files = traits.Either(
        File(exists=False),
        traits.List(File(exists=False)),
        mandatory=True,
        desc='Input imaging file(s)',
    )


class _BIDSURIOutputSpec(TraitedSpec):
    out = traits.List(
        traits.Str,
        desc='BIDS URI(s) for file',
    )


class BIDSURI(SimpleInterface):
    """Simple clipping interface that clips values to specified minimum/maximum

    If no values are outside the bounds, nothing is done and the in_files is passed
    as the out_file without copying.
    """

    input_spec = _BIDSURIInputSpec
    output_spec = _BIDSURIOutputSpec

    def _run_interface(self, runtime):
        in_files = listify(self.inputs.in_files)
        updated_keys = {f'bids:{k}:': v for k, v in config.execution.dataset_links.items()}
        updated_keys['bids::'] = config.execution.fmriprep_dir
        out = [_find_nearest_path(updated_keys, Path(f)) for f in in_files]
        self._results['out'] = out

        return runtime
