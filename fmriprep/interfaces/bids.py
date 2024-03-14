"""BIDS-related interfaces."""

from pathlib import Path

from bids.utils import listify
from nipype.interfaces.base import (
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    traits,
)

from ..utils.bids import _find_nearest_path


class _BIDSURIInputSpec(TraitedSpec):
    in_files = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='Input imaging file(s)',
    )
    dataset_links = traits.Dict(mandatory=True, desc='Dataset links')
    out_dir = traits.Str(mandatory=True, desc='Output directory')


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
        updated_keys = {f'bids:{k}:': v for k, v in self.inputs.dataset_links.items()}
        updated_keys['bids::'] = Path(self.inputs.out_dir)
        out = [_find_nearest_path(updated_keys, Path(f)) for f in in_files]
        self._results['out'] = out

        return runtime
