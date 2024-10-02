import json
from pathlib import Path

import pytest

from fmriprep.data import load as load_data
from fmriprep.utils import bids


@pytest.mark.parametrize('xfm', ['boldref2fmap', 'boldref2anat', 'hmc'])
def test_transforms_found_as_str(tmp_path: Path, xfm: str):
    entities = {
        'subject': '0',
        'task': 'rest',
        'suffix': 'bold',
        'extension': '.nii.gz',
    }
    if xfm == 'boldref2fmap':
        to_find = f'sub-{entities["subject"]}_task-{entities["task"]}_from-{spec["from"]}_to-auto00000_mode-image_xfm.txt'  # noqa: E501
    else:
        to_find = f'sub-{entities["subject"]}_task-{entities["task"]}_from-{spec["from"]}_to-{spec["to"]}_mode-image_xfm.txt'  # noqa: E501

    funcd = tmp_path / f'sub-{entities["subject"]}' / 'func'
    funcd.mkdir(parents=True)
    (funcd / to_find).touch()

    derivs = bids.collect_derivatives(
        derivatives_dir=tmp_path,
        entities=entities,
        fieldmap_id='auto_00000',
    )
    transforms_in_derivs = 'transforms' in derivs
    xfm_in_transforms = xfm in derivs.get('transforms')
    transform_is_str = isinstance(derivs.get('transforms').get(xfm), str)
    assert all((transforms_in_derivs, xfm_in_transforms, transform_is_str))
