#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Miscellaneous utilities
"""


def fix_multi_T1w_source_name(in_files, session):
    """
    Make up a generic source name when there are multiple T1s

    >>> fix_multi_T1w_source_name([
    ...     '/path/to/sub-045_ses-test_T1w.nii.gz',
    ...     '/path/to/sub-045_ses-retest_T1w.nii.gz'])
    '/path/to/sub-045_T1w.nii.gz'

    """
    import os
    import re
    from nipype.utils.filemanip import filename_to_list
    base, in_file = os.path.split(filename_to_list(in_files)[0])
    subject_label = in_file.split("_", 1)[0].split("-")[1]
    if session:
        temp = re.search('ses-[0-9a-zA-Z]_', in_file).group()
        session_id = temp.split('ses-')[1].split('_')[0]
        filename = os.path.join(
            base, "sub-{0}_ses-{1}_T1w.nii.gz".format(subject_label, session_id))
    else:
        filename = os.path.join(base, "sub-{}_T1w.nii.gz".format(subject_label))
    return filename


def add_suffix(in_files, suffix):
    """
    Wrap nipype's fname_presuffix to conveniently just add a prefix

    >>> add_suffix([
    ...     '/path/to/sub-045_ses-test_T1w.nii.gz',
    ...     '/path/to/sub-045_ses-retest_T1w.nii.gz'], '_test')
    'sub-045_ses-test_T1w_test.nii.gz'

    """
    import os.path as op
    from nipype.utils.filemanip import fname_presuffix, filename_to_list
    return op.basename(fname_presuffix(filename_to_list(in_files)[0],
                                       suffix=suffix))


if __name__ == '__main__':
    pass
