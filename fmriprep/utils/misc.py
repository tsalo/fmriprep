# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Miscellaneous utilities."""


def check_deps(workflow):
    """Make sure dependencies are present in this system."""
    from nipype.utils.filemanip import which
    return sorted(
        (node.interface.__class__.__name__, node.interface._cmd)
        for node in workflow._get_all_nodes()
        if (hasattr(node.interface, '_cmd') and
            which(node.interface._cmd.split()[0]) is None))


def select_first(in_files):
    """
    Select the first file from a list of filenames.
    Used to grab the first echo's file when processing
    multi-echo data through workflows that only accept
    a single file.

    Examples
    --------
    >>> select_first('some/file.nii.gz')
    'some/file.nii.gz'
    >>> select_first(['some/file1.nii.gz', 'some/file2.nii.gz'])
    'some/file1.nii.gz'
    """
    if isinstance(in_files, (list, tuple)):
        return in_files[0]
    return in_files
