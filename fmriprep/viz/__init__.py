# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The fmriprep reporting engine for visual assessment
"""
from .reports import run_reports, generate_reports

__all__ = ['run_reports', 'generate_reports']
