# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
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
from pathlib import Path

from nireports.assembler.report import Report as _Report

# This patch is intended to permit fMRIPrep 20.2.0 LTS to use the YODA-style
# derivatives directory. Ideally, we will remove this in 20.3.x and use an
# updated niworkflows.


class Report(_Report):
    def _load_config(self, config):
        from yaml import safe_load as load

        settings = load(config.read_text())
        self.packagename = self.packagename or settings.get("package", None)

        # Removed from here: Appending self.packagename to self.root and self.out_dir
        # In this version, pass reportlets_dir and out_dir with fmriprep in the path.

        if self.subject_id is not None:
            self.root = self.root / f"sub-{self.subject_id}"

        if "template_path" in settings:
            self.template_path = config.parent / settings["template_path"]

        self.index(settings["sections"])


#
# The following are the interface used directly by fMRIPrep
#


def generate_reports(subject_list, output_dir, run_uuid, config=None, work_dir=None):
    """Generate reports for a list of subjects."""
    reportlets_dir = None
    if work_dir is not None:
        reportlets_dir = Path(work_dir) / "reportlets"

    report_errors = []
    for subject_label in subject_list:
        entities = {}
        entities["sub"] = subject_label

        robj = Report(
            output_dir,
            run_uuid,
            bootstrapfile=config,
            reportlets_dir=reportlets_dir,
            plugins=None,
            plugin_meta=None,
            metadata=None,
            **entities,
        )

        # Count nbr of subject for which report generation failed
        errno = 0
        try:
            robj.generate_report()
        except:
            errno += 1

    if errno:
        import logging

        logger = logging.getLogger("cli")
        error_list = ", ".join(
            f"{subid} ({err})" for subid, err in zip(subject_list, report_errors) if err
        )
        logger.error(
            "Preprocessing did not finish successfully. Errors occurred while processing "
            "data from participants: %s. Check the HTML reports for details.",
            error_list,
        )
    return errno
