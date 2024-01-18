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

from nireports.assembler.report import Report

from .. import config


def generate_reports(subject_list, output_dir, run_uuid, config=None, work_dir=None):
    """Generate reports for a list of subjects."""
    reportlets_dir = None
    if work_dir is not None:
        reportlets_dir = Path(work_dir) / "reportlets"

    errors = []
    for subject_label in subject_list:
        entities = {}
        entities["subject"] = subject_label

        robj = Report(
            output_dir,
            run_uuid,
            bootstrap_file=config,
            reportlets_dir=reportlets_dir,
            plugins=None,
            plugin_meta=None,
            metadata=None,
            **entities,
        )

        # Count nbr of subject for which report generation failed
        try:
            robj.generate_report()
        except:
            import sys
            import traceback

            errors.append(subject_label)
            traceback.print_exception(
                *sys.exc_info(),
                file=str(Path(output_dir) / "logs" / f"report-{run_uuid}-{subject_label}.err"),
            )

    if errors:
        logger.debug(
            "Report generation was not successful for the following participants : %s.",
            ", ".join(errors),
        )
    return len(errors)
