# SPDX-FileCopyrightText: 2022 RStudio, PBC
# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: MIT
#
# Adapted from https://github.com/rstudio/vetiver-python/blob/main/docs/version_config.py

from importlib_metadata import version as _version

v = f"""VERSION={_version('spotoptim')}"""

f = open("_environment", "w")
f.write(v)