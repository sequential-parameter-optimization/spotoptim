#!/bin/sh
rm -f dist/spotoptim*; python -m build; python -m pip install dist/spotoptim*.tar.gz
python -m mkdocs build
