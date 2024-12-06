#!/bin/sh
rm -f dist/optspeed*; python -m build; python -m pip install dist/optspeed*.tar.gz
python -m mkdocs build
