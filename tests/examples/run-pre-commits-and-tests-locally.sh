#!/bin/bash
# Copyrights (C) StatsBomb Services Ltd. 2021. - All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly
# prohibited. Proprietary and confidential

# run pre-commits
poetry run pre-commit run --all-files --verbose --config .pre-commit-config.yaml
# $? stores exit value of the last command
if [ $? -ne 0 ]; then
 echo "Pre-commit hooks must pass before commit!"
 exit 1
fi

# run tests
poetry run coverage erase
poetry run coverage run -m pytest tests/unit -v -s $1 #passing cli parameters to pytest
if [ $? -ne 0 ]; then
 echo "Tests must pass before commit!"
 exit 1
fi
poetry run coverage report -m

# check that docs build
poetry run make docs -C docs
if [ $? -ne 0 ]; then
 echo "Docs must build without warning before commit!"
 exit 1
fi
