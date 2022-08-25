#!/bin/bash
# Copyrights (C) StatsBomb Services Ltd. 2021. - All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly
# prohibited. Proprietary and confidential

GIT_DIR=$(git rev-parse --git-dir)

pwd
echo "Installing hooks..."
# this command creates symlink to our pre-commit script
ln -s ../../scripts/run-pre-commits-and-tests-locally.sh $GIT_DIR/hooks/pre-commit
echo "Done"!
