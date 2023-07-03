#!/usr/bin/env bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
ROOTPATH=$SCRIPTPATH/../../
TAG=$($SCRIPTPATH/../get-sem-ver.py ci_git_tag)
git add $ROOTPATH/version.json
git commit -m "Version $TAG frozen."
git push origin-push HEAD:refs/tags/$TAG
