#!/usr/bin/env bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
BRANCH=$($SCRIPTPATH/../get-sem-ver.py ci_git_release_branch)
git push origin-push HEAD:refs/heads/$BRANCH
