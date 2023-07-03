#!/usr/bin/env bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
git config --global user.email "gitlab-ci@arclighttest.cn"
git config --global user.name "Gitlab Runner"
