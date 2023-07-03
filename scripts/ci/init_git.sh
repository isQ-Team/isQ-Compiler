#!/usr/bin/env bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
git config --global user.email "gitlab-ci@arclighttest.cn"
git config --global user.name "Gitlab Runner"
giturl=${CI_SERVER_PROTOCOL}://gitlab-ci:${GITLAB_CI_PUSH_TOKEN}@${CI_SERVER_HOST}:${CI_SERVER_PORT}/${CI_PROJECT_NAMESPACE}/${CI_PROJECT_NAME}.git
git remote add origin-push $giturl || true
git remote set-url origin-push $giturl