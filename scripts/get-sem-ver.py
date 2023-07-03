#!/usr/bin/env python
'''
Get semantic version for current git working tree.
'''
import os
import subprocess
import sys
import json
import regex
from os import path
root = os.path.join(os.path.dirname(__file__), "../")
def get_version_json():
    with open(os.path.join(root, "version.json")) as f:
        return json.load(f)

def system_output(s):
    return subprocess.check_output(["bash", "-c", s],encoding="utf8").strip()
def is_gittree_ditry():
    return (os.system("git diff --quiet") >> 8)==0

def get_git_hash(short=True):
    if short:
        return system_out("git rev-parse --short HEAD")
    else:
        return system_out("git rev-parse HEAD")
def version():
    return get_version_json()["version"]

def semver_with_rev():
    return version() + "+" + get_git_hash()


def git_tag_only():
    return system_output("git describe --tags --abbrev=0")
def git_tag():
    return system_output("git describe --tags")
def git_is_tag():
    return git_tag() == git_tag_only()

def semver_git_describe():
    return system_output("git describe --tags --dirty | sed -e 's/-[[:digit:]]\+-g/+/' -e 's/+\([a-zA-Z0-9]*\)-dirty/+\\1.dirty/' -e 's/-dirty/+dirty/'")



def ci_git_release_branch():
    return "release/"+regex.match("([0-9]+\\.[0-9]+\\.)[0-9]+", version())[1] + "x"
def ci_git_tag():
    return "v"+version()

def main():
    methods = {
        "semver_with_rev": semver_with_rev,
        "semver_git_describe": semver_git_describe,
        "ci_git_release_branch": ci_git_release_branch,
        "ci_git_tag": ci_git_tag
    }
    if not len(sys.argv) == 2:
        all_methods = " | ".join(map(str, methods.keys()))
        print(f"Usage: get-sem-ver.py [{all_methods}]")
        sys.exit(1)
    print(methods[sys.argv[1]]())

if __name__ == "__main__":
    main()