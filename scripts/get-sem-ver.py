#!/usr/bin/env python
'''
Get semantic version for current git working tree.
'''
import os
import subprocess
import sys
import json

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
    return "0.114.514"
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


def semver_current_tree(tag = git_tag_only()):
    tag = git_tag_only()


def main():
    methods = {
        "version": version,
        "semver_with_rev": semver_with_rev,
        "semver_git_describe": semver_git_describe,
        "semver_json": semver_json
    }
    if not len(sys.argv) == 2:
        all_methods = " | ".join(map(str, methods.keys()))
        print(f"Usage: get-sem-ver.py [{all_methods}]")
        sys.exit(1)
    print(methods[sys.argv[1]]())

if __name__ == "__main__":
    main()