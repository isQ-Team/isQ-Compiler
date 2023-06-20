#!/usr/bin/env python
'''
Get semantic version for current git working tree.
'''
import os
import subprocess
import sys
def is_gittree_ditry():
    return (os.system("git diff --quiet") >> 8)==0

def get_git_hash(short=True):
    if short:
        return subprocess.check_output(["git","rev-parse", "--short", "HEAD"],encoding="utf8").strip()
    else:
        return subprocess.check_output(["git","rev-parse","HEAD"],encoding="utf8").strip()
def version():
    return "0.114.514"
def semver_with_rev():
    return version() + "+" + get_git_hash()
def main():
    methods = {
        "version": version,
        "semver_with_rev": semver_with_rev
    }
    if not len(sys.argv) == 2:
        all_methods = " | ".join(map(str, methods.keys()))
        print(f"Usage: get-sem-ver.py [{all_methods}]")
        sys.exit(1)
    print(methods[sys.argv[1]]())

if __name__ == "__main__":
    main()