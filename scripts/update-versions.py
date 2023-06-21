#!/usr/bin/env python
'''
Update package metadata.
'''
import json
import sys
import subprocess
import os

root = os.path.join(os.path.dirname(__file__), "../")
def system_output(s, cwd = root):
    return subprocess.check_output(["bash", "-c", s],encoding="utf8", cwd=cwd).strip()


with open(os.path.join(root, "version.json")) as f:
    version = json.load(f)

all_cargo_tomls = system_output("find -name Cargo.toml").split("\n")
all_package_yamls = system_output("find -name package.yaml").split("\n")

def update_cargo_toml(file):
    file_abs = os.path.join(root, file)
    proj_path = os.path.dirname(file_abs)
    v = version["version"]
    system_output(f'sed -i -e "s/^version\\s*=\\s*\\"[^\\"]*\\"/version = \\"{v}\\"/"  {file_abs}')
    system_output("cargo update -w", proj_path)
def update_package_yaml(file):
    file_abs = os.path.join(root, file)
    v = version["version"]
    system_output(f'sed -i -e "s/^version:\s*.*$/version: {v}/"  {file_abs}')

for cargo_toml in all_cargo_tomls:
    update_cargo_toml(cargo_toml)
for package_toml in all_package_yamls:
    update_package_yaml(package_toml)
