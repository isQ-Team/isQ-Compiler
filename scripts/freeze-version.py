#!/usr/bin/env python
import json
import os
root = os.path.join(os.path.dirname(__file__), "../")
with open(os.path.join(root, "version.json")) as f:
    version = json.load(f)
version["frozen"]=True
with open(os.path.join(root, "version.json"), 'w') as f:
    json.dump(version, f)