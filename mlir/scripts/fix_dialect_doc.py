#!/usr/bin/env python3
import sys
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: "+sys.argv[0]+" <__tmp_IR.md> <IR.md>")
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        lines[1] = lines[1] + " {#ISQDialectDef}"
        with open(sys.argv[2], "w", encoding="utf-8") as f2:
            flag = True
            for line in lines:
                if line == "## Type constraint definition":
                    flag = False
                if line == "## Operation definition":
                    flag = True
                if flag:
                    f2.writelines([line+"\n"])
