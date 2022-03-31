#!/usr/bin/env bash
date "+Time of running test: %Y-%m-%d %H:%M:%S" > Testing_Report.txt
echo "" >> Testing_Report.txt
bash `pwd`/testmain.sh input output info `pwd`/subroutines/hasError.sh | tee Testing_Report.txt
cat Testing_Report.txt