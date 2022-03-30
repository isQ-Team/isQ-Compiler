date "+Time of running test: %Y-%m-%d %H:%M:%S" > Testing_Report.txt
echo "" >> Testing_Report.txt
bash /home/mgcosa/Test_isqv2/Test_Compiler/testmain.sh input output info /home/mgcosa/Test_isqv2/Test_Compiler/subroutines/hasError.sh >> Testing_Report.txt
cat Testing_Report.txt