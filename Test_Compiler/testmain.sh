#第一个参数是输入文件路径，第二个参数是输出文件路径，第三个参数是输出的报错信息路径，第四个参数是分析测试结果的脚本
if [[ -n $1 ]] && [[ -n $2 ]]; then

    PWD=`dirname $0`
    InputFiles=`ls $1`
    for rawinput in $InputFiles
    do
        catpath="${PWD}/catpath.exe"
        input=`$catpath $1 $rawinput`
        output=`$catpath $2 $rawinput.so`
        info=`$catpath $3 $rawinput.txt`
        bash "${PWD}/program.sh" $input $output $info
    done
    if [[ -n $4 ]]; then
        bash $4 $3 $2      #对报错信息进行分析
    fi

else
    echo "Please give input directory and output directory!"
fi