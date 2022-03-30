#第一个参数是输出报错信息路径，第二个参数是输出文件路径
if [[ -n $1 ]]; then

    PWD=`dirname $0`
    ResultFiles=`ls $1`
    for rawfile in $ResultFiles
    do
        catpath="${PWD}/../catpath.exe"
        infofile=`$catpath $1 $rawfile`
        outfile=`$catpath $2 ${rawfile%.*}.so`
        mainline=`cat $infofile`
        errline=`cat $infofile | grep "Error:"`

        if [[ -s $outfile ]]; then     #是否生成了so文件？
            if [[ -z $mainline ]]; then    #控制台是否有额外输出信息？
                echo "$rawfile ----- no error : pass"
            else
                echo "$rawfile ----- Build .so successfully, but there are some extra output in console : Note"
            fi
        else    #没有
            if [[ -z $errline ]]; then     #报错信息是否正常？
                echo "$rawfile ----- Failed to build .so : ERROR!!!!!; but the error info may have some problems."
            else
                echo "$rawfile ----- Failed to build .so : ERROR!!!!!"
            fi
        fi
    done
else
    echo "Please give the output result directory!"
fi