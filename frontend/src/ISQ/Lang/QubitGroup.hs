module ISQ.Lang.QubitGroup where
import ISQ.Lang.Codegen
import ISQ.Lang.AST
data QubitGroupIter ann = QubitBundleIter {
    qubitBundleLength :: SSA,
    qubitBundleSSA :: SSA,
    qubitGroupAnnotation :: ann
} | QubitRangeIter {
    qubitRangeSteps :: SSA,
    loSSA :: SSA,
    stepSSA :: SSA,
    qubitGroupAnnotation :: ann
} deriving Show 


initQubitGroup :: (CodeSunk m)=> QubitGroup Pos -> CodegenM m (QubitGroupIter Pos)
initQubitGroup (QubitBundle list ann) = do
    l <- nextSSA
    emitConst ann (length list) l
    bundleElems<-mapM evalExpr list
    bundleElems'<-mapM requireInt bundleElems
    arr<-nextSSA
    emitIntArray ann bundleElems' arr
    return $ QubitBundleIter l arr ann
initQubitGroup (QubitRange lo hi step ann) = do
    lo' <- evalExpr lo
    lo'' <- requireInt lo'
    hi' <- evalExpr hi
    hi'' <- requireInt hi'
    step' <- evalExpr step
    step'' <- requireInt step'
    one <- nextSSA
    emitConst ann 1 one
    hi_sub_one <- nextSSA
    emitBinaryOp ann Sub hi'' one hi_sub_one
    r <- nextSSA
    emitBinaryOp ann Div hi_sub_one step'' r
    return $ QubitRangeIter r lo'' step'' ann

evalQubitGroup :: (CodeSunk m)=>SSA->QubitGroupIter Pos->CodegenM m SSA
evalQubitGroup i (QubitRangeIter _ lo step ann) = do
    tmp1 <- nextSSA
    emitBinaryOp ann Mul i step tmp1
    tmp2 <- nextSSA
    emitBinaryOp ann Add tmp1 lo tmp2
    return tmp2
evalQubitGroup i (QubitBundleIter _ arr ann) = do
    tmp1 <- nextSSA
    emitReadIntAtOp ann tmp1 arr i
    return tmp1