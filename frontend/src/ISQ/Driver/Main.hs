module ISQ.Driver.Main where
import ISQ.Lang.ISQv2Parser
import ISQ.Lang.ISQv2Tokenizer
import ISQ.Lang.ISQv2Grammar
import ISQ.Lang.CompileError
import Control.Monad.Except 
import ISQ.Lang.TypeCheck (typeCheckTop, TCAST)
import ISQ.Lang.RAIICheck (raiiCheck)
import Text.Pretty.Simple
import ISQ.Lang.MLIRGen (generateMLIRModule)
import ISQ.Lang.MLIRTree (emitOp)
import System.Environment (getArgs)
parseToAST :: String->Either String [LAST]
parseToAST s = do
    tokens <- tokenize s
    return $ isqv2 tokens
    
parseFile :: String->IO (Either String [LAST])
parseFile path = do
    f<-readFile path
    return $ parseToAST f

parseStdin :: IO ()
parseStdin = do
    f<-getContents
    print $ parseToAST f

pass :: (CompileErr e, Monad m)=>Either e a->ExceptT CompileError m a
pass (Left x) = throwError $ fromError x
pass (Right y) = return y


compileRAII :: [LAST] -> Either CompileError [LAST]
compileRAII ast = runExcept $ do
    ast_verify_defgate <- pass $ passVerifyDefgate ast
    ast_verify_top_vardef<-pass $ checkTopLevelVardef ast_verify_defgate
    ast_raii <- pass $ raiiCheck ast_verify_top_vardef
    return ast_raii

compileTypecheck :: [LAST]->Either CompileError [TCAST]
compileTypecheck ast = runExcept $ do
    ast_raii<- liftEither $ compileRAII ast
    ast_tc <- pass $ typeCheckTop ast_raii
    return ast_tc

compile :: String->[LAST] -> Either CompileError String
compile s ast = runExcept $ do
    ast_tc<-liftEither $ compileTypecheck ast
    let mlir_module = generateMLIRModule s ast_tc
    return $ emitOp mlir_module

p :: Show a=>a->IO ()
p = pPrint

printRAII s = do
    Right f<-parseFile s
    return $ compileRAII f
printTC s = do
    Right f<-parseFile s
    return $ compileTypecheck f

genMLIR s = do
    Right f<-parseFile s
    case compile s f of
        Left x -> print x
        Right mlir -> putStrLn mlir

raiiMain :: IO ()
raiiMain = do
    (input:_) <- getArgs;
    ast<-printRAII input
    pPrintNoColor ast
    return ()