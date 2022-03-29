{-# LANGUAGE FlexibleInstances #-}
module ISQ.Driver.Passes where
import ISQ.Lang.ISQv2Parser
import ISQ.Lang.ISQv2Tokenizer
import ISQ.Lang.ISQv2Grammar
import ISQ.Lang.CompileError
import Control.Monad.Except 
import ISQ.Lang.TypeCheck (typeCheckTop, TCAST)
import ISQ.Lang.RAIICheck (raiiCheck)
import Text.Pretty.Simple ( pPrint, pPrintNoColor )
import ISQ.Lang.MLIRGen (generateMLIRModule)
import ISQ.Lang.MLIRTree (emitOp)
import System.Environment (getArgs)
import Control.Exception (catch, Exception, evaluate)
import Data.Bifunctor (first)
import Control.DeepSeq
import ISQ.Driver.Jsonify
import Data.Char (isDigit)
import ISQ.Lang.DeriveGate (passDeriveGate)
import ISQ.Lang.OraclePass (passOracle)

syntaxError :: String->CompileError 
syntaxError x = 
    let t1 = dropWhile (not . isDigit) x
        (l, t2) = span isDigit t1
        t3 = dropWhile (not . isDigit) t2
        c = takeWhile isDigit t3
    in SyntaxError (Pos {line = read l, column = read c} )

parseToAST :: String->ExceptT CompileError IO [LAST]
parseToAST s = do
    tokens <- liftEither $ first syntaxError $ tokenize s
    x <-lift $ catch (Right <$> (evaluate $ force $ isqv2 tokens)) 
        (\e-> case e of
                x:_->(return $ Left $ GrammarError $ UnexpectedToken x)
                [] -> return $ Left $ GrammarError $ UnexpectedEOF)
    liftEither x
parseFile :: String->IO (Either CompileError [LAST])
parseFile path = do
    f<-readFile path
    runExceptT $ parseToAST f

parseStdin :: IO (Either CompileError [LAST])
parseStdin = do
    f<-getContents
    runExceptT $ parseToAST f

parseFileOrStdin :: Maybe String -> IO (Either CompileError [LAST])
parseFileOrStdin (Just x) = parseFile x
parseFileOrStdin Nothing = parseStdin


pass :: (CompileErr e, Monad m)=>Either e a->ExceptT CompileError m a
pass (Left x) = throwError $ fromError x
pass (Right y) = return y


compileRAII :: [LAST] -> Either CompileError [LAST]
compileRAII ast = runExcept $ do
    ast_oracle <- pass $ passOracle ast
    ast_derive <- pass $ passDeriveGate ast_oracle
    ast_verify_defgate <- pass $ passVerifyDefgate ast_derive
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