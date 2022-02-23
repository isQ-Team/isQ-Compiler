module ISQ.Driver.Main where
import ISQ.Lang.ISQv2Parser
import ISQ.Lang.ISQv2Tokenizer
import ISQ.Lang.ISQv2Grammar
import ISQ.Lang.CompileError
import Control.Monad.Except 
import ISQ.Lang.TypeCheck (typeCheckTop)
import ISQ.Lang.RAIICheck (raiiCheck)
import Text.Pretty.Simple
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

compile ast = runExcept $ do
    ast_raii <- pass $ raiiCheck ast
    --ast_tc <- pass $ typeCheckTop ast_raii
    return ast_raii

p :: Show a=>a->IO ()
p = pPrint