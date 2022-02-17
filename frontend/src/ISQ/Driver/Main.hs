module ISQ.Driver.Main where
import ISQ.Lang.ISQv2Parser
import ISQ.Lang.ISQv2Tokenizer
import ISQ.Lang.ISQv2Grammar
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