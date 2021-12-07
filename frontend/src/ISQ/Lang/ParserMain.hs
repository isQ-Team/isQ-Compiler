module ISQ.Lang.ParserMain where
import ISQ.Lang.Language
import ISQ.Lang.MLIRGen
import Text.Parsec
import Data.Functor.Identity
import Data.List
parserMain :: IO ()
parserMain = do
    f<-getContents
    let r = runIdentity $ runParserT parseISQ () "main.isq" f
    case r of
        Left err -> print err
        Right ast -> do
            let ast' = mlirGen ast
            case ast' of
                Left err -> print err
                Right ast'' -> putStrLn $ intercalate "\n" ast''