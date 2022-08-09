{-# LANGUAGE GADTs #-}
module ISQ.Driver.Main where
import System.Console.GetOpt
import System.Environment
import Control.Monad (when, foldM, forM_)
import System.Exit (exitSuccess, exitWith, ExitCode (ExitFailure))
import Data.IORef
import Data.Maybe (fromMaybe)
import ISQ.Driver.Passes
import Text.Pretty.Simple
import Data.Text.Lazy (unpack)
import Data.Bifunctor
import ISQ.Lang.CompileError
import System.IO (hPutStrLn, stderr, stdout)
import Control.Monad.Cont.Class
import Control.Monad.Cont
import Control.Monad.Except 
import Data.Aeson
import ISQ.Driver.Jsonify
import qualified Data.List
import qualified Data.ByteString.Lazy as BS
import ISQ.Lang.ISQv2Tokenizer
import Data.Typeable
data Flag = Input String | Include String | Output String | Version | Mode String | Help  deriving Eq



options :: [OptDescr Flag]
options = [
    Option ['i'] ["input"] (ReqArg Input "FILE") "Input isQ source file.",
    Option ['I'] ["include"] (ReqArg Include "PATH") "isQ include file directory",
    Option ['o'] ["output"] (ReqArg Output "FILE") "Output file.",
    Option ['v'] ["version"] (NoArg Version) "Show version.",
    Option ['m'] ["mode"] (ReqArg Mode "MODE") "Output mode. Supported modes: ast, raii, typecheck, mlir, mlir-llvm, llvm, so(default)",
    Option ['h'] ["help"] (NoArg Help) "Show help."
    ]



header :: [Char]
header = "Usage: isqc1 [OPTION...] files..."

usage :: [Char]
usage = usageInfo header options

raiseError :: Bool->String->IO a
raiseError show_usage err = do
    hPutStrLn stderr err
    when show_usage $ do
        hPutStrLn stderr usage
    exitWith $ ExitFailure 1


ensureJust :: String->Maybe a->IO a
ensureJust errs Nothing = raiseError True errs
ensureJust _ (Just a) = return a
compilerOpts :: [String]->IO [Flag]
compilerOpts argv = do
    case getOpt Permute options argv of
        (o, [], []) -> return o
        (_,_,errs) -> raiseError True ("Bad usage.\n" ++concat errs )


setExactlyOnce :: IORef (Maybe a)->a->String->IO ()
setExactlyOnce r v err = do
    v'<-readIORef r
    case v' of
        Nothing -> writeIORef r (Just v)
        Just _ -> raiseError True err

writeOut :: (ToJSON a, ToJSON b)=>Maybe String->Either a b->IO ()
writeOut (Just p) x = BS.writeFile p (encode x)
writeOut Nothing x = BS.hPut stdout (encode x)


{-
writeOut :: (Show a)=>Maybe String->Either a String->IO ()
writeOut _ (Left err) = raiseError False (show err)
writeOut (Just p) (Right f) = writeFile p f
writeOut Nothing (Right f) = putStrLn f
-}

{-headers :: String
headers = Data.List.intercalate "\n" $ ["extern defgate Rz(double) : gate(1) = \"__quantum__qis__rz__body\";",
    "extern defgate Rx(double) : gate(1) = \"__quantum__qis__rx__body\";",
    "extern defgate Ry(double) : gate(1) = \"__quantum__qis__ry__body\";",
    "extern defgate U3(double, double, double) : gate(1) = \"__quantum__qis__u3\";",
    "extern defgate H() : gate(1) = \"__quantum__qis__h__body\";",
    "extern defgate S() : gate(1) = \"__quantum__qis__s__body\";",
    "extern defgate T() : gate(1) = \"__quantum__qis__t__body\";",
    "extern defgate X() : gate(1) = \"__quantum__qis__x__body\";",
    "extern defgate Y() : gate(1) = \"__quantum__qis__y__body\";",
    "extern defgate Z() : gate(1) = \"__quantum__qis__z__body\";",
    "extern defgate CNOT() : gate(2) = \"__quantum__qis__cnot\";",
    "extern defgate Toffoli() : gate(3) = \"__quantum__qis__toffoli\";",
    "extern defgate X2M() : gate(1) = \"__quantum__qis__x2m\";",
    "extern defgate X2P() : gate(1) = \"__quantum__qis__x2p\";",
    "extern defgate Y2M() : gate(1) = \"__quantum__qis__y2m\";",
    "extern defgate Y2P() : gate(1) = \"__quantum__qis__y2p\";",
    "extern defgate CZ() : gate(2) = \"__quantum__qis__cz\";",
    "extern defgate GPhase(double) : gate(0) = \"__quantum__qis__gphase\";"]-}

main = do
    args<-getArgs
    flags<-compilerOpts args
    when (Help `elem` flags) $ do
        putStrLn usage 
        exitSuccess
    when (Version `elem` flags) $ do
        putStrLn $ "isqc (isQ Compiler)"
        exitSuccess
        return ()
    input<-newIORef Nothing
    inc<-newIORef Nothing
    output<-newIORef Nothing
    mode<-newIORef Nothing
    forM_ flags $ \f->do
            case f of
                Input s -> setExactlyOnce input s "Input file set multiple times!"
                Include s -> setExactlyOnce inc s "Include file directory set multiple times!"
                Output s -> setExactlyOnce output s  "Output file set multiple times!"
                Mode m -> setExactlyOnce mode m "Mode set multiple times!"
                Version -> undefined
    input'<-readIORef input
    inc' <- readIORef inc
    output'<-readIORef output
    mode'<-fromMaybe "mlir" <$> readIORef mode
    let incpath = fromMaybe "" inc'
    --putStrLn $ show $ typeOf ast
    let inputFileName = fromMaybe "<stdin>" input'
    ast <- generateTcast incpath inputFileName
    
    --header_ast<-fmap (\x->case x of {Right y->y; Left e->error (show e)})(runExceptT $ parseToAST headers)
    --let zeroed_ast = fmap (fmap (const $ Pos 0 0 "")) header_ast
    --let ast = fmap (zeroed_ast ++) ast_body;
    case mode' of
        "mlir"-> do
            writeOut output' (ast >>= compile inputFileName)
        "ast"-> do
            writeOut output' ast
        --"raii"-> do
        --    writeOut output' (ast >>=  compileRAII)
        --"typecheck" -> do
        --    writeOut output' (ast >>= compileTypecheck)
        _-> raiseError True ("Bad mode "++mode')
    return ()