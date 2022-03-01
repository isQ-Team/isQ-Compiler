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
import System.IO (hPutStrLn, stderr)
data Flag = Input String | Output String | Version | Mode String deriving Eq




options :: [OptDescr Flag]
options = [
    Option ['i'] ["input"] (ReqArg Input "FILE") "Input isQ source file.",
    Option ['o'] ["output"] (ReqArg Output "FILE") "Output file.",
    Option ['v'] ["version"] (NoArg Version) "Show version.",
    Option ['m'] ["mode"] (ReqArg Mode "MODE") "Compilation mode. Supported modes: ast, raii, typecheck, mlir(default)"
    ]



header :: [Char]
header = "Usage: isqc [OPTION...] files..."

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

writeOut :: (Show a)=>Maybe String->Either a String->IO ()
writeOut _ (Left err) = raiseError False (show err)
writeOut (Just p) (Right f) = writeFile p f
writeOut Nothing (Right f) = putStrLn f
main = do
    args<-getArgs
    flags<-compilerOpts args
    when (Version `elem` flags) $ do
        putStrLn $ "isqc (isQ Compiler)"
        exitSuccess
        return ()
    input<-newIORef Nothing
    output<-newIORef Nothing
    mode<-newIORef Nothing
    forM_ flags $ \f->do
            case f of
                Input s -> setExactlyOnce input s "Input file set multiple times!"
                Output s -> setExactlyOnce output s  "Output file set multiple times!"
                Mode m -> setExactlyOnce mode m "Mode set multiple times!"
                Version -> undefined
    input'<-readIORef input
    output'<-readIORef output
    mode'<-fromMaybe "mlir" <$> readIORef mode
    ast<-first SyntaxError <$> parseFileOrStdin input'
    let inputFileName = fromMaybe "<stdin>" input'
    case mode' of
        "mlir"-> do
            writeOut output' (ast >>= compile inputFileName)
        "ast"-> do
            writeOut output' (unpack . pShowNoColor <$> ast)
        "raii"-> do
            writeOut output' (ast >>= (unpack . pShowNoColor<$>) <$> compileRAII)
        "typecheck" -> do
            writeOut output' (ast >>= (unpack . pShowNoColor<$>) <$> compileTypecheck)
        _-> raiseError True  $ ("Bad mode "++mode')
    return ()