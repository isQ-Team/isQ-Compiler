module ISQ.Driver.Main where
import System.Console.GetOpt
import System.Environment
import Control.Monad (when, foldM, forM_)
import System.Exit (exitSuccess)
import Data.IORef
import Data.Maybe (fromMaybe)
import ISQ.Driver.Passes
import Text.Pretty.Simple
import Data.Text.Lazy (unpack)
import Data.Bifunctor
import ISQ.Lang.CompileError
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
ensureJust :: String->Maybe a->IO a
ensureJust errs Nothing = ioError (userError ( errs ++ usageInfo header options))
ensureJust _ (Just a) = return a
compilerOpts :: [String]->IO [Flag]
compilerOpts argv = do
    case getOpt Permute options argv of
        (o, [], []) -> return o
        (_,_,errs) -> ioError (userError (concat errs ++ usageInfo header options))


setExactlyOnce :: IORef (Maybe a)->a->String->IO ()
setExactlyOnce r v err = do
    v'<-readIORef r
    case v' of
        Nothing -> writeIORef r (Just v)
        Just _ -> ioError (userError (err ++ usageInfo header options))

writeOut :: (Show a)=>String->Either a String->IO ()
writeOut _ (Left err) = error (show err)
writeOut p (Right f) = writeFile p f
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
    input'<-readIORef input >>= ensureJust "Input not specified!\n"
    output'<-readIORef output >>= ensureJust "Output not specified!\n"
    mode'<-fromMaybe "mlir" <$> readIORef mode
    ast<-first SyntaxError <$> parseFile input'
    case mode' of
        "mlir"-> do
            writeOut output' (ast >>= compile input')
        "ast"-> do
            writeOut output' (unpack . pShowNoColor <$> ast)
        "raii"-> do
            writeOut output' (ast >>= (unpack . pShowNoColor<$>) <$> compileRAII)
        "typecheck" -> do
            writeOut output' (ast >>= (unpack . pShowNoColor<$>) <$> compileTypecheck)
        _-> ioError (userError $ "Bad mode "++mode')
    return ()