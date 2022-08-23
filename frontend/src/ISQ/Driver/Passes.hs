{-# LANGUAGE FlexibleInstances #-}
module ISQ.Driver.Passes where
import Control.DeepSeq (force)
import Control.Exception (catch, Exception, evaluate, SomeException, try)
import Control.Monad.Except 
import Control.Monad.State
import Data.Aeson
import Data.Bifunctor (first)
import qualified Data.ByteString.Lazy as BS
import Data.Char (isDigit)
import Data.Either (lefts, rights)
import Data.Function (on)
import Data.List (filter, findIndex, findIndices, groupBy, intercalate, maximumBy)
import Data.List.Split (splitOn)
import qualified Data.Map.Lazy as Map
import qualified Data.MultiMap as MultiMap
import ISQ.Driver.Jsonify
import ISQ.Lang.CompileError
import ISQ.Lang.DeriveGate (passDeriveGate)
import ISQ.Lang.ISQv2Parser
import ISQ.Lang.ISQv2Tokenizer
import ISQ.Lang.ISQv2Grammar
import ISQ.Lang.MLIRGen (generateMLIRModule)
import ISQ.Lang.MLIRTree (emitOp)
import ISQ.Lang.OraclePass (passOracle)
import ISQ.Lang.TypeCheck (SymbolTableLayer, TCAST, typeCheckTop)
import ISQ.Lang.RAIICheck (raiiCheck)
--import Text.Pretty.Simple ( pPrint, pPrintNoColor )
--import System.Environment (getArgs)
--import ISQ.Lang.FlatInc (parseIncFile, parseIncStdin)
import System.Directory (canonicalizePath, doesFileExist)
import System.Environment (lookupEnv)
import System.FilePath (addExtension, dropExtensions, joinPath, splitDirectories)
import System.IO (stdout)

syntaxError :: FilePath -> String->CompileError 
syntaxError file x = 
    let t1 = dropWhile (not . isDigit) x
        (l, t2) = span isDigit t1
        t3 = dropWhile (not . isDigit) t2
        (c, t4) = span isDigit t3
    in SyntaxError (Pos {line = read l, column = read c, filename = file} )

parseToAST :: FilePath -> String->ExceptT CompileError IO LAST
parseToAST file s = do
    let assignFile = (\x -> do
            let ann = annotationToken x
            let newAnn = (\y -> y{filename=file}) ann
            x{annotationToken = newAnn})
    tokens <- liftEither $ first (syntaxError file) $ tokenize s
    let newTokens = map assignFile tokens
    x <-lift $ catch (Right <$> (evaluate $ force $ isqv2 newTokens)) 
        (\e-> case e of
                x:_->(return $ Left $ GrammarError $ UnexpectedToken x)
                [] -> return $ Left $ GrammarError $ UnexpectedEOF)
    liftEither x

{-parseFile :: IO FilePath->String->IO (Either CompileError LAST)
parseFile path incpath = fmap (\ path -> do
    str <- readFile path
    runExceptT $ parseToAST str) path-}
    --flat_code <- parseIncFile path incpath
    --case flat_code of
    --    Left err -> return $ Left $ fromError err
    --    Right f -> runExceptT $ parseToAST f

{-parseStdin :: String->IO (Either CompileError LAST)
parseStdin incpath = do
    flat_code <- parseIncStdin incpath
    case flat_code of
        Left err -> return $ Left $ fromError err
        Right f -> runExceptT $ parseToAST f

parseFileOrStdin :: Maybe String->String-> IO (Either CompileError LAST)
parseFileOrStdin (Just x) incpath = parseFile x incpath
parseFileOrStdin Nothing incpath = parseStdin incpath-}


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

getConcatName :: [String] -> FilePath -> FilePath
getConcatName fields prefix= do
    let joined = joinPath ([prefix] ++ fields)
    addExtension joined ".isq"

getImportedFile :: [FilePath] -> [FilePath] -> LAST -> IO (Either CompileError FilePath)
getImportedFile froms incPath imp = do
    let dotName = importName imp
    let splited = splitOn "." dotName
    let names = map (getConcatName splited) incPath
    exist <- filterM doesFileExist names
    return $ case exist of
        [] -> Left $ GrammarError $ ImportNotFound dotName
        [x] -> do
            let index = findIndex (x ==) froms
            case index of
                Nothing -> Right x
                Just y -> Left $ GrammarError $ CyclicImport $ take (y + 1) froms
        (x:y:xs) -> Left $ GrammarError $ AmbiguousImport dotName x y

getImportedFiles :: [FilePath] -> [FilePath] -> [LAST] -> IO (Either CompileError [FilePath])
getImportedFiles froms incPath impList = do
    lis <- mapM (getImportedFile froms incPath) impList
    let errorLis = lefts lis
    case errorLis of
        [] -> return $ Right $ rights lis
        (x:xs) -> return $ Left x

data ImportEnv = ImportEnv {
    symbolTable :: Map.Map String SymbolTableLayer,
    ssaId :: Int
}

getImportedTcasts :: [FilePath] -> [FilePath] -> [LAST] -> StateT ImportEnv IO (Either CompileError ([TCAST], SymbolTableLayer))
getImportedTcasts froms incPath impList = do
    errOrFiles <- lift $ getImportedFiles froms incPath impList
    case errOrFiles of
        Left err -> return $ Left err
        Right files -> do
            errOrTcast <- mapM (fileToTcast incPath froms) files
            let errs = lefts errOrTcast
            case errs of
                [] -> do
                    let tupList = rights errOrTcast
                    let tcast = concat $ map fst tupList
                    let table = MultiMap.fromList $ concat $ map (MultiMap.toList . snd) tupList
                    return $ Right (tcast, table)
                (x:xs) -> return $ Left x

doImport :: [FilePath] -> [FilePath] -> FilePath -> LAST -> StateT ImportEnv IO (Either CompileError ([TCAST], SymbolTableLayer))
doImport incPath froms file node = do
    let pkg = package node
    let pathList = splitDirectories $ dropExtensions file
    let pacName = case pkg of {Nothing -> last pathList; Just pack -> packageName pack}
    let indices = findIndices (pacName ==) pathList
    case indices of
        [] -> return $ Left $ GrammarError $ BadPackageName pacName
        lis -> do
            let pacIndex = last lis
            let prefix = intercalate "." (drop pacIndex pathList) ++ "."
            let defList = defMemberList node
            let impList = importList node
            let isMain = null froms
            let rootPath = joinPath $ take pacIndex pathList
            let newIncPath = case isMain of {True -> rootPath:incPath; False -> incPath}
            case isMain || rootPath `elem` newIncPath of
                False -> return $ Left $ GrammarError $ InconsistentRoot file rootPath
                True -> do
                    let impNames = map importName impList
                    let groups = case impNames of
                            [] -> [[]]
                            _ -> groupBy (\x y -> x == y) impNames
                    let most = maximumBy (compare `on` length) groups
                    case length most of
                        i | i >= 2 -> return $ Left $ GrammarError $ DuplicatedImport $ head most
                        _ -> do
                            errOrTcasts <- getImportedTcasts (file:froms) newIncPath impList
                            case errOrTcasts of
                                Left x -> return $ Left x
                                Right (importTcast, importTable) -> do
                                    let errOrRaii = compileRAII defList
                                    case errOrRaii of
                                        Left x -> return $ Left $ fromError x
                                        Right raii -> do
                                            --liftIO $ BS.hPut stdout (encode raii) -- for debug
                                            oldId <- gets ssaId
                                            let errOrTuple = typeCheckTop isMain prefix raii importTable oldId
                                            case errOrTuple of
                                                Left x -> return $ Left $ fromError x
                                                Right (tcast, table, newId) -> do
                                                    modify' (\x->x{ssaId = newId})
                                                    return $ Right (tcast ++ importTcast, table)

fileToTcast :: [FilePath] -> [FilePath] -> FilePath -> StateT ImportEnv IO (Either CompileError ([TCAST], SymbolTableLayer))
fileToTcast incPath froms file = do
    stl <- gets symbolTable
    let maybeSymbol = Map.lookup file stl
    case maybeSymbol of
        Just symbol -> return $ Right ([], symbol)
        Nothing -> do
            excptionOrStr <- lift $ do try(readFile file) :: IO (Either SomeException String)
            case excptionOrStr of
                Left _ -> return $ Left $ GrammarError $ ReadFileError file
                Right str -> do
                    ast <- lift $ runExceptT $ parseToAST file str
                    case ast of
                        Left x -> return $ Left x
                        Right node -> do
                            errOrTuple <- doImport incPath froms file node
                            case errOrTuple of
                                Left err -> return $ Left err
                                Right tuple -> do
                                    let newStl = Map.insert file (snd tuple) stl
                                    modify' (\x->x{symbolTable = newStl})
                                    return $ Right tuple

generateTcast :: String -> FilePath -> IO (Either CompileError [TCAST])
generateTcast incPathStr inputFileName = do
    absolutPath <- canonicalizePath inputFileName
    let splitedPath = splitOn ":" incPathStr
    incPath <- mapM canonicalizePath splitedPath
    errOrTuple <- evalStateT (fileToTcast incPath [] absolutPath) $ ImportEnv Map.empty 0
    case errOrTuple of
        Left x -> return $ Left x
        Right tuple -> return $ Right $ fst tuple

compile :: String -> [TCAST] -> Either CompileError String
compile s ast_tc = runExcept $ do
    let mlir_module = generateMLIRModule s ast_tc
    return $ emitOp mlir_module


{-p :: Show a=>a->IO ()
p = pPrint

printRAII s = do
    Right f<-parseFile s ""
    return $ compileRAII f
printTC s = do
    Right f<-parseFile s ""
    return $ compileTypecheck f

genMLIR s = do
    Right f<-parseFile s ""
    case compile s f of
        Left x -> print x
        Right mlir -> putStrLn mlir

raiiMain :: IO ()
raiiMain = do
    (input:_) <- getArgs;
    ast<-printRAII input
    pPrintNoColor ast
    return ()-}