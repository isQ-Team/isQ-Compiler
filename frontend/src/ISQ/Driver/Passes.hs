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
import Data.List (filter, findIndex, findIndices, groupBy, intercalate, maximumBy, nub)
import Data.List.Split (splitOn)
import qualified Data.Map.Lazy as Map
import qualified Data.MultiMap as MultiMap
import System.IO
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
import System.Directory (canonicalizePath, doesFileExist)
import System.Environment (lookupEnv)
import System.FilePath (addExtension, dropExtensions, joinPath, splitDirectories)
import System.IO (stdout)

syntaxError :: FilePath -> String -> CompileError 
syntaxError file x = 
    let t1 = dropWhile (not . isDigit) x
        (l, t2) = span isDigit t1
        t3 = dropWhile (not . isDigit) t2
        (c, t4) = span isDigit t3
    in SyntaxError (Pos {line = read l, column = read c, filename = file} )

data ImportEnv = ImportEnv {
    globalTable :: SymbolTableLayer,
    symbolTable :: Map.Map String SymbolTableLayer,
    ssaId :: Int,
    qcis :: Bool,
    libraryPath:: [FilePath]
}

type PassMonad = ExceptT CompileError (StateT ImportEnv IO)

parseToAST :: FilePath -> String -> PassMonad LAST
parseToAST file s = do
    let assignFile = (\x -> do
            let ann = annotationToken x
            let newAnn = (\y -> y{filename=file}) ann
            x{annotationToken = newAnn})
    tokens <- liftEither $ first (syntaxError file) $ tokenize s
    let newTokens = map assignFile tokens
    x <- liftIO $ catch (Right <$> (evaluate $ force $ isqv2 newTokens)) 
        (\e-> case e of
                x:_->(return $ Left $ GrammarError $ UnexpectedToken x)
                [] -> return $ Left $ GrammarError $ UnexpectedEOF)
    liftEither x

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

processGlobal :: String -> Bool -> ([TCAST], SymbolTableLayer, Int)
processGlobal s qcis = do
    case tokenize s of
        Left _ -> error "Global code pass error"
        Right tokens -> do
            let ast = defMemberList $ isqv2 tokens
            case compileRAII ast of
                Left _ -> error "Global code RAII error"
                Right raii -> do
                    case typeCheckTop False "." raii MultiMap.empty 0 qcis of
                        Left x -> error $ "Global code type check error: " ++ (show $ encode x)
                        Right x -> x

getConcatName :: [String] -> FilePath -> FilePath
getConcatName fields prefix = do
    let joined = joinPath ([prefix] ++ fields)
    addExtension joined ".isq"

getImportedFile :: [FilePath] -> FilePath -> LAST -> PassMonad FilePath
getImportedFile froms rootPath imp = do
    let dotName = importName imp
    let splited = splitOn "." dotName
    libPath <- gets libraryPath
    let names = nub $ map (getConcatName splited) $ rootPath : libPath
    exist <- liftIO $ filterM doesFileExist names
    case exist of
        [] -> throwError $ GrammarError $ ImportNotFound dotName
        [x] -> do
            let index = findIndex (x ==) froms
            case index of
                Nothing -> return x
                Just y -> throwError $ GrammarError $ CyclicImport $ take (y + 1) froms
        (x:y:xs) -> throwError $ GrammarError $ AmbiguousImport dotName x y

getImportedFiles :: [FilePath] -> FilePath -> [LAST] -> PassMonad [FilePath]
getImportedFiles froms rootPath impList = mapM (getImportedFile froms rootPath) impList

getImportedTcasts :: [FilePath] -> FilePath -> [LAST] -> PassMonad ([TCAST], SymbolTableLayer)
getImportedTcasts froms rootPath impList = do
    files <- getImportedFiles froms rootPath impList
    tupList <- mapM (fileToTcast froms) files
    let tcast = concat $ map fst tupList
    gtable <- gets globalTable
    let tables = gtable : map snd tupList
    let table = MultiMap.fromList $ concat $ map MultiMap.toList tables
    return (tcast, table)

notMain :: LAST -> Bool
notMain (NProcedureWithDerive ann rttype name arg body derive) = name /= "main"
notMain _ = True

doImport :: [FilePath] -> FilePath -> LAST -> PassMonad ([TCAST], SymbolTableLayer)
doImport froms file node = do
    let pkg = package node
    let pathList = splitDirectories $ dropExtensions file
    let pacName = case pkg of {Nothing -> last pathList; Just pack -> packageName pack}
    let indices = findIndices (pacName ==) pathList
    case indices of
        [] -> throwError $ GrammarError $ BadPackageName pacName
        lis -> do
            let pacIndex = last lis
            let prefix = intercalate "." (drop pacIndex pathList) ++ "."
            let isMain = null froms
            let defList = case isMain of
                    True -> defMemberList node
                    False -> filter notMain $ defMemberList node
            let impList = importList node
            let rootPath = joinPath $ take pacIndex pathList
            let impNames = map importName impList
            let groups = case impNames of
                    [] -> [[]]
                    _ -> groupBy (\x y -> x == y) impNames
            let most = maximumBy (compare `on` length) groups
            case length most of
                i | i >= 2 -> throwError $ GrammarError $ DuplicatedImport $ head most
                _ -> do
                    (importTcast, importTable) <- getImportedTcasts (file:froms) rootPath impList
                    let errOrRaii = compileRAII defList
                    case errOrRaii of
                        Left x -> throwError $ fromError x
                        Right raii -> do
                            --liftIO $ BS.hPut stdout (encode raii) -- for debug
                            oldId <- gets ssaId
                            qcis <- gets qcis
                            let errOrTuple = typeCheckTop isMain prefix raii importTable oldId qcis
                            case errOrTuple of
                                Left x -> throwError $ fromError x
                                Right (tcast, table, newId) -> do
                                    modify' (\x->x{ssaId = newId})
                                    return (tcast ++ importTcast, table)

myReadFile :: FilePath -> IO (String)
myReadFile file = do
    inputHandle <- openFile file ReadMode 
    hSetEncoding inputHandle utf8
    theInput <- hGetContents inputHandle
    return theInput

fileToTcast :: [FilePath] -> FilePath -> PassMonad ([TCAST], SymbolTableLayer)
fileToTcast froms file = do
    stl <- gets symbolTable
    let maybeSymbol = Map.lookup file stl
    case maybeSymbol of
        Just symbol -> return ([], symbol)
        Nothing -> do
            exceptionOrStr <- liftIO $ do try(myReadFile file) :: IO (Either SomeException String)
            case exceptionOrStr of
                Left _ -> throwError $ GrammarError $ ReadFileError file
                Right str -> do
                    ast <- parseToAST file str
                    tuple <- doImport froms file ast
                    stl' <- gets symbolTable
                    let newStl = Map.insert file (snd tuple) stl'
                    modify' (\x->x{symbolTable = newStl})
                    return $ tuple

globalSource :: String
globalSource = "int __measure_bundle(qbit q[])\
\{\
\    int res = 0;\
\    int i = q.length - 1;\
\    while (i >= 0) {\
\        res = res * 2 + M(q[i]);\
\        i = i - 1;\
\    }\
\    return res;\
\}"

globalSourceQcis :: String
globalSourceQcis = "int __measure_bundle(qbit q[])\
\{\
\    for i in 0 : q.length {\
\        M(q[i]);\
\    }\
\    return 0;\
\}"

globalQmpiSource :: String
globalQmpiSource = "int __qmpi_rank;\
\int qmpi_comm_rank()\
\{\
\    return __qmpi_rank;\
\}"

generateTcast :: String -> FilePath -> Bool -> IO (Either CompileError ([TCAST], Int))
generateTcast incPathStr inputFileName qcis = do
    let gc = case qcis of
            True -> globalSourceQcis
            False -> globalSource
    let (globalTcasts, globalTable, ssaId) = processGlobal (gc ++ globalQmpiSource) qcis
    absolutPath <- canonicalizePath inputFileName
    let splitedPath = splitOn ":" incPathStr
    incPath <- mapM canonicalizePath splitedPath
    let env = ImportEnv globalTable Map.empty ssaId qcis incPath
    (errOrTuple, (ImportEnv _ _ ssa _ _)) <- runStateT (runExceptT $ fileToTcast [] absolutPath) env
    case errOrTuple of
        Left x -> return $ Left x
        Right tuple -> return $ Right (globalTcasts ++ fst tuple, ssa)

compile :: String -> ([TCAST], Int) -> Either CompileError String
compile s ast_tc = runExcept $ do
    let mlir_module = generateMLIRModule s ast_tc
    return $ emitOp mlir_module

