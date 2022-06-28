module ISQ.Lang.FlatInc where
import Data.List.Split
import System.Directory
import System.Environment
import Control.Exception
import Control.Monad.State.Lazy
import qualified Data.Set as Set
import Debug.Trace

data IncFileError = FileOpenError {incpath::String, filename::String, line::Int, column::Int} deriving Show

type FileList = Set.Set String

data FileEnv = FileEnv {
    absDir :: String,
    fileTable :: FileList,
    nowFile :: String,
    row :: Int,
    incPath:: [String]
}

type FileState = StateT FileEnv

getCurrentDir :: String -> String
getCurrentDir s = concatMap (\x->"/"++x) $ tail $ init $ splitOn "/" s

isIncFile :: String -> Bool
isIncFile s = let sl = splitOn "\"" s in if (take 9 s) == "#include " && (length sl) > 2 then True else False

getIncFile :: String -> String
getIncFile s = (splitOn "\"" s)!!1

addLineInfo :: Int -> String -> String
addLineInfo l f = "--"++(show l)++" "++f++"--"

--find include file in context and pass them
flatIncFile :: [(String, Int)] -> FileState IO (Either IncFileError String)
flatIncFile [] = return $ Right ""
flatIncFile [(s, r)] = do
    nf <- gets nowFile
    abs_root <- gets absDir
    if isIncFile s then do
        modify' (\x -> x{row=r})
        f <- parseFile $ getIncFile s
        modify' (\x -> x{nowFile=nf, absDir = abs_root})
        case f of
            Left err -> return $ Left err
            Right val -> do
                let new_val = val ++ "\n" ++ (addLineInfo (r+1) nf)
                return $ Right new_val
    else return $ Right s
flatIncFile ((s,r):y) = do
    l <- flatIncFile [(s, r)]
    case l of
        Left err -> return $ Left err
        Right lval -> do
            r <- flatIncFile y
            case r of
                Left err -> return $ Left err
                Right rval -> return $ Right (lval ++"\n"++ rval)

-- file in include directory
getAbsFilePathFromSysPath :: String->[String]->IO (Either String String)
getAbsFilePathFromSysPath path [] = return $ Left ""
getAbsFilePathFromSysPath path (root:res) = do
    if root == "" then getAbsFilePathFromSysPath path res else do
        abs_path <- canonicalizePath (root ++ "/" ++ path)
        file_exit <- doesFileExist abs_path
        if file_exit then return $ Right abs_path
        else
            getAbsFilePathFromSysPath path res

-- file in current directory or ...
getAbsFilePath :: String->String->[String]-> IO (Either String String)
getAbsFilePath path abs_root inc_path_list = do
    file_path <- canonicalizePath ( if (path !! 0 == '/' || abs_root == "" )  then path else (abs_root ++ "/" ++ path))
    file_exit <- doesFileExist file_path
    if file_exit then
        return $ Right file_path
    else if path !! 0 == '/' then return $ Left "" else getAbsFilePathFromSysPath path inc_path_list

-- get include file context
parseFile :: String -> FileState IO (Either IncFileError String)
parseFile path = do
    nf <- gets nowFile
    row <- gets row
    ft <- gets fileTable
    abs_root <- gets absDir
    inc_path <- gets incPath

    abs_file_path <- lift $ do 
        getAbsFilePath path abs_root inc_path
        
    case abs_file_path of
        Right abs_path -> do
            if Set.member abs_path ft then return $ Right "\n"
            else do
                ctx <- lift $ do try(readFile abs_path) :: IO (Either SomeException String)
                case ctx of
                    Left err -> return $ Left FileOpenError{incpath=path, filename=nf, line=row, column=1}
                    Right val -> do
                        let new_val = (addLineInfo 1 abs_path) ++ "\n" ++ val
                        modify' (\x -> x{nowFile = abs_path, absDir = getCurrentDir abs_path, fileTable = Set.insert abs_path ft})
                        flatIncFile $ zip (init $ splitOn "\n" new_val) [0..]

        Left _ -> return $ Left FileOpenError{incpath=path, filename=nf, line=row, column=1}

-- get defualt include directory
getSysEnv :: IO (String)
getSysEnv = do
    res <- lookupEnv "ISQ_INCLUDE"
    case res of
        Just x -> return x
        Nothing -> return ""

parseIncStdin :: String->IO (Either IncFileError String)
parseIncStdin incpath = do
    s <- getContents
    dir <- getCurrentDirectory
    sys_path <- getSysEnv
    evalStateT (flatIncFile $ zip (splitOn "\n" ((addLineInfo 1 "stdin") ++ "\n" ++ s)) [0..]) FileEnv{absDir=dir, fileTable=Set.empty, nowFile="stdin", row=0, incPath=(splitOn ":" (incpath++":"++sys_path))}
    

parseIncFile :: String->String -> IO (Either IncFileError String)
parseIncFile path incpath = do
    sys_path <- getSysEnv
    evalStateT (parseFile path) FileEnv{absDir="", fileTable=Set.empty, nowFile="", row=0, incPath=(splitOn ":" (incpath++":"++sys_path))}