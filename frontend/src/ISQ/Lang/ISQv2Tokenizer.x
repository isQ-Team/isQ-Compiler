{
{-# LANGUAGE FlexibleInstances #-}
module ISQ.Lang.ISQv2Tokenizer where
import Control.Exception (Exception)
}

%wrapper "monadUserState"

$digit = 0-9
$alpha = [a-zA-Z]

$idfirstchar  = [$alpha \_ \@]
$idrestchar   = [$alpha $digit \_]

@ident = $idfirstchar $idrestchar*

@decimal = $digit+
@exponent = [eE] [\-\+] @decimal

@reservedid = 
	if|else|for|in|while|procedure|int|qbit|measure|print|defgate|pass|return|
    ctrl|nctrl|inv|bool|true|false|let|const|unit|M|break|continue|double|as|extern|gate|deriving|oracle|pi
@reservedop = "|0>"|"=="|"="|"+"|"-"|"*"|"/"|"<"|">"|"<="|">="|"!="|"&&"|"||"|"!"|"%"|
              ","|"("|")"|"{"|"}"|"["|"]"|"."|":"|";"|"->"

tokens :-
    <0> $white+ {skip}
    <0, commentSC> "/*" {beginComment}
    <commentSC> "*/" {endComment}
    <commentSC> [.\n] {skip}
    <0> \" {beginString}
    <stringSC> \" {endString}
    <stringSC> . {appendString}
    <stringSC> \\[\nt\"] {escapeString}
    <0> "//".* {skip}
    <0> @reservedid {tokenReservedId}
    <0> @reservedop {tokenReservedOp}
    <0> @decimal { tokenNatural }
    <0> @decimal \. @decimal @exponent?
      | @decimal @exponent  { tokenFloat }
    <0> @decimal [ij]
      | @decimal \. @decimal @exponent? [ij]
      | @decimal @exponent j  { tokenImagPart }
    <0> @ident {tokenIdent}


{
data Token ann = 
    TokenReservedId {annotationToken :: ann, tokenReservedIdV :: String}
  | TokenReservedOp {annotationToken :: ann, tokenReservedOpV :: String} | TokenNatural {annotationToken :: ann, tokenNaturalV :: Int}
  | TokenFloat {annotationToken :: ann, tokenFloatV :: Double}  | TokenImagPart {annotationToken :: ann, tokenImagPartV :: Double}  | TokenIdent {annotationToken :: ann, tokenIdentV :: String} | TokenEOF {annotationToken :: ann} | TokenStringLit {annotationToken :: ann, tokenStringLitV :: String} deriving Show
data Pos = Pos {line :: Int, column :: Int} deriving Show
type ISQv2Token = Token Pos
instance Exception [Token Pos]
class Annotated x where
  annotation :: x ann->ann

data AlexUserState = AlexUserState{
  stringBuffer :: String,
  stringPos :: Pos,
  commentNestLevel :: Int
} deriving Show

alexInitUserState = AlexUserState "" undefined 0

get = Alex (\s->Right (s, alex_ust s))
put x = Alex (\s->Right (s{alex_ust = x}, ()))

instance Annotated Token where
  annotation = annotationToken

updateCommentStatus = do
  s<-get
  alexSetStartCode $ if (commentNestLevel s)==0 then 0 else commentSC

beginComment _ _ = do
  s<-get
  put s{commentNestLevel = (commentNestLevel s)+1}
  updateCommentStatus
  return Nothing
endComment _ _ = do
  s<-get
  put s{commentNestLevel = (commentNestLevel s)-1}
  updateCommentStatus
  return Nothing

beginString (pos, _, _, _) _ = do
  s<-get
  put s{stringBuffer = "", stringPos = position pos}
  alexSetStartCode stringSC
  return Nothing
appendString (_, _, _, c:_) _ = do
  s<-get
  put s {stringBuffer = c:(stringBuffer s)}
  return Nothing
escapeString (_, _, _, _:c:_) _ = do
  let unesc =
        case c of
          'n' -> '\n'
          't' -> '\t'
          '"' -> '"'
  s <- get
  put s{stringBuffer = unesc:(stringBuffer s)}
  return Nothing
endString _ _ = do
  s<-get
  let buf = stringBuffer s
  put s{stringBuffer = ""}
  alexSetStartCode 0
  return $ Just $ TokenStringLit (stringPos s) $ reverse buf
position (AlexPn _ line column) = Pos line column
tokenReservedId (pos, _, _, str) len = return $ Just $ TokenReservedId (position pos) (take len str)
tokenReservedOp (pos, _, _, str) len = return $ Just $ TokenReservedOp (position pos) (take len str)
tokenIdent (pos, _, _, str) len = return $ Just $ TokenIdent (position pos) (take len str)
tokenFloat (pos, _, _, str) len = return $ Just $ TokenFloat (position pos) (read $ take len str)
tokenImagPart (pos, _, _, str) len = return $ Just $ TokenImagPart (position pos) (read $ take (len-1) str)
tokenNatural (pos, _, _, str) len = return $ Just $ TokenNatural (position pos) (read $ take len str)
alexEOF = return $ Just $ TokenEOF undefined

tokenize str = runAlex str $ do
  let step = do
        tok<-alexMonadScan
        case tok of
          (Just (TokenEOF _)) -> return []
          Nothing -> step
          (Just tok) -> do
              rest<-step
              return $ tok:rest
  step

}