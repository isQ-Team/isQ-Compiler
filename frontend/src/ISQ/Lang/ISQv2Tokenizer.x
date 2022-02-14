{
module ISQ.Lang.ISQv2Tokenizer where
}

%wrapper "monad"

$digit = 0-9
$alpha = [a-zA-Z]

$idfirstchar  = [$alpha \_]
$idrestchar   = [$alpha $digit \_]

@ident = $idfirstchar $idrestchar*

@decimal = $digit+
@exponent = [eE] [\-\+] @decimal

@reservedid = 
	if|else|for|in|while|procedure|int|qbit|measure|print|defgate|pass|return|
    ctrl|nctrl|inv
@reservedop = "|0>"|"=="|"="|"+"|"-"|"*"|"/"|"<"|">"|"<="|">="|"!="|"&&"|"||"|"!"|
              ","|"("|")"|"{"|"}"|"["|"]"|"."|":"

tokens :-
    <0> $white+ {skip}
    <0, commentSC> "/*" {beginComment}
    <commentSC> "*/" {endComment}
    <commentSC> [.\n] {skip}
    <0> "//".* {skip}
    <0> @reservedid {tokenReservedId}
    <0> @reservedop {tokenReservedOp}
    <0> @decimal { tokenNatural }
    <0> @decimal \. @decimal @exponent?
      | @decimal @exponent  { tokenFloat }
    <0> @decimal j
      | @decimal \. @decimal @exponent? j
      | @decimal @exponent j  { tokenImagPart }
    <0> @ident {tokenIdent}


{
data Token ann = 
    TokenReservedId {annotationToken :: ann, tokenReservedIdV :: String}
  | TokenReservedOp {annotationToken :: ann, tokenReservedOpV :: String} | TokenNatural {annotationToken :: ann, tokenNaturalV :: Int}
  | TokenFloat {annotationToken :: ann, tokenFloatV :: Double}  | TokenImagPart {annotationToken :: ann, tokenImagPartV :: Double}  | TokenIdent {annotationToken :: ann, tokenIdentV :: String} | TokenEOF {annotationToken :: ann} deriving Show
data Pos = Pos {line :: Int, column :: Int} deriving Show
type ISQv2Token = Token Pos

class Annotated x where
  annotation :: x ann->ann

instance Annotated Token where
  annotation = annotationToken
beginComment _ _ = alexSetStartCode commentSC >> return Nothing
endComment _ _ =alexSetStartCode 0 >> return Nothing
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