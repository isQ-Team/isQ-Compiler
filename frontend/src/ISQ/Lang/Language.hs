module ISQ.Lang.Language where
import Text.Parsec
import qualified Text.Parsec.Token as P
import qualified Text.Parsec.Language as L
import Data.Maybe
import Data.Complex
import Data.List (foldl')
import ISQ.Lang.AST
import Data.Functor.Identity
import Text.Parsec.Pos
import Data.Functor
import Control.Lens hiding (op)

--import GHC.Data.Maybe
isqTokenRules = L.javaStyle{
    P.reservedNames = ["if", "then", "else", "fi", "for", "to", "in", "while", "procedure", "int", "qbit", "M", "print", "Defgate", "pass", "return", "Ctrl", "NCtrl", "Inv", "|0>"],
    P.reservedOpNames = ["=", "+", "-", "*", "/", "<", ">", ",", "(", ")", "[", "]",
    "{", "}",
    ";", "==", "<=", ">=", "!="],
    P.caseSensitive = True
}

isqTokenizer = P.makeTokenParser isqTokenRules

tokNumber = do
    x<-P.naturalOrFloat isqTokenizer
    return $ case x of
        Left i -> fromInteger i
        Right f -> f
imagNumber = do
    x<-tokNumber
    char 'i' <|> char 'j'
    return (0:+x)
complexPart = do
    x<-P.float isqTokenizer
    has_i<-optionMaybe (char 'i' <|> char 'j')
    if isJust has_i then return (0:+x) else return (x:+0)

lexeme = P.lexeme isqTokenizer
tok t = lexeme $ P.reserved isqTokenizer t
op o = lexeme $ P.reservedOp isqTokenizer o
integer = lexeme $ P.integer isqTokenizer <&> fromInteger
parens = P.parens isqTokenizer
brackets = P.brackets isqTokenizer
braces = P.braces isqTokenizer
angles = P.angles isqTokenizer
commaSep = P.commaSep isqTokenizer
semiSep = P.semiSep isqTokenizer
complexFloatPart = lexeme complexPart
ket0 = tok "|0>"

locStart :: ParsecT String u Identity (SourcePos -> a) -> ParsecT String u Identity a
locStart parser = lexeme $ do {p<-getPosition; v<-parser; return $ v p }

ident = locStart $ lexeme (P.identifier isqTokenizer) <&> Ident
termImmImag = locStart $ imagNumber <&> ImmComplex
termImmComplex = locStart $ complexFloatPart <&> ImmComplex
termImmInt = locStart $ integer <&> ImmInt
termParens = parens parseExpr
--termSubscript = brackets parseExpr
--termIdent = locStart $ ident <&> (\i->LeftExpr (AST (location i) $ VarRef i))
termMeasure = locStart $ do {tok "M"; r<-angles leftValue; return $ MeasureExpr r}
unaryopLevel2 = (op "+" >> return Positive) <|> (op "-" >> return Neg)
opLevel3 = (op "*" >> return Mul) <|> (op "/" >> return Div)
opLevel4 = (op "+" >> return Add) <|> (op "-" >> return Sub)
opLevel6 = (op ">" >> return Greater) <|> (op "<" >> return Less)
        <|>(op ">=" >> return GreaterEqual) <|> (op "<=" >> return LessEqual)
opLevel7 = (op "==" >> return Equal) <|> (op "!=" >> return NEqual)

-- AST Ident->AST Expr
--level1Suffix :: (Monad m)=>ParsecT [Char] u m (AST Ident->AST Expr)
leftTermSuffix :: ParsecT
  String u Identity (Ident SourcePos -> LeftValueExpr SourcePos)
leftTermSuffix = (do {p<-getPosition; subscript<-brackets parseExpr; return (\e->ArrayRef e subscript (view annotation e))})
    <|> return (\e-> VarRef e (view annotation e))

callSuffix = do {p<-getPosition; args<-parens (commaSep parseExpr); return (\e-> CallExpr (ProcedureCall e args (view annotation e))  (view annotation e))}

level1Suffix =  callSuffix <|> (leftTermSuffix <&> (\e pos->let lexpr = e pos in LeftExpr lexpr (view annotation pos)))

leftValue :: ParsecT String u Identity (LeftValueExpr SourcePos)
leftValue = do {i<-ident; smap<-leftTermSuffix; return $ smap i}
termLeft :: ParsecT String u Identity (Expr SourcePos)
termLeft = leftValue <&> (\e->LeftExpr e (view annotation e))
termComposite :: ParsecT String u Identity (Expr Pos)
termComposite = do {i<-ident; smap<-level1Suffix; return $ smap i}
term = try termImmImag <|> try termImmComplex <|> termImmInt <|> termParens <|> termMeasure <|> termComposite


--parseExprLevel2 :: ParsecT [Char] u m (AST Expr)
parseExprLevel2 = locStart $ do {u<-optionMaybe unaryopLevel2; e<-term; return $ case u of {Just ty-> UnaryOp ty e; Nothing-> flip (set annotation) e}}
--parseExprLevel :: (Monad m)=>ParsecT [Char] u m (AST BinaryOp)->ParsecT [Char] u m (AST Expr)->ParsecT [Char] u m (AST Expr)
parseExprLevel level previous = do {first<-previous; remains<-many (do {o<-level; v<-previous; return (o, v)}); return $ foldl' (\l (o, r)->BinaryOp o l r (view annotation l)) first remains}
--parseExprLevel3 :: ParsecT [Char] u m (AST Expr)
parseExprLevel3 :: ParsecT String u Identity (Expr SourcePos)
parseExprLevel3 = parseExprLevel opLevel3 parseExprLevel2
--parseExprLevel6 :: ParsecT [Char] u m (AST Expr)
parseExprLevel4 = parseExprLevel opLevel4 parseExprLevel3
--parseExprLevel4 :: ParsecT [Char] u m (AST Expr)
parseExprLevel6 = parseExprLevel opLevel6 parseExprLevel4
--parseExprLevel7 :: ParsecT [Char] u m (AST Expr)
parseExprLevel7 = parseExprLevel opLevel7 parseExprLevel6
--parseExpr :: (Monad m)=>ParsecT [Char] u m (AST Expr)
parseExpr = parseExprLevel7


qubitInitStatement = locStart $ do {lhs<-leftValue; op "="; ket0; return $ QbitInitStmt lhs}
controlSize = parens integer <|> return 1
parseDecorator = (do {tok "Ctrl"; n<-controlSize; return (\x->x{_controlStates = replicate n True ++ _controlStates x})}) <|> (do {tok "NCtrl"; n<-controlSize; return (\x->x{_controlStates = replicate n False ++ _controlStates x})}) <|> (do {tok "Inv"; return (\x->x{_adjoint = not (_adjoint x)})})
parseDecorators = do {xs<-many parseDecorator; return $ foldr ($) (GateDecorator [] False) xs}

qubitUnitaryStatement = locStart $ do {deco<-parseDecorators; name<-ident; xs<-angles $ commaSep leftValue; return $ QbitGateStmt deco xs name}

cintAssignStatement = locStart $ do {lhs<-leftValue; op "="; CintAssignStmt lhs <$> parseExpr;}
elseStmtBranch = (do {tok "else"; region}) <|> return []
ifStatement = locStart $ do {tok "if"; cond<-parseExpr;  thenStmt<-region; IfStatement cond thenStmt <$> elseStmtBranch;}

callStatement = locStart $ do {p<-getPosition; name<-ident; args<-parens (commaSep parseExpr); return $ CallStatement (ProcedureCall name args p)}

whileStatement = locStart $ do {tok "while"; cond<-parens parseExpr; WhileStatement cond <$> region;}

--forStatement = locStart $ do {tok "for"; op "("; name<-ident; op "="; start<-parseExpr; tok "to"; end<-parseExpr; op ")"; ForStatement name start end <$> region;}

rangeExpr = do {
    start<-parseExpr; op ":"; end<-parseExpr; p<-getPosition; step<-optionMaybe (do {op ":"; P.natural isqTokenizer});
    return (start, end, fromMaybe 1 step)
}
bundleExpr = sepBy1 parseExpr (op ",")

forStatement = locStart $ do {tok "for"; name<-ident; tok "in";  (start, end, step)<-rangeExpr; ForStatement name start end (fromIntegral step) <$> region;}

printStatement = locStart $ do {tok "print"; PrintStatement <$> parseExpr;}
returnStatement = locStart $ do {tok "return"; expr<-optionMaybe parseExpr; return $ ReturnStatement expr}
passStatement = locStart $ do {tok "pass"; return PassStatement}

baseUnitKind = (tok "qbit" >> return Qbit) <|> (tok "int" >> return Int)

singleDefVar base = locStart $ do {name<-ident; subscript<-optionMaybe (brackets (optionMaybe integer)); case subscript of {Just s->return $ \loc->VarDef (Composite base s loc) name loc; Nothing->return $ \loc->VarDef (UnitType base loc) name loc}}

defStatement = locStart $ do {baseType <- baseUnitKind; vars<-commaSep (singleDefVar baseType); return $ VarDefStatement vars}

oneDefStatement = do {baseType <- baseUnitKind; singleDefVar baseType;}

semicolonStmt stmt = do {st<-stmt; op ";"; return st}

statement = ifStatement <|> forStatement <|> whileStatement <|>  semicolonStmt (printStatement <|> returnStatement <|> passStatement <|> defStatement <|> try qubitUnitaryStatement <|> try cintAssignStatement <|> try qubitInitStatement <|> try callStatement)

statementBlock = many statement

region = braces statementBlock <|> (do {st<-statement; return [st]})

parseProcedure = locStart $ do {pos<-getPosition; ty<-(tok "int" >> return Int) <|> (tok "procedure" >> return Void); name<-ident; args<-parens (commaSep oneDefStatement); stmt<-braces statementBlock; return $ ProcDef ty name args stmt}

gatedefStatement = locStart $ do {tok "Defgate"; name<-ident; op "="; mat<-brackets parseMatrix; op ";"; return $ GateDef name mat}
parseMatrix = locStart $ do {mat<-semiSep (commaSep parseExpr); return $ Matrix mat}

parseISQ = locStart $ do {gates<-many gatedefStatement; top_var_defs<-many $ try $ do {st<-defStatement; op ";"; return $ _localVarDef st}; proc_defs<-many parseProcedure; return $ Program gates (concat top_var_defs) proc_defs}
--tparse a b = runParser a () "" b