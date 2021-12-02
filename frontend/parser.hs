{-# LANGUAGE FlexibleContexts, DeriveFunctor, NoMonomorphismRestriction #-}
module ISQ.Lang where
import Text.Parsec
import qualified Text.Parsec.Token as P
import qualified Text.Parsec.Language as L
import Text.Parsec.Pos
import Data.Maybe
import Data.Complex
import Data.Functor
import Data.Functor.Identity
import Data.List (foldl')


isqTokenRules = L.javaStyle{
    P.reservedNames = ["if", "then", "else", "fi", "for", "to", "while", "do", "od", "procedure", "int", "qbit", "M", "print", "Defgate", "pass", "return", "Ctrl", "NCtrl", "Inv", "|0>"],
    P.reservedOpNames = ["=", "+", "-", "*", "/", "<", ">", ",", "(", ")", "[", "]", 
    "{", "}", 
    ";", "==", "<=", ">=", "!="]
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

data AST a = AST {
    location :: SourcePos,
    value :: a
} deriving (Show, Functor)
data Matrix = Matrix [[AST Expr]] deriving Show
data Ident = Ident {identName :: String} deriving Show
data GateDef = GateDef {
    gateName :: AST Ident,
    matrix :: AST Matrix
} deriving Show
data UnitKind = Int | Qbit deriving Show
data VarType = UnitType {ty :: AST UnitKind} 
             | Composite {base :: AST UnitKind, len :: Maybe Int} deriving Show

data VarDef = VarDef{
    varType :: AST VarType,
    varName :: AST Ident
} deriving Show

data ProcDef = ProcDef{
    returnType :: Maybe (AST UnitKind),
    procName :: AST Ident,
    parameters :: [AST VarDef],
    body :: [AST Statement]
} deriving Show

data LeftValueExpr = ArrayRef {
    arrayRefName :: AST Ident,
    offset :: AST Expr
} | VarRef {
    varRefName :: AST Ident
} deriving Show
data BinaryOp = Add | Sub | Mul | Div | Less | Equal | NEqual | Greater | LessEqual | GreaterEqual deriving Show
data UnaryOp = Positive | Neg deriving Show
data ProcedureCall = ProcedureCall{
    calleeName :: AST Ident,
    callingArgs :: [AST Expr]
} deriving Show
data Expr = ImmInt Int
    | ImmComplex (Complex Double)
    | BinaryOp {binaryOpType :: AST BinaryOp, binaryLhs :: AST Expr, binaryRhs :: AST Expr}
    | UnaryOp {unaryOpType :: AST UnaryOp, unaryVal :: AST Expr}
    | LeftExpr {leftRef :: AST LeftValueExpr}
    | MeasureExpr {measureRef :: AST LeftValueExpr} 
    | CallExpr {exprCalling :: AST ProcedureCall} deriving Show
data GateDecorator = GateDecorator {controlStates :: [Bool], adjoint :: Bool} deriving Show
data Statement = 
      QbitInitStmt { qubitOperand :: AST LeftValueExpr }
    | QbitGateStmt { decorator :: AST GateDecorator, qubitOperands :: [AST LeftValueExpr], appliedGateName :: AST Ident}
    | CintAssignStmt { assignLhs :: AST LeftValueExpr, assignRhs :: AST Expr}
    | IfStatement { ifCondition :: AST Expr, thenBranch :: [AST Statement], elseBranch :: [AST Statement]}
    | ForStatement { forIdent :: AST Ident, lo :: AST Expr, hi :: AST Expr, forBody :: [AST Statement]}
    | WhileStatement {whileCondition :: AST Expr, whileBody :: [AST Statement]}
    | PrintStatement {printedExpr :: AST Expr}
    | PassStatement
    | ReturnStatement {returnedExpr :: AST Expr}
    | CallStatement {callExpr :: AST ProcedureCall}
    | VarDefStatement {localVarDef :: [AST VarDef]}
    deriving Show

data Program = Program {
    topGatedefs :: [AST GateDef],
    topVardefs :: [AST [AST VarDef]],
    procedures :: [AST ProcDef]} deriving Show

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
here = getPosition <&> (\loc p->AST loc p)
locStart parser = lexeme $ do {p<-here; v<-parser; return $ p v}

ident = locStart $ (lexeme $ P.identifier isqTokenizer) <&> Ident
termImmImag = locStart $ imagNumber <&> ImmComplex
termImmComplex = locStart $ complexFloatPart <&> ImmComplex
termImmInt = locStart $ integer <&> ImmInt
termParens = parens parseExpr
termSubscript = brackets parseExpr
termIdent = locStart $ ident <&> (\i->LeftExpr (AST (location i) $ VarRef i))
unaryopLevel2 = locStart $ (op "+" >> return Positive) <|> (op "-" >> return Neg)
opLevel3 = locStart $ (op "*" >> return Mul) <|> (op "/" >> return Div)
opLevel4 = locStart $ (op "+" >> return Add) <|> (op "-" >> return Sub)
opLevel6 = locStart $ (op ">" >> return Greater) <|> (op "<" >> return Less)
        <|>(op ">=" >> return GreaterEqual) <|> (op "<=" >> return LessEqual)
opLevel7 = locStart $ (op "==" >> return Equal) <|> (op "!=" >> return NEqual)

-- AST Ident->AST Expr
--level1Suffix :: (Monad m)=>ParsecT [Char] u m (AST Ident->AST Expr)
leftTermSuffix = (do {p<-getPosition; subscript<-brackets parseExpr; return (\e->(AST (location e) $ LeftExpr $ AST (location e) $ ArrayRef e subscript))}) 
    <|> (do {return (\e->AST (location e) $ LeftExpr $ AST (location e) $ VarRef e);})
callSuffix = (do {p<-getPosition; args<-parens (commaSep parseExpr); return (\e->(AST (location e) $ CallExpr $ AST (location e) $ ProcedureCall e args))})
level1Suffix =  callSuffix <|> leftTermSuffix

termLeft = do {i<-ident; smap<-leftTermSuffix; return $ smap i}
leftValue = termLeft <&> (leftRef . value)
termComposite = do {i<-ident; smap<-level1Suffix; return $ smap i}
term = (try termImmImag) <|> (try termImmComplex) <|> termImmInt <|> termParens <|> termComposite


--parseExprLevel2 :: ParsecT [Char] u m (AST Expr)
parseExprLevel2 = locStart $ do {u<-optionMaybe unaryopLevel2; e<-term; return $ case u of {Just ty-> UnaryOp ty e; Nothing->value e}}
--parseExprLevel :: (Monad m)=>ParsecT [Char] u m (AST BinaryOp)->ParsecT [Char] u m (AST Expr)->ParsecT [Char] u m (AST Expr)
parseExprLevel level previous = do {first<-previous; remains<-many (do {o<-level; v<-previous; return (o, v)}); return $ foldl' (\l (o, r)->AST (location l) (BinaryOp o l r)) first remains}
--parseExprLevel3 :: ParsecT [Char] u m (AST Expr)
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
parseDecorator = (do {tok "Ctrl"; n<-parens integer; return (\x->x{controlStates = (replicate n True) ++ (controlStates x)})}) <|> (do {tok "NCtrl"; n<-parens integer; return (\x->x{controlStates = (replicate n False) ++ (controlStates x)})}) <|> (do {tok "Adj"; return (\x->x{adjoint = not (adjoint x)})})
parseDecorators = locStart $ do {xs<-many parseDecorator; return $ foldr ($) (GateDecorator [] False) xs}

qubitUnitaryStatement = locStart $ do {deco<-parseDecorators; name<-ident; xs<-angles $ commaSep leftValue; return $ QbitGateStmt deco xs name}

cintAssignStatement = locStart $ do {lhs<-leftValue; op "="; rhs<-parseExpr; return $ CintAssignStmt lhs rhs}
ifStatement = locStart $ do {tok "if"; cond<-parseExpr;  thenStmt<-braces statementBlock;  elseStmt<-braces statementBlock; return $ IfStatement cond thenStmt elseStmt}

callStatement = locStart $ do {p<-getPosition; name<-ident; args<-parens (commaSep parseExpr); return $ CallStatement $ AST p $ ProcedureCall name args}

whileStatement = locStart $ do {tok "while"; cond<-parens parseExpr; stmt<-braces statementBlock; return $ WhileStatement cond stmt}

forStatement = locStart $ do {tok "for"; name<-ident; op "="; start<-parseExpr; tok "to"; end<-parseExpr; stmt<-braces statementBlock; return $ ForStatement name start end stmt}

printStatement = locStart $ do {tok "print"; x<-parseExpr; return $ PrintStatement x}
passStatement = locStart $ do {tok "pass"; return PassStatement}

baseUnitKind = locStart $ (tok "qbit" >> return Qbit) <|> (tok "int" >> return Int)

singleDefVar base = locStart $ do {name<-ident; subscript<-optionMaybe (brackets (optionMaybe integer)); case subscript of {Just s->return $ VarDef (AST (location base) $ Composite base s) name; Nothing->return $ VarDef (AST (location base) $ UnitType base) name}}

defStatement = locStart $ do {baseType <- baseUnitKind; vars<-commaSep (singleDefVar baseType); return $ VarDefStatement vars}

oneDefStatement = do {baseType <- baseUnitKind; var<-singleDefVar baseType; return $ var}


statement = ifStatement <|> callStatement <|> printStatement <|> passStatement <|> forStatement <|> whileStatement <|> defStatement <|> try qubitUnitaryStatement <|> try cintAssignStatement <|> try qubitInitStatement

statementBlock = many (do {st<-statement; op ";"; return st})

parseProcedure = locStart $ do {ty<-(baseUnitKind <&> Just) <|> (tok "procedure" >> (return Nothing)); name<-ident; args<-parens (commaSep oneDefStatement); stmt<-braces statementBlock; return $ ProcDef ty  name args stmt}

gatedefStatement = locStart $ do {tok "Defgate"; name<-ident; op "="; mat<-brackets parseMatrix; op ";"; return $ GateDef name mat}
parseMatrix = locStart $ do {mat<-semiSep (commaSep parseExpr); return $ Matrix mat}

parseISQ = do {gates<-many gatedefStatement; top_var_defs<-many $ do {st<-defStatement; op ";"; return $ AST (location st) $ (localVarDef $ value st)}; proc_defs<-many parseProcedure; return $ Program gates top_var_defs proc_defs}
tparse a b = runParser a () "" b
