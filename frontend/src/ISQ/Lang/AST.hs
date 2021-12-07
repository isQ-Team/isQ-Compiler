{-# LANGUAGE TemplateHaskell #-}
module ISQ.Lang.AST where
import Control.Lens
import Data.Complex
import Text.Parsec.Pos (SourcePos)

class Annotated x where
    annotation :: Lens (x s) (x s) s s

data Matrix ann = Matrix {_matrixData :: [[Expr ann]], _matrixAnnotation :: ann} deriving Show


data Ident ann = Ident {_identName :: String, _identAnnotation :: ann} deriving Show
data GateDef ann = GateDef {
    _gateName :: Ident ann,
    _matrix :: Matrix ann,
    _gatedefAnnotation :: ann
} deriving Show

gateSize :: GateDef ann -> Int
gateSize = floor . sqrt . fromIntegral . length . _matrixData . _matrix
data UnitKind = Int | Qbit | Complex | Void deriving (Show, Eq)
data VarType ann = UnitType {_ty :: UnitKind, _vartypeAnnotation :: ann}
             | Composite {_base :: UnitKind, _arrLen :: Maybe Int, _vartypeAnnotation :: ann} 
             | Proc {_vartypeAnnotation :: ann}
            deriving Show

data VarDef ann = VarDef{
    _varType :: VarType ann,
    _varName :: Ident ann,
    _vardefAnnotation :: ann
} deriving Show

data ProcDef ann = ProcDef{
    _returnType :: UnitKind,
    _procName :: Ident ann,
    _parameters :: [VarDef ann],
    _body :: [Statement ann],
    _procdefAnnotation :: ann
} deriving Show

data LeftValueExpr ann = ArrayRef {
    _arrayRefName :: Ident ann,
    _offset :: Expr ann,
    _leftvalueexprAnnotation :: ann
} | VarRef {
    _varRefName :: Ident ann,
    _leftvalueexprAnnotation :: ann
} deriving Show
data BinaryOp = Add | Sub | Mul | Div | Less | Equal | NEqual | Greater | LessEqual | GreaterEqual deriving Show
data UnaryOp = Positive | Neg deriving Show
data ProcedureCall ann = ProcedureCall{
    _calleeName :: Ident ann,
    _callingArgs :: [Expr ann],
    _procedurecallAnnotation :: ann
} deriving Show
data Expr ann = ImmInt {_immInt :: Int, _exprAnnotation :: ann}
    | ImmComplex {_immComplex :: Complex Double, _exprAnnotation :: ann}
    | BinaryOp {_binaryOpType :: BinaryOp, _binaryLhs :: Expr ann, _binaryRhs :: Expr ann, _exprAnnotation :: ann}
    | UnaryOp {_unaryOpType :: UnaryOp, _unaryVal :: Expr ann, _exprAnnotation :: ann}
    | LeftExpr {_leftRef :: LeftValueExpr ann, _exprAnnotation :: ann}
    | MeasureExpr {_measureRef :: LeftValueExpr ann, _exprAnnotation :: ann} 
    | CallExpr {_exprCalling :: ProcedureCall ann, _exprAnnotation :: ann} deriving Show
data GateDecorator = GateDecorator {_controlStates :: [Bool], _adjoint :: Bool} deriving Show
data Statement ann = 
      QbitInitStmt { _qubitOperand :: LeftValueExpr ann, _statementAnnotation :: ann}
    | QbitGateStmt { _decorator :: GateDecorator, _qubitOperands :: [LeftValueExpr ann], _appliedGateName :: Ident ann, _statementAnnotation :: ann}
    | CintAssignStmt { _assignLhs :: LeftValueExpr ann, _assignRhs :: Expr ann, _statementAnnotation :: ann}
    | IfStatement { _ifCondition :: Expr ann, _thenBranch :: [Statement ann], _elseBranch :: [Statement ann], _statementAnnotation :: ann}
    | ForStatement { _forIdent :: Ident ann, _lo :: Expr ann, _hi :: Expr ann, _forBody :: [Statement ann], _statementAnnotation :: ann}
    | WhileStatement {_whileCondition :: Expr ann, _whileBody :: [Statement ann], _statementAnnotation :: ann}
    | PrintStatement {_printedExpr :: Expr ann, _statementAnnotation :: ann}
    | PassStatement {_statementAnnotation :: ann}
    | ReturnStatement {_returnedExpr :: Maybe (Expr ann), _statementAnnotation :: ann}
    | CallStatement {_callExpr :: ProcedureCall ann, _statementAnnotation :: ann}
    | VarDefStatement {_localVarDef :: [VarDef ann], _statementAnnotation :: ann}
    deriving Show

data Program ann = Program {
    _topGatedefs :: [GateDef ann],
    _topVardefs :: [VarDef ann],
    _procedures :: [ProcDef ann],
    _programAnnotation :: ann} deriving Show

makeLenses ''Matrix
makeLenses ''Ident
makeLenses ''GateDef
makeLenses ''UnitKind
makeLenses ''VarType
makeLenses ''VarDef
makeLenses ''ProcDef
makeLenses ''LeftValueExpr
makeLenses ''BinaryOp
makeLenses ''UnaryOp
makeLenses ''ProcedureCall
makeLenses ''Expr
makeLenses ''GateDecorator
makeLenses ''Statement
makeLenses ''Program

instance Annotated Matrix where {annotation = matrixAnnotation}
instance Annotated Ident where {annotation = identAnnotation}
instance Annotated GateDef where {annotation = gatedefAnnotation}
instance Annotated VarType where {annotation = vartypeAnnotation}
instance Annotated VarDef where {annotation = vardefAnnotation}
instance Annotated ProcDef where {annotation = procdefAnnotation}
instance Annotated LeftValueExpr where {annotation = leftvalueexprAnnotation}
instance Annotated ProcedureCall where {annotation = procedurecallAnnotation}
instance Annotated Expr where {annotation = exprAnnotation}
--instance Annotated GateDecorator where {annotation = gatedecoratorAnnotation}
instance Annotated Statement where {annotation = statementAnnotation}
instance Annotated Program where {annotation = programAnnotation}

type Pos = SourcePos
