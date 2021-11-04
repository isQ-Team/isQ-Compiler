parser grammar ISQParser;

options {
    tokenVocab = ISQLexer;
}

program: 
    gateDefclause* defclause+ programBody;

gateDefclause:
    Defgate Identifier Assign LeftBracket matrixContents RightBracket Semi
    ;

matrixContents:
    cNumber          # matrixvaldef
    | cNumber (Comma | Semi) matrixContents     # matrixdef
    ;

cNumber:
    numberExpr
    | Minus numberExpr;

numberExpr:
    Number
    | Number Plus Number
    | Number Minus Number;

varType:
    Qbit | Int;


defclause:
    varType idlist Semi;

idlist:
    Identifier      # singleiddef
    | Identifier LeftBracket Number RightBracket  # arraydef
    | idlist Comma idlist  # idlistdef
;

programBody:
    procedureBlock* procedureMain
    ;

procedureBlock:
    (Procedure | Int) Identifier LeftParen RightParen LeftBrace procedureBody RightBrace
    | (Procedure | Int) Identifier LeftParen callParas RightParen LeftBrace procedureBody RightBrace
//    | Int Identifier LeftParen RightParen LeftBrace procedureBody RightBrace
//    | Int Identifier LeftParen callParas RightParen LeftBrace procedureBody RightBrace
;

callParas:
    varType Identifier
    | varType Identifier LeftBracket RightBracket
    | callParas Comma callParas;

procedureMain:
    Procedure Main LeftParen RightParen LeftBrace procedureBody RightBrace;

procedureBody:
    statementBlock
    | statementBlock returnStatement
;

statement:
    qbitInitStatement                       #qbitinitdef
    | qbitUnitaryStatement                  #qbitunitarydef
    | cintAssign                            #cinassigndef
    | ifStatement                           #ifdef
    | callStatement                         #calldef
    | whileStatement                        #whiledef
    | forStatement                          #fordef
    | printStatement                        #printdef
    | passStatement                         #passdef
    | defclause                             #vardef
;

statementBlock:
    statement+
;

uGate:
    H | X | Y | Z | S | T | CZ | CX | CNOT | Identifier
;

variable:
    Identifier | Number | Identifier LeftBracket variable RightBracket;

variableList:
    variableList Comma variableList
    | variable
;

/* 
expression:
    variable
    | expression Plus expression
    | expression Minus expression
    | expression Mult expression
    | expression Div expression
    | LeftParen expression RightParen;
*/

binopPlus:
    Plus | Minus;

binopMult:
    Mult | Div;

expression:
    multexp (binopPlus multexp)*
;

multexp:
    atomexp (binopMult atomexp)*
;

atomexp:
    variable
    | LeftParen expression RightParen
;

mExpression:
    M Less variable Greater;

association:
    Equal | GreaterEqual | LessEqual | Greater | Less
;

qbitInitStatement:
    variable Assign KetZero Semi
;

qbitUnitaryStatement:
    uGate Less variableList Greater Semi
;

cintAssign:
    variable Assign expression Semi
    | variable Assign mExpression Semi
    | variable Assign callStatement
;

regionBody:
    statement
    | LeftBrace statementBlock RightBrace
;

ifStatement:
    If LeftParen expression association expression RightParen regionBody
    | If LeftParen expression association expression RightParen regionBody Else regionBody
;

forStatement:
    For LeftParen Identifier Assign variable To variable RightParen regionBody
;

whileStatement:
    While LeftParen expression association expression RightParen regionBody
;

callStatement:
    Identifier LeftParen (|variableList) RightParen Semi
;

printStatement:
    Print variable Semi
;

passStatement:
    Pass Semi
;

returnStatement:
    Return variable Semi
;