
// Generated from ISQParser.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  ISQParser : public antlr4::Parser {
public:
  enum {
    WhiteSpace = 1, NewLine = 2, BlockComment = 3, LineComment = 4, If = 5, 
    Then = 6, Else = 7, Fi = 8, For = 9, To = 10, While = 11, Do = 12, Od = 13, 
    Procedure = 14, Main = 15, Int = 16, Qbit = 17, H = 18, X = 19, Y = 20, 
    Z = 21, S = 22, T = 23, CZ = 24, CX = 25, CNOT = 26, M = 27, Print = 28, 
    Defgate = 29, Pass = 30, Return = 31, Assign = 32, Plus = 33, Minus = 34, 
    Mult = 35, Div = 36, Less = 37, Greater = 38, Comma = 39, LeftParen = 40, 
    RightParen = 41, LeftBracket = 42, RightBracket = 43, LeftBrace = 44, 
    RightBrace = 45, Semi = 46, Equal = 47, LessEqual = 48, GreaterEqual = 49, 
    KetZero = 50, Identifier = 51, Number = 52
  };

  enum {
    RuleProgram = 0, RuleGateDefclause = 1, RuleMatrixContents = 2, RuleCNumber = 3, 
    RuleNumberExpr = 4, RuleVarType = 5, RuleDefclause = 6, RuleIdlist = 7, 
    RuleProgramBody = 8, RuleProcedureBlock = 9, RuleCallParas = 10, RuleProcedureMain = 11, 
    RuleProcedureBody = 12, RuleStatement = 13, RuleStatementBlock = 14, 
    RuleUGate = 15, RuleVariable = 16, RuleVariableList = 17, RuleBinopPlus = 18, 
    RuleBinopMult = 19, RuleExpression = 20, RuleMultexp = 21, RuleAtomexp = 22, 
    RuleMExpression = 23, RuleAssociation = 24, RuleQbitInitStatement = 25, 
    RuleQbitUnitaryStatement = 26, RuleCintAssign = 27, RuleRegionBody = 28, 
    RuleIfStatement = 29, RuleForStatement = 30, RuleWhileStatement = 31, 
    RuleCallStatement = 32, RulePrintStatement = 33, RulePassStatement = 34, 
    RuleReturnStatement = 35
  };

  ISQParser(antlr4::TokenStream *input);
  ~ISQParser();

  virtual std::string getGrammarFileName() const override;
  virtual const antlr4::atn::ATN& getATN() const override { return _atn; };
  virtual const std::vector<std::string>& getTokenNames() const override { return _tokenNames; }; // deprecated: use vocabulary instead.
  virtual const std::vector<std::string>& getRuleNames() const override;
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;


  class ProgramContext;
  class GateDefclauseContext;
  class MatrixContentsContext;
  class CNumberContext;
  class NumberExprContext;
  class VarTypeContext;
  class DefclauseContext;
  class IdlistContext;
  class ProgramBodyContext;
  class ProcedureBlockContext;
  class CallParasContext;
  class ProcedureMainContext;
  class ProcedureBodyContext;
  class StatementContext;
  class StatementBlockContext;
  class UGateContext;
  class VariableContext;
  class VariableListContext;
  class BinopPlusContext;
  class BinopMultContext;
  class ExpressionContext;
  class MultexpContext;
  class AtomexpContext;
  class MExpressionContext;
  class AssociationContext;
  class QbitInitStatementContext;
  class QbitUnitaryStatementContext;
  class CintAssignContext;
  class RegionBodyContext;
  class IfStatementContext;
  class ForStatementContext;
  class WhileStatementContext;
  class CallStatementContext;
  class PrintStatementContext;
  class PassStatementContext;
  class ReturnStatementContext; 

  class  ProgramContext : public antlr4::ParserRuleContext {
  public:
    ProgramContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ProgramBodyContext *programBody();
    std::vector<GateDefclauseContext *> gateDefclause();
    GateDefclauseContext* gateDefclause(size_t i);
    std::vector<DefclauseContext *> defclause();
    DefclauseContext* defclause(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ProgramContext* program();

  class  GateDefclauseContext : public antlr4::ParserRuleContext {
  public:
    GateDefclauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Defgate();
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *Assign();
    antlr4::tree::TerminalNode *LeftBracket();
    MatrixContentsContext *matrixContents();
    antlr4::tree::TerminalNode *RightBracket();
    antlr4::tree::TerminalNode *Semi();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  GateDefclauseContext* gateDefclause();

  class  MatrixContentsContext : public antlr4::ParserRuleContext {
  public:
    MatrixContentsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    MatrixContentsContext() = default;
    void copyFrom(MatrixContentsContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  MatrixdefContext : public MatrixContentsContext {
  public:
    MatrixdefContext(MatrixContentsContext *ctx);

    CNumberContext *cNumber();
    MatrixContentsContext *matrixContents();
    antlr4::tree::TerminalNode *Comma();
    antlr4::tree::TerminalNode *Semi();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  MatrixvaldefContext : public MatrixContentsContext {
  public:
    MatrixvaldefContext(MatrixContentsContext *ctx);

    CNumberContext *cNumber();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MatrixContentsContext* matrixContents();

  class  CNumberContext : public antlr4::ParserRuleContext {
  public:
    CNumberContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    NumberExprContext *numberExpr();
    antlr4::tree::TerminalNode *Minus();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CNumberContext* cNumber();

  class  NumberExprContext : public antlr4::ParserRuleContext {
  public:
    NumberExprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> Number();
    antlr4::tree::TerminalNode* Number(size_t i);
    antlr4::tree::TerminalNode *Plus();
    antlr4::tree::TerminalNode *Minus();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  NumberExprContext* numberExpr();

  class  VarTypeContext : public antlr4::ParserRuleContext {
  public:
    VarTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Qbit();
    antlr4::tree::TerminalNode *Int();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  VarTypeContext* varType();

  class  DefclauseContext : public antlr4::ParserRuleContext {
  public:
    DefclauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    VarTypeContext *varType();
    IdlistContext *idlist();
    antlr4::tree::TerminalNode *Semi();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  DefclauseContext* defclause();

  class  IdlistContext : public antlr4::ParserRuleContext {
  public:
    IdlistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    IdlistContext() = default;
    void copyFrom(IdlistContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  IdlistdefContext : public IdlistContext {
  public:
    IdlistdefContext(IdlistContext *ctx);

    std::vector<IdlistContext *> idlist();
    IdlistContext* idlist(size_t i);
    antlr4::tree::TerminalNode *Comma();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ArraydefContext : public IdlistContext {
  public:
    ArraydefContext(IdlistContext *ctx);

    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *LeftBracket();
    antlr4::tree::TerminalNode *Number();
    antlr4::tree::TerminalNode *RightBracket();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  SingleiddefContext : public IdlistContext {
  public:
    SingleiddefContext(IdlistContext *ctx);

    antlr4::tree::TerminalNode *Identifier();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  IdlistContext* idlist();
  IdlistContext* idlist(int precedence);
  class  ProgramBodyContext : public antlr4::ParserRuleContext {
  public:
    ProgramBodyContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ProcedureMainContext *procedureMain();
    std::vector<ProcedureBlockContext *> procedureBlock();
    ProcedureBlockContext* procedureBlock(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ProgramBodyContext* programBody();

  class  ProcedureBlockContext : public antlr4::ParserRuleContext {
  public:
    ProcedureBlockContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *LeftParen();
    antlr4::tree::TerminalNode *RightParen();
    antlr4::tree::TerminalNode *LeftBrace();
    ProcedureBodyContext *procedureBody();
    antlr4::tree::TerminalNode *RightBrace();
    antlr4::tree::TerminalNode *Procedure();
    antlr4::tree::TerminalNode *Int();
    CallParasContext *callParas();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ProcedureBlockContext* procedureBlock();

  class  CallParasContext : public antlr4::ParserRuleContext {
  public:
    CallParasContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    VarTypeContext *varType();
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *LeftBracket();
    antlr4::tree::TerminalNode *RightBracket();
    std::vector<CallParasContext *> callParas();
    CallParasContext* callParas(size_t i);
    antlr4::tree::TerminalNode *Comma();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CallParasContext* callParas();
  CallParasContext* callParas(int precedence);
  class  ProcedureMainContext : public antlr4::ParserRuleContext {
  public:
    ProcedureMainContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Procedure();
    antlr4::tree::TerminalNode *Main();
    antlr4::tree::TerminalNode *LeftParen();
    antlr4::tree::TerminalNode *RightParen();
    antlr4::tree::TerminalNode *LeftBrace();
    ProcedureBodyContext *procedureBody();
    antlr4::tree::TerminalNode *RightBrace();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ProcedureMainContext* procedureMain();

  class  ProcedureBodyContext : public antlr4::ParserRuleContext {
  public:
    ProcedureBodyContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    StatementBlockContext *statementBlock();
    ReturnStatementContext *returnStatement();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ProcedureBodyContext* procedureBody();

  class  StatementContext : public antlr4::ParserRuleContext {
  public:
    StatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    StatementContext() = default;
    void copyFrom(StatementContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  QbitinitdefContext : public StatementContext {
  public:
    QbitinitdefContext(StatementContext *ctx);

    QbitInitStatementContext *qbitInitStatement();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  CinassigndefContext : public StatementContext {
  public:
    CinassigndefContext(StatementContext *ctx);

    CintAssignContext *cintAssign();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  FordefContext : public StatementContext {
  public:
    FordefContext(StatementContext *ctx);

    ForStatementContext *forStatement();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  PrintdefContext : public StatementContext {
  public:
    PrintdefContext(StatementContext *ctx);

    PrintStatementContext *printStatement();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  CalldefContext : public StatementContext {
  public:
    CalldefContext(StatementContext *ctx);

    CallStatementContext *callStatement();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  QbitunitarydefContext : public StatementContext {
  public:
    QbitunitarydefContext(StatementContext *ctx);

    QbitUnitaryStatementContext *qbitUnitaryStatement();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  WhiledefContext : public StatementContext {
  public:
    WhiledefContext(StatementContext *ctx);

    WhileStatementContext *whileStatement();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  VardefContext : public StatementContext {
  public:
    VardefContext(StatementContext *ctx);

    DefclauseContext *defclause();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  PassdefContext : public StatementContext {
  public:
    PassdefContext(StatementContext *ctx);

    PassStatementContext *passStatement();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  IfdefContext : public StatementContext {
  public:
    IfdefContext(StatementContext *ctx);

    IfStatementContext *ifStatement();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StatementContext* statement();

  class  StatementBlockContext : public antlr4::ParserRuleContext {
  public:
    StatementBlockContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  StatementBlockContext* statementBlock();

  class  UGateContext : public antlr4::ParserRuleContext {
  public:
    UGateContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *H();
    antlr4::tree::TerminalNode *X();
    antlr4::tree::TerminalNode *Y();
    antlr4::tree::TerminalNode *Z();
    antlr4::tree::TerminalNode *S();
    antlr4::tree::TerminalNode *T();
    antlr4::tree::TerminalNode *CZ();
    antlr4::tree::TerminalNode *CX();
    antlr4::tree::TerminalNode *CNOT();
    antlr4::tree::TerminalNode *Identifier();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  UGateContext* uGate();

  class  VariableContext : public antlr4::ParserRuleContext {
  public:
    VariableContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *Number();
    antlr4::tree::TerminalNode *LeftBracket();
    VariableContext *variable();
    antlr4::tree::TerminalNode *RightBracket();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  VariableContext* variable();

  class  VariableListContext : public antlr4::ParserRuleContext {
  public:
    VariableListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    VariableContext *variable();
    std::vector<VariableListContext *> variableList();
    VariableListContext* variableList(size_t i);
    antlr4::tree::TerminalNode *Comma();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  VariableListContext* variableList();
  VariableListContext* variableList(int precedence);
  class  BinopPlusContext : public antlr4::ParserRuleContext {
  public:
    BinopPlusContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Plus();
    antlr4::tree::TerminalNode *Minus();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BinopPlusContext* binopPlus();

  class  BinopMultContext : public antlr4::ParserRuleContext {
  public:
    BinopMultContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Mult();
    antlr4::tree::TerminalNode *Div();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BinopMultContext* binopMult();

  class  ExpressionContext : public antlr4::ParserRuleContext {
  public:
    ExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<MultexpContext *> multexp();
    MultexpContext* multexp(size_t i);
    std::vector<BinopPlusContext *> binopPlus();
    BinopPlusContext* binopPlus(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ExpressionContext* expression();

  class  MultexpContext : public antlr4::ParserRuleContext {
  public:
    MultexpContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<AtomexpContext *> atomexp();
    AtomexpContext* atomexp(size_t i);
    std::vector<BinopMultContext *> binopMult();
    BinopMultContext* binopMult(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  MultexpContext* multexp();

  class  AtomexpContext : public antlr4::ParserRuleContext {
  public:
    AtomexpContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    VariableContext *variable();
    antlr4::tree::TerminalNode *LeftParen();
    ExpressionContext *expression();
    antlr4::tree::TerminalNode *RightParen();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  AtomexpContext* atomexp();

  class  MExpressionContext : public antlr4::ParserRuleContext {
  public:
    MExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *M();
    antlr4::tree::TerminalNode *Less();
    VariableContext *variable();
    antlr4::tree::TerminalNode *Greater();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  MExpressionContext* mExpression();

  class  AssociationContext : public antlr4::ParserRuleContext {
  public:
    AssociationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Equal();
    antlr4::tree::TerminalNode *GreaterEqual();
    antlr4::tree::TerminalNode *LessEqual();
    antlr4::tree::TerminalNode *Greater();
    antlr4::tree::TerminalNode *Less();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  AssociationContext* association();

  class  QbitInitStatementContext : public antlr4::ParserRuleContext {
  public:
    QbitInitStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    VariableContext *variable();
    antlr4::tree::TerminalNode *Assign();
    antlr4::tree::TerminalNode *KetZero();
    antlr4::tree::TerminalNode *Semi();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QbitInitStatementContext* qbitInitStatement();

  class  QbitUnitaryStatementContext : public antlr4::ParserRuleContext {
  public:
    QbitUnitaryStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    UGateContext *uGate();
    antlr4::tree::TerminalNode *Less();
    VariableListContext *variableList();
    antlr4::tree::TerminalNode *Greater();
    antlr4::tree::TerminalNode *Semi();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  QbitUnitaryStatementContext* qbitUnitaryStatement();

  class  CintAssignContext : public antlr4::ParserRuleContext {
  public:
    CintAssignContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    VariableContext *variable();
    antlr4::tree::TerminalNode *Assign();
    ExpressionContext *expression();
    antlr4::tree::TerminalNode *Semi();
    MExpressionContext *mExpression();
    CallStatementContext *callStatement();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CintAssignContext* cintAssign();

  class  RegionBodyContext : public antlr4::ParserRuleContext {
  public:
    RegionBodyContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    StatementContext *statement();
    antlr4::tree::TerminalNode *LeftBrace();
    StatementBlockContext *statementBlock();
    antlr4::tree::TerminalNode *RightBrace();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  RegionBodyContext* regionBody();

  class  IfStatementContext : public antlr4::ParserRuleContext {
  public:
    IfStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *If();
    antlr4::tree::TerminalNode *LeftParen();
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    AssociationContext *association();
    antlr4::tree::TerminalNode *RightParen();
    std::vector<RegionBodyContext *> regionBody();
    RegionBodyContext* regionBody(size_t i);
    antlr4::tree::TerminalNode *Else();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  IfStatementContext* ifStatement();

  class  ForStatementContext : public antlr4::ParserRuleContext {
  public:
    ForStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *For();
    antlr4::tree::TerminalNode *LeftParen();
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *Assign();
    std::vector<VariableContext *> variable();
    VariableContext* variable(size_t i);
    antlr4::tree::TerminalNode *To();
    antlr4::tree::TerminalNode *RightParen();
    RegionBodyContext *regionBody();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ForStatementContext* forStatement();

  class  WhileStatementContext : public antlr4::ParserRuleContext {
  public:
    WhileStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *While();
    antlr4::tree::TerminalNode *LeftParen();
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    AssociationContext *association();
    antlr4::tree::TerminalNode *RightParen();
    RegionBodyContext *regionBody();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  WhileStatementContext* whileStatement();

  class  CallStatementContext : public antlr4::ParserRuleContext {
  public:
    CallStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *LeftParen();
    antlr4::tree::TerminalNode *RightParen();
    antlr4::tree::TerminalNode *Semi();
    VariableListContext *variableList();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CallStatementContext* callStatement();

  class  PrintStatementContext : public antlr4::ParserRuleContext {
  public:
    PrintStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Print();
    VariableContext *variable();
    antlr4::tree::TerminalNode *Semi();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  PrintStatementContext* printStatement();

  class  PassStatementContext : public antlr4::ParserRuleContext {
  public:
    PassStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Pass();
    antlr4::tree::TerminalNode *Semi();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  PassStatementContext* passStatement();

  class  ReturnStatementContext : public antlr4::ParserRuleContext {
  public:
    ReturnStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Return();
    VariableContext *variable();
    antlr4::tree::TerminalNode *Semi();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ReturnStatementContext* returnStatement();


  virtual bool sempred(antlr4::RuleContext *_localctx, size_t ruleIndex, size_t predicateIndex) override;
  bool idlistSempred(IdlistContext *_localctx, size_t predicateIndex);
  bool callParasSempred(CallParasContext *_localctx, size_t predicateIndex);
  bool variableListSempred(VariableListContext *_localctx, size_t predicateIndex);

private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;


  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};

