
// Generated from ISQParser.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"
#include "ISQParser.h"


/**
 * This interface defines an abstract listener for a parse tree produced by ISQParser.
 */
class  ISQParserListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterProgram(ISQParser::ProgramContext *ctx) = 0;
  virtual void exitProgram(ISQParser::ProgramContext *ctx) = 0;

  virtual void enterGateDefclause(ISQParser::GateDefclauseContext *ctx) = 0;
  virtual void exitGateDefclause(ISQParser::GateDefclauseContext *ctx) = 0;

  virtual void enterMatrixvaldef(ISQParser::MatrixvaldefContext *ctx) = 0;
  virtual void exitMatrixvaldef(ISQParser::MatrixvaldefContext *ctx) = 0;

  virtual void enterMatrixdef(ISQParser::MatrixdefContext *ctx) = 0;
  virtual void exitMatrixdef(ISQParser::MatrixdefContext *ctx) = 0;

  virtual void enterCNumber(ISQParser::CNumberContext *ctx) = 0;
  virtual void exitCNumber(ISQParser::CNumberContext *ctx) = 0;

  virtual void enterNumberExpr(ISQParser::NumberExprContext *ctx) = 0;
  virtual void exitNumberExpr(ISQParser::NumberExprContext *ctx) = 0;

  virtual void enterVarType(ISQParser::VarTypeContext *ctx) = 0;
  virtual void exitVarType(ISQParser::VarTypeContext *ctx) = 0;

  virtual void enterDefclause(ISQParser::DefclauseContext *ctx) = 0;
  virtual void exitDefclause(ISQParser::DefclauseContext *ctx) = 0;

  virtual void enterIdlistdef(ISQParser::IdlistdefContext *ctx) = 0;
  virtual void exitIdlistdef(ISQParser::IdlistdefContext *ctx) = 0;

  virtual void enterArraydef(ISQParser::ArraydefContext *ctx) = 0;
  virtual void exitArraydef(ISQParser::ArraydefContext *ctx) = 0;

  virtual void enterSingleiddef(ISQParser::SingleiddefContext *ctx) = 0;
  virtual void exitSingleiddef(ISQParser::SingleiddefContext *ctx) = 0;

  virtual void enterProgramBody(ISQParser::ProgramBodyContext *ctx) = 0;
  virtual void exitProgramBody(ISQParser::ProgramBodyContext *ctx) = 0;

  virtual void enterProcedureBlock(ISQParser::ProcedureBlockContext *ctx) = 0;
  virtual void exitProcedureBlock(ISQParser::ProcedureBlockContext *ctx) = 0;

  virtual void enterCallParas(ISQParser::CallParasContext *ctx) = 0;
  virtual void exitCallParas(ISQParser::CallParasContext *ctx) = 0;

  virtual void enterProcedureMain(ISQParser::ProcedureMainContext *ctx) = 0;
  virtual void exitProcedureMain(ISQParser::ProcedureMainContext *ctx) = 0;

  virtual void enterProcedureBody(ISQParser::ProcedureBodyContext *ctx) = 0;
  virtual void exitProcedureBody(ISQParser::ProcedureBodyContext *ctx) = 0;

  virtual void enterQbitinitdef(ISQParser::QbitinitdefContext *ctx) = 0;
  virtual void exitQbitinitdef(ISQParser::QbitinitdefContext *ctx) = 0;

  virtual void enterQbitunitarydef(ISQParser::QbitunitarydefContext *ctx) = 0;
  virtual void exitQbitunitarydef(ISQParser::QbitunitarydefContext *ctx) = 0;

  virtual void enterCinassigndef(ISQParser::CinassigndefContext *ctx) = 0;
  virtual void exitCinassigndef(ISQParser::CinassigndefContext *ctx) = 0;

  virtual void enterIfdef(ISQParser::IfdefContext *ctx) = 0;
  virtual void exitIfdef(ISQParser::IfdefContext *ctx) = 0;

  virtual void enterCalldef(ISQParser::CalldefContext *ctx) = 0;
  virtual void exitCalldef(ISQParser::CalldefContext *ctx) = 0;

  virtual void enterWhiledef(ISQParser::WhiledefContext *ctx) = 0;
  virtual void exitWhiledef(ISQParser::WhiledefContext *ctx) = 0;

  virtual void enterFordef(ISQParser::FordefContext *ctx) = 0;
  virtual void exitFordef(ISQParser::FordefContext *ctx) = 0;

  virtual void enterPrintdef(ISQParser::PrintdefContext *ctx) = 0;
  virtual void exitPrintdef(ISQParser::PrintdefContext *ctx) = 0;

  virtual void enterPassdef(ISQParser::PassdefContext *ctx) = 0;
  virtual void exitPassdef(ISQParser::PassdefContext *ctx) = 0;

  virtual void enterVardef(ISQParser::VardefContext *ctx) = 0;
  virtual void exitVardef(ISQParser::VardefContext *ctx) = 0;

  virtual void enterStatementBlock(ISQParser::StatementBlockContext *ctx) = 0;
  virtual void exitStatementBlock(ISQParser::StatementBlockContext *ctx) = 0;

  virtual void enterUGate(ISQParser::UGateContext *ctx) = 0;
  virtual void exitUGate(ISQParser::UGateContext *ctx) = 0;

  virtual void enterVariable(ISQParser::VariableContext *ctx) = 0;
  virtual void exitVariable(ISQParser::VariableContext *ctx) = 0;

  virtual void enterVariableList(ISQParser::VariableListContext *ctx) = 0;
  virtual void exitVariableList(ISQParser::VariableListContext *ctx) = 0;

  virtual void enterBinopPlus(ISQParser::BinopPlusContext *ctx) = 0;
  virtual void exitBinopPlus(ISQParser::BinopPlusContext *ctx) = 0;

  virtual void enterBinopMult(ISQParser::BinopMultContext *ctx) = 0;
  virtual void exitBinopMult(ISQParser::BinopMultContext *ctx) = 0;

  virtual void enterExpression(ISQParser::ExpressionContext *ctx) = 0;
  virtual void exitExpression(ISQParser::ExpressionContext *ctx) = 0;

  virtual void enterMultexp(ISQParser::MultexpContext *ctx) = 0;
  virtual void exitMultexp(ISQParser::MultexpContext *ctx) = 0;

  virtual void enterAtomexp(ISQParser::AtomexpContext *ctx) = 0;
  virtual void exitAtomexp(ISQParser::AtomexpContext *ctx) = 0;

  virtual void enterMExpression(ISQParser::MExpressionContext *ctx) = 0;
  virtual void exitMExpression(ISQParser::MExpressionContext *ctx) = 0;

  virtual void enterAssociation(ISQParser::AssociationContext *ctx) = 0;
  virtual void exitAssociation(ISQParser::AssociationContext *ctx) = 0;

  virtual void enterQbitInitStatement(ISQParser::QbitInitStatementContext *ctx) = 0;
  virtual void exitQbitInitStatement(ISQParser::QbitInitStatementContext *ctx) = 0;

  virtual void enterQbitUnitaryStatement(ISQParser::QbitUnitaryStatementContext *ctx) = 0;
  virtual void exitQbitUnitaryStatement(ISQParser::QbitUnitaryStatementContext *ctx) = 0;

  virtual void enterCintAssign(ISQParser::CintAssignContext *ctx) = 0;
  virtual void exitCintAssign(ISQParser::CintAssignContext *ctx) = 0;

  virtual void enterRegionBody(ISQParser::RegionBodyContext *ctx) = 0;
  virtual void exitRegionBody(ISQParser::RegionBodyContext *ctx) = 0;

  virtual void enterIfStatement(ISQParser::IfStatementContext *ctx) = 0;
  virtual void exitIfStatement(ISQParser::IfStatementContext *ctx) = 0;

  virtual void enterForStatement(ISQParser::ForStatementContext *ctx) = 0;
  virtual void exitForStatement(ISQParser::ForStatementContext *ctx) = 0;

  virtual void enterWhileStatement(ISQParser::WhileStatementContext *ctx) = 0;
  virtual void exitWhileStatement(ISQParser::WhileStatementContext *ctx) = 0;

  virtual void enterCallStatement(ISQParser::CallStatementContext *ctx) = 0;
  virtual void exitCallStatement(ISQParser::CallStatementContext *ctx) = 0;

  virtual void enterPrintStatement(ISQParser::PrintStatementContext *ctx) = 0;
  virtual void exitPrintStatement(ISQParser::PrintStatementContext *ctx) = 0;

  virtual void enterPassStatement(ISQParser::PassStatementContext *ctx) = 0;
  virtual void exitPassStatement(ISQParser::PassStatementContext *ctx) = 0;

  virtual void enterReturnStatement(ISQParser::ReturnStatementContext *ctx) = 0;
  virtual void exitReturnStatement(ISQParser::ReturnStatementContext *ctx) = 0;


};

