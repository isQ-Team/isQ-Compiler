
// Generated from ISQParser.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"
#include "ISQParserListener.h"


/**
 * This class provides an empty implementation of ISQParserListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  ISQParserBaseListener : public ISQParserListener {
public:

  virtual void enterProgram(ISQParser::ProgramContext * /*ctx*/) override { }
  virtual void exitProgram(ISQParser::ProgramContext * /*ctx*/) override { }

  virtual void enterGateDefclause(ISQParser::GateDefclauseContext * /*ctx*/) override { }
  virtual void exitGateDefclause(ISQParser::GateDefclauseContext * /*ctx*/) override { }

  virtual void enterMatrixvaldef(ISQParser::MatrixvaldefContext * /*ctx*/) override { }
  virtual void exitMatrixvaldef(ISQParser::MatrixvaldefContext * /*ctx*/) override { }

  virtual void enterMatrixdef(ISQParser::MatrixdefContext * /*ctx*/) override { }
  virtual void exitMatrixdef(ISQParser::MatrixdefContext * /*ctx*/) override { }

  virtual void enterCNumber(ISQParser::CNumberContext * /*ctx*/) override { }
  virtual void exitCNumber(ISQParser::CNumberContext * /*ctx*/) override { }

  virtual void enterNumberExpr(ISQParser::NumberExprContext * /*ctx*/) override { }
  virtual void exitNumberExpr(ISQParser::NumberExprContext * /*ctx*/) override { }

  virtual void enterVarType(ISQParser::VarTypeContext * /*ctx*/) override { }
  virtual void exitVarType(ISQParser::VarTypeContext * /*ctx*/) override { }

  virtual void enterDefclause(ISQParser::DefclauseContext * /*ctx*/) override { }
  virtual void exitDefclause(ISQParser::DefclauseContext * /*ctx*/) override { }

  virtual void enterIdlistdef(ISQParser::IdlistdefContext * /*ctx*/) override { }
  virtual void exitIdlistdef(ISQParser::IdlistdefContext * /*ctx*/) override { }

  virtual void enterArraydef(ISQParser::ArraydefContext * /*ctx*/) override { }
  virtual void exitArraydef(ISQParser::ArraydefContext * /*ctx*/) override { }

  virtual void enterSingleiddef(ISQParser::SingleiddefContext * /*ctx*/) override { }
  virtual void exitSingleiddef(ISQParser::SingleiddefContext * /*ctx*/) override { }

  virtual void enterProgramBody(ISQParser::ProgramBodyContext * /*ctx*/) override { }
  virtual void exitProgramBody(ISQParser::ProgramBodyContext * /*ctx*/) override { }

  virtual void enterProcedureBlock(ISQParser::ProcedureBlockContext * /*ctx*/) override { }
  virtual void exitProcedureBlock(ISQParser::ProcedureBlockContext * /*ctx*/) override { }

  virtual void enterCallParas(ISQParser::CallParasContext * /*ctx*/) override { }
  virtual void exitCallParas(ISQParser::CallParasContext * /*ctx*/) override { }

  virtual void enterProcedureMain(ISQParser::ProcedureMainContext * /*ctx*/) override { }
  virtual void exitProcedureMain(ISQParser::ProcedureMainContext * /*ctx*/) override { }

  virtual void enterProcedureBody(ISQParser::ProcedureBodyContext * /*ctx*/) override { }
  virtual void exitProcedureBody(ISQParser::ProcedureBodyContext * /*ctx*/) override { }

  virtual void enterQbitinitdef(ISQParser::QbitinitdefContext * /*ctx*/) override { }
  virtual void exitQbitinitdef(ISQParser::QbitinitdefContext * /*ctx*/) override { }

  virtual void enterQbitunitarydef(ISQParser::QbitunitarydefContext * /*ctx*/) override { }
  virtual void exitQbitunitarydef(ISQParser::QbitunitarydefContext * /*ctx*/) override { }

  virtual void enterCinassigndef(ISQParser::CinassigndefContext * /*ctx*/) override { }
  virtual void exitCinassigndef(ISQParser::CinassigndefContext * /*ctx*/) override { }

  virtual void enterIfdef(ISQParser::IfdefContext * /*ctx*/) override { }
  virtual void exitIfdef(ISQParser::IfdefContext * /*ctx*/) override { }

  virtual void enterCalldef(ISQParser::CalldefContext * /*ctx*/) override { }
  virtual void exitCalldef(ISQParser::CalldefContext * /*ctx*/) override { }

  virtual void enterWhiledef(ISQParser::WhiledefContext * /*ctx*/) override { }
  virtual void exitWhiledef(ISQParser::WhiledefContext * /*ctx*/) override { }

  virtual void enterFordef(ISQParser::FordefContext * /*ctx*/) override { }
  virtual void exitFordef(ISQParser::FordefContext * /*ctx*/) override { }

  virtual void enterPrintdef(ISQParser::PrintdefContext * /*ctx*/) override { }
  virtual void exitPrintdef(ISQParser::PrintdefContext * /*ctx*/) override { }

  virtual void enterPassdef(ISQParser::PassdefContext * /*ctx*/) override { }
  virtual void exitPassdef(ISQParser::PassdefContext * /*ctx*/) override { }

  virtual void enterVardef(ISQParser::VardefContext * /*ctx*/) override { }
  virtual void exitVardef(ISQParser::VardefContext * /*ctx*/) override { }

  virtual void enterStatementBlock(ISQParser::StatementBlockContext * /*ctx*/) override { }
  virtual void exitStatementBlock(ISQParser::StatementBlockContext * /*ctx*/) override { }

  virtual void enterUGate(ISQParser::UGateContext * /*ctx*/) override { }
  virtual void exitUGate(ISQParser::UGateContext * /*ctx*/) override { }

  virtual void enterVariable(ISQParser::VariableContext * /*ctx*/) override { }
  virtual void exitVariable(ISQParser::VariableContext * /*ctx*/) override { }

  virtual void enterVariableList(ISQParser::VariableListContext * /*ctx*/) override { }
  virtual void exitVariableList(ISQParser::VariableListContext * /*ctx*/) override { }

  virtual void enterBinopPlus(ISQParser::BinopPlusContext * /*ctx*/) override { }
  virtual void exitBinopPlus(ISQParser::BinopPlusContext * /*ctx*/) override { }

  virtual void enterBinopMult(ISQParser::BinopMultContext * /*ctx*/) override { }
  virtual void exitBinopMult(ISQParser::BinopMultContext * /*ctx*/) override { }

  virtual void enterExpression(ISQParser::ExpressionContext * /*ctx*/) override { }
  virtual void exitExpression(ISQParser::ExpressionContext * /*ctx*/) override { }

  virtual void enterMultexp(ISQParser::MultexpContext * /*ctx*/) override { }
  virtual void exitMultexp(ISQParser::MultexpContext * /*ctx*/) override { }

  virtual void enterAtomexp(ISQParser::AtomexpContext * /*ctx*/) override { }
  virtual void exitAtomexp(ISQParser::AtomexpContext * /*ctx*/) override { }

  virtual void enterMExpression(ISQParser::MExpressionContext * /*ctx*/) override { }
  virtual void exitMExpression(ISQParser::MExpressionContext * /*ctx*/) override { }

  virtual void enterAssociation(ISQParser::AssociationContext * /*ctx*/) override { }
  virtual void exitAssociation(ISQParser::AssociationContext * /*ctx*/) override { }

  virtual void enterQbitInitStatement(ISQParser::QbitInitStatementContext * /*ctx*/) override { }
  virtual void exitQbitInitStatement(ISQParser::QbitInitStatementContext * /*ctx*/) override { }

  virtual void enterQbitUnitaryStatement(ISQParser::QbitUnitaryStatementContext * /*ctx*/) override { }
  virtual void exitQbitUnitaryStatement(ISQParser::QbitUnitaryStatementContext * /*ctx*/) override { }

  virtual void enterCintAssign(ISQParser::CintAssignContext * /*ctx*/) override { }
  virtual void exitCintAssign(ISQParser::CintAssignContext * /*ctx*/) override { }

  virtual void enterRegionBody(ISQParser::RegionBodyContext * /*ctx*/) override { }
  virtual void exitRegionBody(ISQParser::RegionBodyContext * /*ctx*/) override { }

  virtual void enterIfStatement(ISQParser::IfStatementContext * /*ctx*/) override { }
  virtual void exitIfStatement(ISQParser::IfStatementContext * /*ctx*/) override { }

  virtual void enterForStatement(ISQParser::ForStatementContext * /*ctx*/) override { }
  virtual void exitForStatement(ISQParser::ForStatementContext * /*ctx*/) override { }

  virtual void enterWhileStatement(ISQParser::WhileStatementContext * /*ctx*/) override { }
  virtual void exitWhileStatement(ISQParser::WhileStatementContext * /*ctx*/) override { }

  virtual void enterCallStatement(ISQParser::CallStatementContext * /*ctx*/) override { }
  virtual void exitCallStatement(ISQParser::CallStatementContext * /*ctx*/) override { }

  virtual void enterPrintStatement(ISQParser::PrintStatementContext * /*ctx*/) override { }
  virtual void exitPrintStatement(ISQParser::PrintStatementContext * /*ctx*/) override { }

  virtual void enterPassStatement(ISQParser::PassStatementContext * /*ctx*/) override { }
  virtual void exitPassStatement(ISQParser::PassStatementContext * /*ctx*/) override { }

  virtual void enterReturnStatement(ISQParser::ReturnStatementContext * /*ctx*/) override { }
  virtual void exitReturnStatement(ISQParser::ReturnStatementContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

