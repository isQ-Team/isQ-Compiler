
// Generated from ISQParser.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"
#include "ISQParserVisitor.h"


/**
 * This class provides an empty implementation of ISQParserVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  ISQParserBaseVisitor : public ISQParserVisitor {
public:

  virtual antlrcpp::Any visitProgram(ISQParser::ProgramContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitGateDefclause(ISQParser::GateDefclauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitMatrixvaldef(ISQParser::MatrixvaldefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitMatrixdef(ISQParser::MatrixdefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCNumber(ISQParser::CNumberContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitNumberExpr(ISQParser::NumberExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitVarType(ISQParser::VarTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDefclause(ISQParser::DefclauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIdlistdef(ISQParser::IdlistdefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitArraydef(ISQParser::ArraydefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSingleiddef(ISQParser::SingleiddefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitProgramBody(ISQParser::ProgramBodyContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitProcedureBlock(ISQParser::ProcedureBlockContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCallParas(ISQParser::CallParasContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitProcedureMain(ISQParser::ProcedureMainContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitProcedureBody(ISQParser::ProcedureBodyContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQbitinitdef(ISQParser::QbitinitdefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQbitunitarydef(ISQParser::QbitunitarydefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCinassigndef(ISQParser::CinassigndefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIfdef(ISQParser::IfdefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCalldef(ISQParser::CalldefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitWhiledef(ISQParser::WhiledefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitFordef(ISQParser::FordefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitPrintdef(ISQParser::PrintdefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitPassdef(ISQParser::PassdefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitVardef(ISQParser::VardefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitStatementBlock(ISQParser::StatementBlockContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitUGate(ISQParser::UGateContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitVariable(ISQParser::VariableContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitVariableList(ISQParser::VariableListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBinopPlus(ISQParser::BinopPlusContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBinopMult(ISQParser::BinopMultContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExpression(ISQParser::ExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitMultexp(ISQParser::MultexpContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAtomexp(ISQParser::AtomexpContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitMExpression(ISQParser::MExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAssociation(ISQParser::AssociationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQbitInitStatement(ISQParser::QbitInitStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQbitUnitaryStatement(ISQParser::QbitUnitaryStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCintAssign(ISQParser::CintAssignContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitRegionBody(ISQParser::RegionBodyContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIfStatement(ISQParser::IfStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitForStatement(ISQParser::ForStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitWhileStatement(ISQParser::WhileStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCallStatement(ISQParser::CallStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitPrintStatement(ISQParser::PrintStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitPassStatement(ISQParser::PassStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitReturnStatement(ISQParser::ReturnStatementContext *ctx) override {
    return visitChildren(ctx);
  }


};

