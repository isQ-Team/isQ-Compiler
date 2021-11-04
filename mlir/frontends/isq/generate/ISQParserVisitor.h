
// Generated from ISQParser.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"
#include "ISQParser.h"



/**
 * This class defines an abstract visitor for a parse tree
 * produced by ISQParser.
 */
class  ISQParserVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by ISQParser.
   */
    virtual antlrcpp::Any visitProgram(ISQParser::ProgramContext *context) = 0;

    virtual antlrcpp::Any visitGateDefclause(ISQParser::GateDefclauseContext *context) = 0;

    virtual antlrcpp::Any visitMatrixvaldef(ISQParser::MatrixvaldefContext *context) = 0;

    virtual antlrcpp::Any visitMatrixdef(ISQParser::MatrixdefContext *context) = 0;

    virtual antlrcpp::Any visitCNumber(ISQParser::CNumberContext *context) = 0;

    virtual antlrcpp::Any visitNumberExpr(ISQParser::NumberExprContext *context) = 0;

    virtual antlrcpp::Any visitVarType(ISQParser::VarTypeContext *context) = 0;

    virtual antlrcpp::Any visitDefclause(ISQParser::DefclauseContext *context) = 0;

    virtual antlrcpp::Any visitIdlistdef(ISQParser::IdlistdefContext *context) = 0;

    virtual antlrcpp::Any visitArraydef(ISQParser::ArraydefContext *context) = 0;

    virtual antlrcpp::Any visitSingleiddef(ISQParser::SingleiddefContext *context) = 0;

    virtual antlrcpp::Any visitProgramBody(ISQParser::ProgramBodyContext *context) = 0;

    virtual antlrcpp::Any visitProcedureBlock(ISQParser::ProcedureBlockContext *context) = 0;

    virtual antlrcpp::Any visitCallParas(ISQParser::CallParasContext *context) = 0;

    virtual antlrcpp::Any visitProcedureMain(ISQParser::ProcedureMainContext *context) = 0;

    virtual antlrcpp::Any visitProcedureBody(ISQParser::ProcedureBodyContext *context) = 0;

    virtual antlrcpp::Any visitQbitinitdef(ISQParser::QbitinitdefContext *context) = 0;

    virtual antlrcpp::Any visitQbitunitarydef(ISQParser::QbitunitarydefContext *context) = 0;

    virtual antlrcpp::Any visitCinassigndef(ISQParser::CinassigndefContext *context) = 0;

    virtual antlrcpp::Any visitIfdef(ISQParser::IfdefContext *context) = 0;

    virtual antlrcpp::Any visitCalldef(ISQParser::CalldefContext *context) = 0;

    virtual antlrcpp::Any visitWhiledef(ISQParser::WhiledefContext *context) = 0;

    virtual antlrcpp::Any visitFordef(ISQParser::FordefContext *context) = 0;

    virtual antlrcpp::Any visitPrintdef(ISQParser::PrintdefContext *context) = 0;

    virtual antlrcpp::Any visitPassdef(ISQParser::PassdefContext *context) = 0;

    virtual antlrcpp::Any visitVardef(ISQParser::VardefContext *context) = 0;

    virtual antlrcpp::Any visitStatementBlock(ISQParser::StatementBlockContext *context) = 0;

    virtual antlrcpp::Any visitUGate(ISQParser::UGateContext *context) = 0;

    virtual antlrcpp::Any visitVariable(ISQParser::VariableContext *context) = 0;

    virtual antlrcpp::Any visitVariableList(ISQParser::VariableListContext *context) = 0;

    virtual antlrcpp::Any visitBinopPlus(ISQParser::BinopPlusContext *context) = 0;

    virtual antlrcpp::Any visitBinopMult(ISQParser::BinopMultContext *context) = 0;

    virtual antlrcpp::Any visitExpression(ISQParser::ExpressionContext *context) = 0;

    virtual antlrcpp::Any visitMultexp(ISQParser::MultexpContext *context) = 0;

    virtual antlrcpp::Any visitAtomexp(ISQParser::AtomexpContext *context) = 0;

    virtual antlrcpp::Any visitMExpression(ISQParser::MExpressionContext *context) = 0;

    virtual antlrcpp::Any visitAssociation(ISQParser::AssociationContext *context) = 0;

    virtual antlrcpp::Any visitQbitInitStatement(ISQParser::QbitInitStatementContext *context) = 0;

    virtual antlrcpp::Any visitQbitUnitaryStatement(ISQParser::QbitUnitaryStatementContext *context) = 0;

    virtual antlrcpp::Any visitCintAssign(ISQParser::CintAssignContext *context) = 0;

    virtual antlrcpp::Any visitRegionBody(ISQParser::RegionBodyContext *context) = 0;

    virtual antlrcpp::Any visitIfStatement(ISQParser::IfStatementContext *context) = 0;

    virtual antlrcpp::Any visitForStatement(ISQParser::ForStatementContext *context) = 0;

    virtual antlrcpp::Any visitWhileStatement(ISQParser::WhileStatementContext *context) = 0;

    virtual antlrcpp::Any visitCallStatement(ISQParser::CallStatementContext *context) = 0;

    virtual antlrcpp::Any visitPrintStatement(ISQParser::PrintStatementContext *context) = 0;

    virtual antlrcpp::Any visitPassStatement(ISQParser::PassStatementContext *context) = 0;

    virtual antlrcpp::Any visitReturnStatement(ISQParser::ReturnStatementContext *context) = 0;


};

