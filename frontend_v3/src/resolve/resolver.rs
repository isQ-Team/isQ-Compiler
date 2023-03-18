use itertools::Itertools;

use crate::{lang::{ast::{ASTBlock, VarDef, HasSpan}, location::Span}, error::{FResult, ISQFrontendError}};

use super::symboltable::{SymAST, SymbolTable, SymbolInfoAnnotation, SymExpr, SymbolInfo};

/**
 * Name resolver module.
 */

pub struct Resolver<'a>{
    local_counter: usize,
    symbol_table: SymbolTable<'a>,
}

impl<'a> Resolver<'a>{
    pub fn new(symbol_table: SymbolTable<'a>)->Self{
        Resolver { 
            symbol_table,
            local_counter: 0,
        }
    }
    pub fn next_local(&mut self)->usize{
        let id = self.local_counter;
        self.local_counter+=1;
        id
    }
    fn def_local(&mut self, sym: &str, location: Span)->FResult<SymbolInfo<'a>>{
        let id = self.next_local();
        self.symbol_table.declare_local(sym, id, location)?;
        Ok(SymbolInfo::Local(id))
    }
    fn push_scope(&mut self){
        self.symbol_table.push_scope();
    }
    fn pop_scope(&mut self){
        self.symbol_table.pop_scope();
    }
    fn visit_block_noscope(&mut self, block: &mut ASTBlock<SymbolInfoAnnotation<'a>, SymbolInfoAnnotation<'a>>)->FResult<()>{
        for stmt in block.0.iter_mut(){
            self.visit_statement(stmt)?;
        }
        return Ok(())
    }
    fn visit_block_scoped_init<F: FnOnce(&mut Self)->FResult<()>>(&mut self, block: &mut ASTBlock<SymbolInfoAnnotation<'a>, SymbolInfoAnnotation<'a>>, init: F)->FResult<()>{
        self.push_scope();
        init(self)?;
        self.visit_block_noscope(block)?;
        self.pop_scope();
        return Ok(())
    }
    fn visit_block_scoped(&mut self, block: &mut ASTBlock<SymbolInfoAnnotation<'a>, SymbolInfoAnnotation<'a>>)->FResult<()>{
        self.visit_block_scoped_init(block, |_|{Ok(())})
    }
    fn visit_expr(&mut self, expr: &mut SymExpr<'a>)->FResult<()>{
        match &mut *expr.0{
            crate::lang::ast::ExprNode::Qualified(qname) => {
                let mut x = self.symbol_table.resolve_symbol(&qname.0[0].0, qname.1.span())?;
                for part in &qname.0[1..]{
                    match x{
                        super::symboltable::SymbolInfo::GlobalModule(m) => {
                            let next = m.get_entry(&part.0).ok_or(ISQFrontendError::undefined_module_symbol_error(m, &part.0, part.1.span()))?;
                            match next{
                                super::package::ModuleEntry::Module(next_module) => {
                                    x = SymbolInfo::GlobalModule(next_module);
                                },
                                super::package::ModuleEntry::Symbol(next_symbol) => {
                                    x = SymbolInfo::GlobalSymbol(next_symbol);
                                },
                            }
                        },
                        super::symboltable::SymbolInfo::GlobalSymbol(sym) => {
                            return Err(ISQFrontendError::NotAModule { qualified_name: sym.qualified_name(), used_here: part.1.span(), defined_here: sym.definition_location() });
                        }
                        super::symboltable::SymbolInfo::Local(id) => {
                            let local_span = self.symbol_table.peek_scope_mut().get_local_info(id).unwrap();
                            // not a module.
                            return Err(ISQFrontendError::NotAModule { qualified_name: part.0.clone(), used_here: part.1.span(), defined_here: local_span.to_owned() });
                        }
                    }   
                }
                if let SymbolInfo::GlobalModule(module) = x{
                    return Err(ISQFrontendError::name_is_module_error(&qname.0.iter().map(|x| &x.0).join("::"), module, qname.1.span()));
                }
                expr.1.symbol_info = Some(x);
                
            },
            crate::lang::ast::ExprNode::Binary { op, lhs, rhs } => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
            },
            crate::lang::ast::ExprNode::Paren(body) => {
                self.visit_expr(body)?;
            },
            crate::lang::ast::ExprNode::Unary { op, arg } => {
                self.visit_expr(arg)?;
            },
            crate::lang::ast::ExprNode::Subscript { base, offset } => {
                self.visit_expr(base)?;
                self.visit_expr(offset)?;
            },
            crate::lang::ast::ExprNode::Call { callee, args } => {
                self.visit_expr(callee)?;
                for arg in args.iter_mut(){
                    self.visit_expr(arg)?;
                }
                
            },
            crate::lang::ast::ExprNode::LitInt(_) => {},
            crate::lang::ast::ExprNode::LitFloat(_) => {},
            crate::lang::ast::ExprNode::LitImag(_) => {},
            crate::lang::ast::ExprNode::LitBool(_) => {},
            crate::lang::ast::ExprNode::Range { lo, hi, step } => {
                if let Some(x) = lo{
                    self.visit_expr(x)?;
                }
                if let Some(x) = hi{
                    self.visit_expr(x)?;
                }
                if let Some(x) = step{
                    self.visit_expr(x)?;
                }
            },
            crate::lang::ast::ExprNode::List(xs) => {
                for x in xs.iter_mut(){
                    self.visit_expr(x)?;
                }
            },
            crate::lang::ast::ExprNode::Unit => {},
        }
        Ok(())
    }
    fn visit_vardef(&mut self, vardef: &mut VarDef<SymbolInfoAnnotation<'a>>)->FResult<()>{
        let id = self.def_local(&vardef.var.0, vardef.var.1.span())?;
        vardef.var.1.symbol_info = Some(id);
        Ok(())
    }
    fn visit_statement(&mut self, ast: &mut SymAST<'a>)->FResult<()>{
        match &mut *ast.0{
            crate::lang::ast::ASTNode::Block(block) => {
                self.visit_block_scoped(block)?;
            }
            crate::lang::ast::ASTNode::Expr(expr) => {
                self.visit_expr(expr)?;
            },
            crate::lang::ast::ASTNode::If { condition, then_block, else_block } => {
                self.visit_expr(condition)?;
                self.visit_block_scoped(then_block)?;
                if let Some(blk) = else_block{
                    self.visit_block_scoped(blk)?;
                }
            },
            crate::lang::ast::ASTNode::For { var, range, body } => {
                self.visit_expr(range)?;
                self.visit_block_scoped_init(body, |self_|{
                    let local = self_.def_local(&var.0, var.1.span())?;
                    var.1.symbol_info = Some(local);
                    return Ok(());
                })?;
            },
            crate::lang::ast::ASTNode::While { condition, body } => {
                self.visit_expr(condition)?;
                self.visit_block_scoped(body)?;
            },
            crate::lang::ast::ASTNode::Defvar { definitions } => {
                for vardef in definitions.iter_mut(){
                    self.visit_vardef(&mut vardef.0)?;
                    if let Some(expr) = &mut vardef.1{
                        self.visit_expr(expr)?;
                    }
                    
                }
            },
            crate::lang::ast::ASTNode::Assign { lhs, rhs } => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
            },
            crate::lang::ast::ASTNode::Gatedef { name, definition } => {
                self.visit_expr(definition)?;
                let local = self.def_local(&name.0, name.1.span())?;
                name.1.symbol_info = Some(local);
            },
            crate::lang::ast::ASTNode::Unitary { modifiers, call } => {
                self.visit_expr(call)?;
            },
            crate::lang::ast::ASTNode::Package(_) => {
                
            },
            crate::lang::ast::ASTNode::Import(_) => {},
            crate::lang::ast::ASTNode::Procedure { name, args, body, deriving_clauses, return_type } => {
                self.visit_block_scoped_init(body, |self_| {
                    for arg in args.iter_mut(){
                        self_.visit_vardef(arg)?;
                    }
                    return Ok(());
                })?;
            },
            crate::lang::ast::ASTNode::Pass => {

            },
            crate::lang::ast::ASTNode::Return(expr) => {
                if let Some(expr) = expr{
                    self.visit_expr(expr)?;
                }
            },
            crate::lang::ast::ASTNode::Continue => {

            },
            crate::lang::ast::ASTNode::Break => {

            },
            crate::lang::ast::ASTNode::Empty => {

            },
        }
        Ok(())
    }
    pub fn resolve_program<'b, 'c>(&'b mut self, program: &'c mut Vec<SymAST<'a>>)->FResult<()>{
        for stmt in program.iter_mut(){
            self.visit_statement(stmt)?;
        }
        Ok(())
    }

}