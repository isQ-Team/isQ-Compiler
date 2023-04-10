use std::sync::Arc;

use itertools::Itertools;

use crate::{lang::ast::{LAST, AST, ASTNode, HasSpan}, error::ISQFrontendError};

use self::{package::{PackageClosure, Module, ModulePath, PackageMetadata}, symboltable::{SymbolInfo, SymAST, SymbolInfoAnnotation, SymbolTable}, resolver::Resolver};

use crate::error::FResult;
pub mod package;
pub mod symboltable;
pub mod resolver;

/*
 * isQ Symbol resolution steps:
 * Step 1: collect all depenedencies of the package.
 * Step 2: collect all definitions in all modules of current package.
 * Step 3: generate PackageClosure from the two results above.
 * Step 4: resolve all symbols in all modules against the PackageClosure.
 */


pub fn collect_definitions(package: &mut PackageClosure, path: &Arc<ModulePath>, prog: &Vec<SymAST>)->FResult<()>{
    let module = package.me_mut();
    let submodule = module.resolve_module_entry(path).unwrap();
    collect_definitions_module(submodule, prog)
}

fn collect_definitions_module(module: &mut Module, prog: &Vec<SymAST>)->FResult<()>{
    for item in prog.iter(){
        match &*item.0{
            ASTNode::Defvar { definitions }=> {
                for def in definitions.iter(){
                    let sym = module.insert_symbol(&def.0.var.0, true, def.0.var.1.span()).map_err(|sym| {
                        let symbol_name = &def.0.var.0;
                        let defined_here  = def.0.var.1.span();
                        return ISQFrontendError::redefined_symbol_with_definition_error(symbol_name, defined_here, &SymbolInfo::from_module_entry(sym));
                        
                    })?;
                }
            }
            ASTNode::Gatedef { name, definition } => {
                let ident = name;
                let sym = module.insert_symbol(&ident.0, true, ident.1.span()).map_err(|sym| {
                    let symbol_name = &ident.0;
                    let defined_here  = ident.1.span();
                    return ISQFrontendError::redefined_symbol_with_definition_error(symbol_name, defined_here, &SymbolInfo::from_module_entry(sym));
                })?;
            }
            ASTNode::Procedure { name, args, body, deriving_clauses, return_type }=>{
                let ident = name;
                let sym = module.insert_symbol(&ident.0, true, ident.1.span()).map_err(|sym| {
                    let symbol_name = &ident.0;
                    let defined_here  = ident.1.span();
                    return ISQFrontendError::redefined_symbol_with_definition_error(symbol_name, defined_here, &SymbolInfo::from_module_entry(sym));
                })?;
            }
            _ => {

            }
        }
    }
    Ok(())
}

/** 
 * Entry for symbol resolution of one package.
 */
pub fn resolve_all_modules<'a>(env: &'a mut PackageClosure,
    modules: Vec<(Arc<ModulePath>, Vec<LAST>)>
)->FResult<Vec<(Arc<ModulePath>, Vec<SymAST<'a>>)>>{
    let mut modules = modules.into_iter().map(|x| (x.0, x.1.into_iter().map(|y| y.lift(&SymbolInfoAnnotation::empty, &SymbolInfoAnnotation::empty)).collect_vec())).collect_vec();
    let me = env.me_mut();
    // first, build the module tree.
    for (path, body) in modules.iter(){
        let module = me.resolve_module_entry(&*path).unwrap();
    }
    // second, collect all declarations into module declarations.
    // this step makes all declarations in all modules in the packages visible to each other.
    for (path, body) in modules.iter(){
        let module = me.resolve_module_entry(&*path).unwrap();
        collect_definitions_module(module, body)?;
    }
    // last, resolve symbols for every module.
    // this step first makes all declaration (and imports) in one module visible to all statements, and then resolve all usages.
    for (path, body) in modules.iter_mut(){
        let symbol_table : SymbolTable<'a> = SymbolTable::new(env, path.clone(), body)?;
        let mut resolver = Resolver::new(symbol_table);
        resolver.resolve_program(body)?;
    }
    Ok(modules)
}
