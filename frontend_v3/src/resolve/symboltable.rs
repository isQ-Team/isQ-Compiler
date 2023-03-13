use std::{sync::Arc, collections::BTreeMap};

use crate::{lang::{location::Span, ast::{AST, LAST, Expr, HasSpan}}, error::{FResult, ISQFrontendError}};

use super::package::*;


pub type SymAST<'a> = AST<SymbolInfoAnnotation<'a>, SymbolInfoAnnotation<'a>>;  
pub type SymExpr<'a> = Expr<SymbolInfoAnnotation<'a>>;
pub fn last_to_symast<'a>(orig: Vec<LAST>)->Vec<SymAST<'a>>{
    orig.into_iter().map(|x| x.lift(&SymbolInfoAnnotation::empty, &SymbolInfoAnnotation::empty)).collect()
}


pub struct SymbolInfoAnnotation<'a>{
    pub symbol_info: Option<SymbolInfo<'a>>,
    pub loc: Span
}
impl<'a> SymbolInfoAnnotation<'a>{
    pub fn empty(loc: Span)->Self{
        SymbolInfoAnnotation { symbol_info: None, loc }
    }
}
impl<'a> HasSpan for SymbolInfoAnnotation<'a>{
    fn span(&self)->Span {
        self.loc
    }
}


#[derive(Clone, Copy, Debug)]
pub enum SymbolInfo<'a>{
    // Imported or local module entry.
    GlobalModule(&'a Module),
    // Imported or local symbol entry.
    GlobalSymbol(&'a Symbol),
    // Local value
    Local(usize)
}

impl<'a> SymbolInfo<'a>{
    pub fn from_module_entry(entry: &'a ModuleEntry)->Self{
        match entry{
            ModuleEntry::Module(x) => Self::GlobalModule(x),
            ModuleEntry::Symbol(x) => Self::GlobalSymbol(x),
        }
    }
}


pub struct SymbolTableLayer<'a>{
    //global_modules: BTreeMap<String, SymbolInfo<'a>>,

    /// Modules imported by glob: use foo::*;
    imported_glob_modules: Vec<&'a Module>,
    /// Imported entries and global packages.
    global_symbols: BTreeMap<String, (SymbolInfo<'a>, Span)>,
    /// Local identifiers.
    locals: BTreeMap<String, (SymbolInfo<'a>, Span)>,
    /// local info
    local_info_map: BTreeMap<usize, Span>
}
impl<'a> SymbolTableLayer<'a>{
    pub fn new()->Self{
        Self{
            imported_glob_modules: Vec::new(), 
            global_symbols: BTreeMap::new(), 
            locals: Default::default(),
            local_info_map: Default::default()
        }
    }
    pub fn add_global_module(&mut self, sym: &str, entry: &'a Module, loc: Span)->Result<(), (SymbolInfo<'a>, Span)>{
        if self.global_symbols.contains_key(sym){
            return Err(*self.global_symbols.get(sym).unwrap());
        }else{
            self.global_symbols.insert(sym.to_owned(), (SymbolInfo::GlobalModule(entry), loc));
            return Ok(());
        }
    }
    pub fn add_global_symbol(&mut self, sym: &str, entry: &'a Symbol, loc: Span)->Result<(), (SymbolInfo<'a>, Span)>{
        if self.global_symbols.contains_key(sym){
            return Err(*self.global_symbols.get(sym).unwrap());
        }else{
            self.global_symbols.insert(sym.to_owned(), (SymbolInfo::GlobalSymbol(entry), loc));
            return Ok(());
        }
    }
    pub fn define_local(&mut self, sym: &str, local: usize, location: Span)->Result<usize, (SymbolInfo<'a>, Span)>{
        if self.locals.contains_key(sym){
            return Err(*self.locals.get(sym).unwrap());
        }else{
            self.locals.insert(sym.to_owned(), (SymbolInfo::Local(local), location));
            self.local_info_map.insert(local, location);
            return Ok(local);
        }
    }
    pub fn get_local_info(&self, local: usize)->Option<&Span>{
        self.local_info_map.get(&local)
    }
    // pull top-level declarations (procedure definitions, gate definitions, global imports) into symbol table.
    pub fn pull_declarations(&mut self, global: &'a PackageClosure, current_module: &'a Module, body: &mut Vec<SymAST<'a>>)->FResult<()>{
        for item in body.iter(){
            match &*item.0{
                crate::lang::ast::ASTNode::Gatedef { name, definition } => {
                    let ent = current_module.get_entry(&name.0).unwrap();
                    let sym = ent.as_symbol().unwrap();
                    self.add_global_symbol(&name.0, sym, name.1.span()).map_err(|sym| {
                        ISQFrontendError::redefined_symbol_error(&name.0, name.1.loc, &sym)
                    });
                },
                crate::lang::ast::ASTNode::Import(imp) => {
                    let imported_entries = imp.imports();
                    for import in imported_entries{
                        let (last_part, is_all, alias_name) = match &import.last{
                            crate::lang::ast::ImportPartLast::All(span) => (None, Some(span), None),
                            crate::lang::ast::ImportPartLast::Single { entry, import_as } => (Some(*entry), None, *import_as),
                        };
                        let mut parts_iter = import.prefix.iter().copied().chain(last_part.iter().copied());
                        let package_name = parts_iter.next().unwrap();
                        let (_, mut package) = global.visible_package(package_name.0).ok_or_else(|| {
                            ISQFrontendError::undefined_symbol_error(package_name.0, package_name.1)
                        })?;
                        let mut curr : Result<&Module, &Symbol> = Ok(package.root_module());
                        let mut last_part = ("", Span::empty());
                        for part in parts_iter{
                            last_part = part;
                            match curr{
                                Ok(module) => {
                                    let next = module.get_entry(part.0).ok_or_else(|| {
                                        ISQFrontendError::undefined_module_symbol_error(module, part.0, part.1)
                                    })?;
                                    match next{
                                        ModuleEntry::Module(module) => curr = Ok(module),
                                        ModuleEntry::Symbol(symbol) => curr = Err(symbol),
                                    }
                                }
                                Err(symbol) => {
                                    return Err(ISQFrontendError::not_a_module_error(symbol, part.1));
                                }
                            }
                        }
                        if let Some(span) = is_all{
                            match curr{
                                Ok(module) => {
                                    self.imported_glob_modules.push(module);
                                },
                                Err(symbol) => {
                                    return Err(ISQFrontendError::not_a_module_error(symbol, *span));
                                }
                            }
                        }else{
                            let imported_name = alias_name.unwrap_or(last_part);
                            match curr{
                                
                                Ok(new_module) => {
                                    self.add_global_module(imported_name.0, new_module, imported_name.1).map_err(|sym| {
                                        ISQFrontendError::redefined_symbol_error(&imported_name.0, imported_name.1, &sym)
                                    })?;
                                }
                                Err(new_symbol) => {
                                    self.add_global_symbol(imported_name.0, new_symbol, imported_name.1).map_err(|sym| {
                                        ISQFrontendError::redefined_symbol_error(&imported_name.0, imported_name.1, &sym)
                                    })?;
                                }
                            }
                        }
                        
                    }
                },
                crate::lang::ast::ASTNode::Defvar { definitions }=>{
                    for def in definitions.iter(){
                        let sym = current_module.get_entry(&def.0.var.0).unwrap().as_symbol().unwrap();
                        self.add_global_symbol(&def.0.var.0, sym, def.0.var.1.span()).map_err(|sym| {
                            ISQFrontendError::redefined_symbol_error(&def.0.var.0, def.0.var.1.span(), &sym)
                        })?;
                    }
                }
                crate::lang::ast::ASTNode::Procedure { name, args, body, deriving_clauses } => {
                    let sym = current_module.get_entry(&name.0).unwrap().as_symbol().unwrap();
                    self.add_global_symbol(&name.0, sym, name.1.span()).map_err(|sym| {
                        ISQFrontendError::redefined_symbol_error(&name.0, name.1.span(), &sym)
                    })?;
                },
                _=>{
                    // nope.
                }
            }
        }
        Ok(())
        /*
        for item in prog.iter(){
            match 
        }
        */
    }
    pub fn resolve(&self, name: &str)->Result<SymbolInfo<'a>, ()>{
        if let Some(local) = self.locals.get(name){
            return Ok(local.0);
        }
        if let Some(global) = self.global_symbols.get(name){
            return Ok(global.0);
        }
        let mut glob_matched = vec![];
        for module in self.imported_glob_modules.iter(){
            if let Some(ent) = module.get_entry(name){
                match ent{
                    ModuleEntry::Module(module) => {
                        glob_matched.push(SymbolInfo::GlobalModule(module))
                    },
                    ModuleEntry::Symbol(symbol) => {
                        glob_matched.push(SymbolInfo::GlobalSymbol(symbol))
                    },
                }
            }
        }
        if glob_matched.len()==1{
            return Ok(glob_matched[0])
        }else if glob_matched.len()>1{
            return Err(())
        }
        return Err(());
    }
    
}

pub struct SymbolTable<'a> {
    package: &'a PackageClosure,
    layers: Vec<SymbolTableLayer<'a>>
}

impl<'a> SymbolTable<'a>{
    /**
        Create a new symbol table and import all modules into the symbol table root.
    */
    pub fn new(package: &'a PackageClosure, module_path: Arc<ModulePath>)->Self{
        let mut package_layer = SymbolTableLayer::new();
        let module_layer = SymbolTableLayer::new();
        for external_package in package.visible_dependency_packages(){
            // must be successful.
            package_layer.add_global_module(external_package.0, external_package.2.root_module(), Span::empty()).unwrap();
        }
        // pull all top level declarations to table.


        let table = SymbolTable{
            layers: vec![package_layer, module_layer],
            package
        };
        
        table
    }
    pub fn push_scope(&mut self){
        self.layers.push(SymbolTableLayer::new())
    }
    pub fn pop_scope(&mut self){
        self.layers.pop();
    }
    pub fn peek_scope_mut(&mut self)->&mut SymbolTableLayer<'a>{
        self.layers.last_mut().unwrap()
    }
    pub fn declare_local(&mut self, symname: &str, local: usize, location: Span)->FResult<()>{
        self.peek_scope_mut().define_local(symname, local, location).map_err(|sym| {
            ISQFrontendError::redefined_symbol_error(symname, location, &sym)
        })?;
        Ok(())
    }
    // resolve a symbol recursively.
    pub fn resolve_symbol(&mut self, sym: &str, location: Span)->FResult<SymbolInfo<'a>>{
        for layer in self.layers.iter().rev(){
            let resolved = layer.resolve(sym);
            if let Ok(sym) = resolved{
                return Ok(sym);
            }
        }
        Err(ISQFrontendError::undefined_symbol_error(sym, location))
    }
}