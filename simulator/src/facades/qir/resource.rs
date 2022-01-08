use core::{
    any::Any,
    borrow::{Borrow, BorrowMut},
    cell::{Cell, Ref, RefCell, RefMut},
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use alloc::{boxed::Box, collections::BTreeMap};

use super::context::QIRContext;

#[repr(C)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ResourceKey<T: 'static + Any> {
    pub key: usize,
    _marker: PhantomData<*mut T>,
}
impl<T: 'static + Any> Clone for ResourceKey<T> {
    fn clone(&self) -> Self {
        ResourceKey {
            key: self.key,
            _marker: PhantomData,
        }
    }
}
impl<T: 'static + Any> Copy for ResourceKey<T> {}

impl<T: 'static + Any> ResourceKey<T> {
    pub fn from_raw(x: usize) -> Self {
        ResourceKey {
            key: x,
            _marker: PhantomData,
        }
    }
    pub fn is_null(self) -> bool {
        self.key == 0
    }
    pub fn get<'a, R: Deref<Target = QIRContext>>(
        self,
        context: &'a R,
    ) -> <ResourceMap as ResourceManager>::GetTRet<'a, T> {
        context
            .deref()
            .get_classical_resource_manager()
            .get_by_key(self)
            .expect("QIRContext sanitizer: Resource {} not found")
    }

    pub fn get_mut<'a, R: DerefMut<Target = QIRContext>>(
        self,
        context: &'a R,
    ) -> <ResourceMap as ResourceManager>::GetTMutRet<'a, T> {
        context
            .deref()
            .get_classical_resource_manager()
            .get_by_key_mut(self)
            .expect("QIRContext sanitizer: Resource {} not found")
    }

    pub fn validate<'a, R: Deref<Target = QIRContext>>(self, context: &'a R) {
        context
            .deref()
            .get_classical_resource_manager()
            .get_key_checked::<T>(self.key);
    }
    pub fn update_ref_count<R: Deref<Target = QIRContext>>(self, context: &R, i: isize) {
        self.validate(context);
        context
            .deref()
            .get_classical_resource_manager()
            .update_ref_count(self.key, i)
    }
}
impl<T: 'static + Any + AliasingTracker> ResourceKey<T> {
    pub fn try_copy<'a, R: Deref<Target = QIRContext>>(
        self,
        context: &'a R,
        force: bool,
    ) -> ResourceKey<T> {
        let ctx = context.deref();
        let k = ctx
            .get_classical_resource_manager()
            .try_copy::<T>(self.key, force);
        ResourceKey::from_raw(k)
    }
    pub fn update_alias_count<R: Deref<Target = QIRContext>>(self, context: &R, i: isize) {
        self.validate(context);
        context
            .deref()
            .get_classical_resource_manager()
            .get_by_key(self)
            .expect("QIRContext sanitizer: Resource {} not found")
            .update_alias_count(i)
    }
}

// External reference counter to work with QIR reference counting mechanism.

// Trait for id alloction, resource adding.
// We allow ``open'' data type here, i.e. we use typeid/any mechanism.
pub trait ResourceManager {
    type GetTRet<'a, T>: Deref<Target = T>
    where
        Self: 'a,
        T: 'static + ?Sized;
    type GetTMutRet<'a, T>: DerefMut<Target = T>
    where
        Self: 'a,
        T: 'static + ?Sized;
    //type GetAnyRet<'a>: Deref<Target = dyn Any> where Self: 'a;
    //type GetAnyMutRet<'a>: DerefMut<Target = dyn Any> where Self: 'a;
    fn next(&self) -> usize;
    fn add_with_id(&self, resource: Box<dyn Any>, id: usize);
    fn get_any<'a>(&'a self, id: usize) -> Option<Self::GetTRet<'a, dyn Any + 'static>>;
    fn get_any_mut<'a>(&'a self, id: usize) -> Option<Self::GetTMutRet<'a, dyn Any + 'static>>;
    fn update_ref_count(&self, id: usize, delta: isize);
    fn ret_downcast<'a, T: 'static>(
        obj_any: Self::GetTRet<'a, dyn Any + 'static>,
    ) -> Option<Self::GetTRet<'a, T>>;
    fn ret_downcast_mut<'a, T: 'static>(
        obj_any: Self::GetTMutRet<'a, dyn Any + 'static>,
    ) -> Option<Self::GetTMutRet<'a, T>>;
}
pub trait ResourceManagerExt: ResourceManager {
    fn add<T: Any + 'static>(&self, resource: T) -> usize;
    fn add_key<T: Any + 'static>(&self, resource: T) -> ResourceKey<T>;
    fn get<'a, T: Any + 'static>(&'a self, id: usize) -> Option<Self::GetTRet<'a, T>>;
    fn get_mut<'a, T: Any + 'static>(&'a self, id: usize) -> Option<Self::GetTMutRet<'a, T>>;
    fn get_by_key<'a, T: Any + 'static>(
        &'a self,
        key: ResourceKey<T>,
    ) -> Option<Self::GetTRet<'a, T>>;
    fn get_by_key_mut<'a, T: Any + 'static>(
        &'a self,
        key: ResourceKey<T>,
    ) -> Option<Self::GetTMutRet<'a, T>>;

    fn get_key_checked<T: Any + 'static>(&self, id: usize) -> ResourceKey<T>;
}

impl<R> ResourceManagerExt for R
where
    R: ResourceManager,
{
    fn add<T: Any + 'static>(&self, resource: T) -> usize {
        let id = self.next();
        self.add_with_id(Box::new(resource), id);
        id
    }
    fn add_key<T: Any + 'static>(&self, resource: T) -> ResourceKey<T> {
        ResourceKey {
            key: self.add(resource),
            _marker: PhantomData,
        }
    }
    fn get<'a, T: Any + 'static>(&'a self, id: usize) -> Option<R::GetTRet<'a, T>> {
        self.get_any(id).and_then(|any| Self::ret_downcast(any))
    }
    fn get_mut<'a, T: Any + 'static>(&'a self, id: usize) -> Option<R::GetTMutRet<'a, T>> {
        self.get_any_mut(id)
            .and_then(|any| Self::ret_downcast_mut(any))
    }

    fn get_by_key<'a, T: Any + 'static>(
        &'a self,
        key: ResourceKey<T>,
    ) -> Option<R::GetTRet<'a, T>> {
        self.get(key.key)
    }
    fn get_by_key_mut<'a, T: Any + 'static>(
        &'a self,
        key: ResourceKey<T>,
    ) -> Option<R::GetTMutRet<'a, T>> {
        self.get_mut(key.key)
    }

    fn get_key_checked<T: Any + 'static>(&self, id: usize) -> ResourceKey<T> {
        let key: ResourceKey<T> = ResourceKey::from_raw(id);
        if self.get_by_key(key).is_none() {
            panic!("Resource with id {} not found or has wrong type", id);
        }
        key
    }
}

pub struct ResourceMap {
    resources: RefCell<BTreeMap<usize, (Box<dyn Any>, Cell<isize>)>>,
    next_id: Cell<usize>,
}

impl ResourceMap {
    pub fn new() -> Self {
        Self {
            resources: RefCell::new(BTreeMap::new()),
            next_id: Cell::new(1),
        }
    }
    pub fn leak_check(&self) {
        let mut resources = self.resources.borrow_mut();
        if resources.is_empty() {
            info!("No resource leak detected.");
            return;
        }
        for (id, (_, ref_count)) in resources.iter_mut() {
            if ref_count.get() > 0 {
                error!("Resource {} leaked!", id);
            }
        }
        // resource management bug
        panic!();
    }
}
impl ResourceManager for ResourceMap {
    fn ret_downcast<'a, T: 'static>(
        obj_any: Self::GetTRet<'a, dyn Any + 'static>,
    ) -> Option<Self::GetTRet<'a, T>> {
        let obj = obj_any.borrow().downcast_ref::<T>();
        if obj.is_none() {
            return None;
        }
        Some(Ref::map(obj_any, |x| x.downcast_ref::<T>().unwrap()))
    }
    fn ret_downcast_mut<'a, T: 'static>(
        mut obj_any: Self::GetTMutRet<'a, dyn Any + 'static>,
    ) -> Option<Self::GetTMutRet<'a, T>> {
        let obj = obj_any.borrow_mut().downcast_mut::<T>();
        if obj.is_none() {
            return None;
        }
        Some(RefMut::map(obj_any, |x| x.downcast_mut::<T>().unwrap()))
    }
    fn next(&self) -> usize {
        let id = self.next_id.get();
        self.next_id.set(id + 1);
        id
    }

    fn add_with_id(&self, resource: Box<dyn Any>, id: usize) {
        let r = self
            .resources
            .borrow_mut()
            .insert(id, (resource, Cell::new(1)));
        if r.is_some() {    
            panic!("Resource with id {} already exists", id);
        }
    }
    type GetTRet<'a, T: 'static + ?Sized> = Ref<'a, T>;
    type GetTMutRet<'a, T: 'static + ?Sized> = RefMut<'a, T>;
    fn get_any<'a>(&'a self, id: usize) -> Option<Ref<'a, dyn Any>> {
        let res = self.resources.borrow();
        if res.borrow().get(&id).is_none() {
            return None;
        }
        Some(Ref::map(res, |m| &*m.get(&id).unwrap().0))
    }

    fn get_any_mut<'a>(&'a self, id: usize) -> Option<RefMut<'a, dyn Any>> {
        let mut res = self.resources.borrow_mut();
        if res.borrow_mut().get(&id).is_none() {
            return None;
        }
        Some(RefMut::map(res, |m| &mut *m.get_mut(&id).unwrap().0))
    }

    fn update_ref_count(&self, id: usize, delta: isize) {
        let res_map = self.resources.borrow();
        let pair = res_map
            .get(&id)
            .expect(&format!("Resource with id {} not found", id));
        pair.1.set(pair.1.get() + delta);

        if pair.1.get() < 0 {
            panic!("Reference count of resource #{} dropped below zero", id);
        } else if pair.1.get() == 0 {
            drop(res_map);
            self.resources.borrow_mut().remove(&id);
        }
    }
}

impl ResourceMap {
    // Implementation for the aliased-copy mechanism.
    pub fn try_copy<T: AliasingTracker + 'static>(&self, id: usize, force: bool) -> usize {
        let current = self
            .get::<T>(id)
            .expect(&format!("Resource with id {} not found", id));
        if force || current.get_alias_count() > 0 {
            drop(current);
            let new_id = self.next();
            let current = self.get::<T>(id).unwrap();
            let new_instance = current.full_copy(new_id);
            drop(current);
            self.add_with_id(Box::new(new_instance), new_id);
            new_id
        } else {
            self.update_ref_count(id, 1);
            return id;
        }
    }
}

pub trait AliasingTracker {
    fn get_alias_count(&self) -> usize;
    fn update_alias_count(&self, delta: isize);
    fn full_copy(&self, allocated_id: usize /* backdoor for tuples */) -> Self;
}
