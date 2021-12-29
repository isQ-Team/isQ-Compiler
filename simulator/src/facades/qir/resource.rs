use core::{any::Any, marker::PhantomData};

use alloc::{boxed::Box, collections::BTreeMap};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ResourceKey<T: 'static + Any> {
    key: usize,
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
}

// External reference counter to work with QIR reference counting mechanism.

// Trait for id alloction, resource adding.
// We allow ``open'' data type here, i.e. we use typeid/any mechanism.
pub trait ResourceManager {
    fn next(&mut self) -> usize;
    fn add_with_id(&mut self, resource: Box<dyn Any>, id: usize);
    fn get_any(&self, id: usize) -> Option<&dyn Any>;
    fn get_any_mut(&mut self, id: usize) -> Option<&mut dyn Any>;
    fn update_ref_count(&mut self, id: usize, delta: isize);
}
pub trait ResourceManagerExt {
    fn add<T: Any + 'static>(&mut self, resource: T) -> usize;
    fn get<T: Any + 'static>(&self, id: usize) -> Option<&T>;
    fn get_mut<T: Any + 'static>(&mut self, id: usize) -> Option<&mut T>;
    fn get_by_key<T: Any + 'static>(&self, key: ResourceKey<T>) -> Option<&T>;
    fn get_by_key_mut<T: Any + 'static>(&mut self, key: ResourceKey<T>) -> Option<&mut T>;
    fn get_key_checked<T: Any + 'static>(&self, id: usize) -> ResourceKey<T>;
}

impl<R> ResourceManagerExt for R
where
    R: ResourceManager,
{
    fn add<T: Any + 'static>(&mut self, resource: T) -> usize {
        let id = self.next();
        self.add_with_id(Box::new(resource), id);
        id
    }
    fn get<T: Any + 'static>(&self, id: usize) -> Option<&T> {
        self.get_any(id).and_then(|any| any.downcast_ref::<T>())
    }
    fn get_mut<T: Any + 'static>(&mut self, id: usize) -> Option<&mut T> {
        self.get_any_mut(id).and_then(|any| any.downcast_mut::<T>())
    }
    fn get_by_key<T: Any + 'static>(&self, key: ResourceKey<T>) -> Option<&T> {
        self.get_any(key.key)
            .and_then(|any| any.downcast_ref::<T>())
    }
    fn get_by_key_mut<T: Any + 'static>(&mut self, key: ResourceKey<T>) -> Option<&mut T> {
        self.get_any_mut(key.key)
            .and_then(|any| any.downcast_mut::<T>())
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
    resources: BTreeMap<usize, (Box<dyn Any>, isize)>,
    next_id: usize,
}

impl ResourceMap {
    pub fn new() -> Self {
        Self {
            resources: BTreeMap::new(),
            next_id: 1,
        }
    }
}
impl ResourceManager for ResourceMap {
    fn next(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn add_with_id(&mut self, resource: Box<dyn Any>, id: usize) {
        let r = self.resources.insert(id, (resource, 1));
        if r.is_some() {
            panic!("Resource with id {} already exists", id);
        }
    }

    fn get_any(&self, id: usize) -> Option<&dyn Any> {
        self.resources.get(&id).map(|x| &*x.0)
    }

    fn get_any_mut(&mut self, id: usize) -> Option<&mut dyn Any> {
        self.resources.get_mut(&id).map(|x| &mut *x.0)
    }

    fn update_ref_count(&mut self, id: usize, delta: isize) {
        let pair = self
            .resources
            .get_mut(&id)
            .expect(&format!("Resource with id {} not found", id));
        pair.1 += delta;
        if pair.1 < 0 {
            panic!("Reference count of resource #{} dropped below zero", id);
        } else if pair.1 == 0 {
            self.resources.remove(&id);
        }
    }
}

impl ResourceMap {
    // Implementation for the aliased-copy mechanism.
    pub fn try_copy<T: AliasingTracker + 'static>(&mut self, id: usize, force: bool) -> usize {
        let current = self
            .get::<T>(id)
            .expect(&format!("Resource with id {} not found", id));
        if force || current.get_alias_count() > 0 {
            let new_id = self.next();
            let current = self.get::<T>(id).unwrap();
            let new_instance = current.full_copy(new_id);
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
    fn full_copy(&self, allocated_id: usize /* backdoor for tuples */) -> Self;
}
