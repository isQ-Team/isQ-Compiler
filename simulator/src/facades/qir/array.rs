use core::cell::RefCell;

use alloc::alloc::*;
use alloc::vec::Vec;

use super::resource::AliasingTracker;
use super::types::QIRRange;

pub struct QArray {
    // Use machine-alignment.
    data: *mut usize,
    elem_size_in_byte: usize,
    dimensions: Vec<usize>,
    dimension_weights: Vec<usize>,
    alias_counter: usize,
}

impl QArray {
    pub fn new(elem_size_in_byte: usize, dimensions: &[usize]) -> Self {
        for i in dimensions.iter().enumerate() {
            if *i.1 == 0 {
                panic!("QArray dimension {} size is zero", i.0);
            }
        }
        if elem_size_in_byte == 0 {
            panic!("QArray element size is zero");
        }
        let elem_size_in_usize =
            (elem_size_in_byte + core::mem::size_of::<usize>() - 1) / core::mem::size_of::<usize>();
        let size = elem_size_in_usize * dimensions.iter().product::<usize>();
        let data = unsafe {
            let ptr = alloc_zeroed(Layout::array::<usize>(size).unwrap());
            if ptr.is_null() {
                panic!("Out of memory");
            }
            ptr
        };
        let mut dimension_weights = Some(1)
            .into_iter()
            .chain(dimensions.iter().scan(1, |acc, &x| {
                let tmp = *acc;
                *acc *= x;
                Some(tmp)
            }))
            .collect::<Vec<_>>();
        dimension_weights.pop();
        QArray {
            data: data as *mut usize,
            elem_size_in_byte,
            dimensions: dimensions.to_vec(),
            dimension_weights,
            alias_counter: 0,
        }
    }
    fn elem_size_in_usize(&self) -> usize {
        (self.elem_size_in_byte + core::mem::size_of::<usize>() - 1) / core::mem::size_of::<usize>()
    }
    pub fn get_raw(&self) -> *mut usize {
        self.data
    }
    fn total_size(&self) -> usize {
        self.dimensions.iter().product::<usize>()
    }
    pub fn get_dimensions(&self) -> &[usize] {
        &self.dimensions
    }
    pub fn concat_1d(&self, other: &Self) -> Self {
        if self.dimensions.len() != 1 {
            panic!("LHS is not 1d array!");
        }
        if other.dimensions.len() != 1 {
            panic!("RHS is not 1d array!");
        }
        if self.elem_size_in_byte != other.elem_size_in_byte {
            panic!("LHS and RHS have different element size!");
        }
        let arr = Self::new(
            self.elem_size_in_byte,
            &[self.total_size() + other.total_size()],
        );
        unsafe {
            let mut i = 0;
            for j in 0..self.total_size() {
                *arr.data.offset(i) = *self.data.offset(j as isize);
                i += 1;
            }
            for j in 0..other.total_size() {
                *arr.data.offset(i) = *other.data.offset(j as isize);
                i += 1;
            }
        }
        arr
    }
    pub fn get_element(&self, indices: &[i64]) -> *mut i8 {
        if indices.len() != self.dimensions.len() {
            panic!(
                "Index length ({}) does not match array dimension ({})!",
                indices.len(),
                self.dimensions.len()
            );
        }
        let mut offset = 0;
        for i in 0..indices.len() {
            let idx = indices[i] as usize;
            if idx >= self.dimensions[i] {
                panic!("Index ({}) out of range ({})!", idx, self.dimensions[i]);
            }
            offset += idx * self.dimension_weights[i];
        }
        unsafe {
            self.data
                .offset((offset * self.elem_size_in_usize()) as isize) as *mut i8
        }
    }
    pub fn slice(&self, index: usize, i: usize) -> Self {
        let mut arr = self.project(
            index,
            QIRRange {
                start: i as i64,
                end: i as i64,
                step: 1,
            },
        );
        arr.dimensions.remove(index);
        arr.dimension_weights.remove(index);
        arr
    }
    pub fn project(&self, index: usize, range: QIRRange) -> Self {
        if index >= self.dimensions.len() {
            panic!(
                "Dimension ({}) out of range ({})!",
                index,
                self.dimensions.len()
            );
        }
        let mut new_dimensions = self.get_dimensions().to_vec();
        let indices = range.iter().collect::<Vec<_>>();
        for i in indices.iter().copied() {
            if i >= self.dimensions[index] {
                panic!("Index ({}) out of range ({})!", i, self.dimensions[index]);
            }
        }
        new_dimensions[index] = indices.len();
        let new_array = Self::new(self.elem_size_in_byte, &new_dimensions);
        let mut block_size = self.elem_size_in_usize();
        for i in 0..index {
            block_size *= self.dimensions[i];
        }
        let mut block_count = 1;
        for i in (index + 1)..self.dimensions.len() {
            block_count *= self.dimensions[i];
        }

        /*
            indices 0..index index index+1..dimensions
            block_size       parts block_count
        */
        // Of course no aliasing.
        let old_arr = unsafe { core::slice::from_raw_parts_mut(self.data, self.total_size()) };
        let new_arr =
            unsafe { core::slice::from_raw_parts_mut(new_array.data, new_array.total_size()) };
        let old_parts = self.dimensions[index];
        let new_parts = indices.len();
        for i in 0..block_count {
            for (new_index, old_index) in indices.iter().copied().enumerate() {
                let old_base = i * block_size * old_parts + old_index * block_size;
                let new_base = i * block_size * new_parts + new_index * block_size;
                let old_slice = &mut old_arr[old_base..old_base + block_size];
                let new_slice = &mut new_arr[new_base..new_base + block_size];
                new_slice.copy_from_slice(old_slice);
            }
        }
        new_array
    }
}

impl AliasingTracker for QArray {
    fn get_alias_count(&self) -> usize {
        self.alias_counter
    }
    fn full_copy(&self, _allocated_id: usize) -> Self {
        let mut arr = Self::new(self.elem_size_in_byte, &self.dimensions);
        arr.alias_counter = 0;
        unsafe { core::ptr::copy_nonoverlapping(self.data, arr.data, self.total_size()) };
        arr
    }
}

pub type QIRArray = RefCell<QArray>;
