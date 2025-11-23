use crate::{
    error::{Error, Result},
    layout::TensorIndexer,
};
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo};

pub trait TensorIndex {
    fn to_indexers(&self, shape: &[usize]) -> Result<Vec<TensorIndexer>>;
}

impl TensorIndex for usize {
    fn to_indexers(&self, _shape: &[usize]) -> Result<Vec<TensorIndexer>> {
        Ok(vec![TensorIndexer::Select(*self)])
    }
}

impl TensorIndex for Range<usize> {
    fn to_indexers(&self, _shape: &[usize]) -> Result<Vec<TensorIndexer>> {
        Ok(vec![TensorIndexer::Slice {
            start: self.start,
            end: self.end,
            step: 1,
        }])
    }
}

impl TensorIndex for RangeInclusive<usize> {
    fn to_indexers(&self, _shape: &[usize]) -> Result<Vec<TensorIndexer>> {
        let start = *self.start();
        let end = self
            .end()
            .checked_add(1)
            .ok_or_else(|| Error::invalid_shape("RangeInclusive end overflow"))?;
        Ok(vec![TensorIndexer::Slice {
            start,
            end,
            step: 1,
        }])
    }
}

impl TensorIndex for RangeFrom<usize> {
    fn to_indexers(&self, shape: &[usize]) -> Result<Vec<TensorIndexer>> {
        if shape.is_empty() {
             return Err(Error::invalid_shape("RangeFrom on 0-rank tensor or indexer mismatch"));
             // Wait, RangeFrom applies to a dimension.
             // If we are just returning the indexer, we assume it corresponds to the *first* available dimension
             // in the context of a single item impl, but `to_indexers` is conceptually "convert this object to a list of indexers for the tensor".
             // If I implement TensorIndex for a single RangeFrom, it implies it's indexing the first dimension (and only the first).
             // But I need the size of THAT dimension.
             // The `shape` argument is the Full tensor shape.
             // So for a single indexer, it consumes shape[0].
        }
        let dim = shape[0];
        Ok(vec![TensorIndexer::Slice {
            start: self.start,
            end: dim,
            step: 1,
        }])
    }
}

impl TensorIndex for RangeTo<usize> {
    fn to_indexers(&self, _shape: &[usize]) -> Result<Vec<TensorIndexer>> {
        Ok(vec![TensorIndexer::Slice {
            start: 0,
            end: self.end,
            step: 1,
        }])
    }
}

impl TensorIndex for RangeFull {
    fn to_indexers(&self, shape: &[usize]) -> Result<Vec<TensorIndexer>> {
        if shape.is_empty() {
             return Ok(vec![]);
        }

        let dim = shape[0];
        Ok(vec![TensorIndexer::Slice {
            start: 0,
            end: dim,
            step: 1,
        }])
    }
}


macro_rules! impl_tuple_index {
    ($($T:ident),+) => {
        impl<$($T),+> TensorIndex for ($($T,)+)
        where
            $($T: TensorIndex),+
        {
            fn to_indexers(&self, shape: &[usize]) -> Result<Vec<TensorIndexer>> {
                let mut indexers = Vec::new();
                let mut current_dim = 0;

                #[allow(non_snake_case)]
                let ($($T,)+) = self;

                $(
                    if current_dim < shape.len() {
                        let sub_shape = &shape[current_dim..];
                        let sub_indexers = $T.to_indexers(sub_shape)?;
                        let count = sub_indexers.len();
                        indexers.extend(sub_indexers);
                        current_dim += count;
                    } else {
                        let sub_indexers = $T.to_indexers(&[])?;
                        let count = sub_indexers.len();
                        indexers.extend(sub_indexers);
                        current_dim += count;
                    }
                )+
                // Suppress unused warning for the last assignment
                let _ = current_dim;
                Ok(indexers)
            }
        }
    }
}

impl_tuple_index!(A);
impl_tuple_index!(A, B);
impl_tuple_index!(A, B, C);
impl_tuple_index!(A, B, C, D);
impl_tuple_index!(A, B, C, D, E);
impl_tuple_index!(A, B, C, D, E, F);
