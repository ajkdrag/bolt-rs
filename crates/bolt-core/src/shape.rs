use crate::error::{Error, Result};
use tinyvec::ArrayVec;

pub const MAX_RANK: usize = 12;
pub const MAX_ELEMENTS: usize = isize::MAX as usize;

/// Runtime-validated shape metadata (rank >= 1, every dimension > 0).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConcreteShape {
    dims: ArrayVec<[usize; MAX_RANK]>,
}

impl ConcreteShape {
    pub fn from_slice(dims: &[usize]) -> Result<Self> {
        Ok(Self {
            dims: Self::collect_dims(dims)?,
        })
    }

    pub fn as_slice(&self) -> &[usize] {
        self.dims.as_slice()
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn num_elements(&self) -> usize {
        let mut num = 1usize;
        for &dim in self.dims.iter() {
            num = num
                .checked_mul(dim)
                .expect("shape invariant violated: num_elements overflow");
        }
        num
    }

    pub fn contiguous_strides(&self) -> ArrayVec<[isize; MAX_RANK]> {
        let rank = self.rank();
        let mut strides = ArrayVec::<[isize; MAX_RANK]>::new();
        for _ in 0..rank {
            strides.push(0);
        }
        let mut stride = 1isize;
        for i in (0..rank).rev() {
            let dim = self.dims[i];
            strides[i] = stride;
            stride *= dim as isize;
        }
        strides
    }

    fn collect_dims(dims: &[usize]) -> Result<ArrayVec<[usize; MAX_RANK]>> {
        if dims.len() > MAX_RANK {
            return Err(Error::invalid_shape(format!(
                "tensor rank must be <= {MAX_RANK}"
            )));
        }
        let mut collected = ArrayVec::<[usize; MAX_RANK]>::new();
        let mut num_elements = 1u128;
        for &dim in dims {
            if dim == 0 {
                return Err(Error::invalid_shape(
                    "zero-sized dimensions are not supported",
                ));
            }
            num_elements = num_elements
                .checked_mul(dim as u128)
                .ok_or(Error::TensorTooLarge {
                    limit: MAX_ELEMENTS,
                    requested: usize::MAX,
                })?;
            if num_elements > MAX_ELEMENTS as u128 {
                return Err(Error::TensorTooLarge {
                    limit: MAX_ELEMENTS,
                    requested: num_elements.try_into().unwrap_or(usize::MAX),
                });
            }
            collected.push(dim);
        }
        Ok(collected)
    }
}

impl From<&ConcreteShape> for ConcreteShape {
    fn from(value: &ConcreteShape) -> Self {
        value.clone()
    }
}

impl TryFrom<Vec<usize>> for ConcreteShape {
    type Error = Error;

    fn try_from(dims: Vec<usize>) -> Result<Self> {
        ConcreteShape::from_slice(&dims)
    }
}

impl TryFrom<&[usize]> for ConcreteShape {
    type Error = Error;

    fn try_from(dims: &[usize]) -> Result<Self> {
        ConcreteShape::from_slice(dims)
    }
}

pub fn broadcast_shapes(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>> {
    let mut shape = Vec::with_capacity(lhs.len().max(rhs.len()));
    let mut l_iter = lhs.iter().rev();
    let mut r_iter = rhs.iter().rev();
    loop {
        let l = l_iter.next();
        let r = r_iter.next();
        match (l, r) {
            (None, None) => break,
            (Some(&lv), None) | (None, Some(&lv)) => {
                shape.push(lv);
            }
            (Some(&lv), Some(&rv)) => {
                if lv == rv {
                    shape.push(lv);
                } else if lv == 1 {
                    shape.push(rv);
                } else if rv == 1 {
                    shape.push(lv);
                } else {
                    return Err(Error::ShapeMismatch {
                        lhs: lhs.to_vec(),
                        rhs: rhs.to_vec(),
                    });
                }
            }
        }
    }
    shape.reverse();
    Ok(shape)
}

pub fn canonical_axes(axes: &[usize], rank: usize) -> Result<Vec<usize>> {
    if axes.is_empty() {
        return Ok((0..rank).collect());
    }
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(Error::InvalidAxes(format!(
                "axis {axis} out of bounds for rank {rank}"
            )));
        }
        if seen[axis] {
            return Err(Error::InvalidAxes(format!("axis {axis} provided twice")));
        }
        seen[axis] = true;
    }
    let mut out = axes.to_vec();
    out.sort_unstable();
    Ok(out)
}

pub fn reduced_shape(shape: &[usize], axes: &[usize]) -> Result<Vec<usize>> {
    let canonical = canonical_axes(axes, shape.len())?;
    let mut result = Vec::with_capacity(shape.len().saturating_sub(canonical.len()));
    for (idx, dim) in shape.iter().enumerate() {
        if canonical.binary_search(&idx).is_err() {
            result.push(*dim);
        }
    }
    Ok(result)
}
