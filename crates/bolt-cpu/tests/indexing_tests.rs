use std::sync::Arc;

use bolt_core::Result;
use bolt_core::tensor::Tensor;
use bolt_cpu::CpuBackend;

#[test]
fn test_scalar_indexing_rank_drop() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[2, 2])?;

    // tensor is:
    // [[1, 2],
    //  [3, 4]]

    // x.i(1) should select row 1: [3, 4] and drop rank (2 -> 1)
    let row = tensor.i(1)?;
    assert_eq!(row.shape(), &[2]);
    assert_eq!(row.to_vec()?, vec![3.0, 4.0]);

    // x.i(1).i(0) -> scalar 3.0
    let scalar = row.i(0)?;
    assert_eq!(scalar.shape(), &[]); // scalar tensor (rank 0)
    assert_eq!(scalar.item()?, 3.0);

    Ok(())
}

#[test]
fn test_range_indexing() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[6])?;

    // x.i(1..4) -> [1, 2, 3]
    let slice = tensor.i(1..4)?;
    assert_eq!(slice.shape(), &[3]);
    assert_eq!(slice.to_vec()?, vec![1.0, 2.0, 3.0]);

    Ok(())
}

#[test]
fn test_range_variants() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[6])?;

    // x.i(..2) -> [0, 1]
    let slice = tensor.i(..2)?;
    assert_eq!(slice.shape(), &[2]);
    assert_eq!(slice.to_vec()?, vec![0.0, 1.0]);

    // x.i(4..) -> [4, 5]
    let slice = tensor.i(4..)?;
    assert_eq!(slice.shape(), &[2]);
    assert_eq!(slice.to_vec()?, vec![4.0, 5.0]);

    // x.i(1..=3) -> [1, 2, 3]
    let slice = tensor.i(1..=3)?;
    assert_eq!(slice.shape(), &[3]);
    assert_eq!(slice.to_vec()?, vec![1.0, 2.0, 3.0]);

    // x.i(..) -> all
    let slice = tensor.i(..)?;
    assert_eq!(slice.shape(), &[6]);
    assert_eq!(slice.to_vec()?, data);

    Ok(())
}

#[test]
fn test_tuple_indexing() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let data = (0..12).map(|x| x as f32).collect::<Vec<_>>();
    // Shape [3, 4]
    // [[0, 1, 2, 3],
    //  [4, 5, 6, 7],
    //  [8, 9, 10, 11]]
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[3, 4])?;

    // x.i((1, 2)) -> element at row 1, col 2 -> 6.0. Rank drops 2->0.
    let val = tensor.i((1, 2))?;
    assert_eq!(val.shape(), &[]);
    assert_eq!(val.item()?, 6.0);

    // x.i((1..3, 1..3)) ->
    // [[5, 6],
    //  [9, 10]]
    let slice = tensor.i((1..3, 1..3))?;
    assert_eq!(slice.shape(), &[2, 2]);
    assert_eq!(slice.to_vec()?, vec![5.0, 6.0, 9.0, 10.0]);

    Ok(())
}

#[test]
fn test_partial_tuple_padding() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    // Shape [2, 3, 2]
    let data = (0..12).map(|x| x as f32).collect::<Vec<_>>();
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[2, 3, 2])?;

    // x.i((1)) -> select dim 0 index 1. implicit (1, .., ..)
    // Result shape [3, 2]
    // Original data:
    // Dim 0 idx 0: [[0,1], [2,3], [4,5]]
    // Dim 0 idx 1: [[6,7], [8,9], [10,11]]
    let slice = tensor.i(1)?; // using scalar `usize` impl
    assert_eq!(slice.shape(), &[3, 2]);
    assert_eq!(slice.to_vec()?, vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);

    // x.i((1,)) -> tuple with single element, same result
    let slice_tup = tensor.i((1,))?;
    assert_eq!(slice_tup.shape(), &[3, 2]);
    assert_eq!(slice_tup.to_vec()?, vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);

    // x.i((0, 1..2)) -> implicit (0, 1..2, ..)
    // dim0=0 -> [[0,1], [2,3], [4,5]]
    // dim1=1..2 -> [[2,3]]
    // dim2=.. -> [2, 3]
    let slice = tensor.i((0, 1..2))?;
    assert_eq!(slice.shape(), &[1, 2]); // dim0 dropped, dim1(1), dim2(2)
    assert_eq!(slice.to_vec()?, vec![2.0, 3.0]);

    Ok(())
}

#[test]
fn test_bounds_errors() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::zeros(&backend, &[2, 2])?;

    // Out of bounds scalar
    assert!(tensor.i(2).is_err());
    assert!(tensor.i((0, 5)).is_err());

    // Out of bounds range
    assert!(tensor.i(0..3).is_err());

    // Too many indices
    assert!(tensor.i((0, 0, 0)).is_err());

    Ok(())
}

#[test]
fn test_range_full_in_tuple() -> Result<()> {
     let backend = Arc::new(CpuBackend::new());
    let data = (0..8).map(|x| x as f32).collect::<Vec<_>>();
    // Shape [2, 2, 2]
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[2, 2, 2])?;

    // x.i((.., 1)) -> implicit (.., 1, ..)?
    // Wait, my implementation of tuple assumes sequential mapping.
    // (A, B) maps A to dim 0, B to dim 1.
    // RangeFull (..) maps to "Slice(0..dim)".
    // So (.., 1) means: dim 0 is full slice, dim 1 is selected index 1.
    // Remaining dims (dim 2) are full slices.
    // result: x[:, 1, :]

    // dim 0 idx 0: [[0,1], [2,3]] -> row 1 is [2,3]
    // dim 0 idx 1: [[4,5], [6,7]] -> row 1 is [6,7]
    // result shape: [2, 2] (dim 0 kept, dim 1 dropped, dim 2 kept)
    // result data: [2, 3, 6, 7]

    let slice = tensor.i((.., 1))?;
    assert_eq!(slice.shape(), &[2, 2]);
    assert_eq!(slice.to_vec()?, vec![2.0, 3.0, 6.0, 7.0]);

    Ok(())
}
