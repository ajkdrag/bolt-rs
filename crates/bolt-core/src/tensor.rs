use std::{marker::PhantomData, sync::Arc};

use crate::{
    allocator::StorageAllocator,
    backend::Backend,
    dtype::{NativeType, OneValue, ToF32},
    error::{Error, Result},
    index::TensorIndex,
    layout::Layout,
    shape::ConcreteShape,
    utils::tensor_creation,
};

#[derive(Clone)]
pub struct Tensor<B, D>
where
    B: Backend<D>,
    D: NativeType,
{
    backend: Arc<B>,
    storage: B::Storage,
    layout: Layout,
    _marker: PhantomData<D>,
}

impl<B, D> Tensor<B, D>
where
    B: Backend<D>,
    D: NativeType,
{
    pub fn from_slice(backend: &Arc<B>, data: &[D], shape: &[usize]) -> Result<Self> {
        let shape = ConcreteShape::from_slice(shape)?;
        if shape.num_elements() != data.len() {
            return Err(Error::SizeMismatch {
                expected: shape.num_elements(),
                actual: data.len(),
            });
        }
        let layout = Layout::contiguous(shape);
        let mut storage = backend.allocator().allocate(data.len())?;
        backend.write(&mut storage, &layout, data)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn zeros(backend: &Arc<B>, shape: &[usize]) -> Result<Self> {
        let shape = ConcreteShape::from_slice(shape)?;
        let numel = shape.num_elements();
        let layout = Layout::contiguous(shape);
        let storage = backend.allocator().allocate_zeroed(numel)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn ones(backend: &Arc<B>, shape: &[usize]) -> Result<Self>
    where
        D: OneValue,
    {
        Self::full(backend, shape, tensor_creation::one_value::<D>())
    }

    pub fn full(backend: &Arc<B>, shape: &[usize], value: D) -> Result<Self> {
        let shape = ConcreteShape::from_slice(shape)?;
        let layout = Layout::contiguous(shape);
        let storage = backend.fill(&layout, value)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn ones_like(other: &Tensor<B, D>) -> Result<Self>
    where
        D: OneValue,
    {
        let layout = Layout::with_strides(
            other.layout.concrete_shape().clone(),
            other.layout.strides(),
            0,
        )?;
        let storage = other
            .backend
            .fill(&layout, tensor_creation::one_value::<D>())?;
        let tensor = Self::from_parts(other.backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn full_like(other: &Tensor<B, D>, value: D) -> Result<Self> {
        let layout = Layout::with_strides(
            other.layout.concrete_shape().clone(),
            other.layout.strides(),
            0,
        )?;
        let storage = other.backend.fill(&layout, value)?;
        let tensor = Self::from_parts(other.backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn arange(backend: &Arc<B>, start: D, end: D, step: D) -> Result<Self> {
        let len = tensor_creation::compute_arange_len(start, end, step)?;
        let shape = ConcreteShape::from_slice(&[len])?;
        let layout = Layout::contiguous(shape);
        let mut storage = backend.allocator().allocate(len)?;
        let values = tensor_creation::build_arange_values(len, start, step)?;
        backend.write(&mut storage, &layout, &values)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn backend(&self) -> Arc<B> {
        self.backend.clone()
    }

    pub fn device(&self) -> &B::Device {
        self.backend.device()
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn numel(&self) -> usize {
        self.layout.num_elements()
    }

    pub fn to_vec(&self) -> Result<Vec<D>> {
        let tensor = if self.layout.is_contiguous() && self.layout.offset_bytes() == 0 {
            self.clone()
        } else {
            self.contiguous()?
        };
        let mut values = vec![D::default(); tensor.numel()];
        tensor
            .backend
            .read(&tensor.storage, &tensor.layout, &mut values)?;
        Ok(values)
    }

    pub fn item(&self) -> Result<D> {
        if self.numel() != 1 {
            return Err(Error::invalid_shape(
                "item requires a tensor with exactly one element",
            ));
        }
        let mut value = [D::default(); 1];
        self.backend.read(&self.storage, &self.layout, &mut value)?;
        Ok(value[0])
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        let shape = ConcreteShape::from_slice(shape)?;
        let layout = self.layout.reshape(shape)?;
        self.validate_layout_for_storage(&self.storage, &layout)?;
        Ok(self.with_layout(layout))
    }

    pub fn i<I: TensorIndex>(&self, index: I) -> Result<Self> {
        let indexers = index.to_indexers(self.shape())?;
        let layout = self.layout.perform_indexing(&indexers, D::DTYPE)?;
        self.validate_layout_for_storage(&self.storage, &layout)?;
        Ok(self.with_layout(layout))
    }

    pub fn slice(&self, axis: usize, start: usize, end: usize, step: usize) -> Result<Self> {
        let layout = self.layout.slice(axis, start, end, step, D::DTYPE)?;
        self.validate_layout_for_storage(&self.storage, &layout)?;
        Ok(self.with_layout(layout))
    }

    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        let layout = self.layout.permute(axes)?;
        self.validate_layout_for_storage(&self.storage, &layout)?;
        Ok(self.with_layout(layout))
    }

    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<Self> {
        let layout = self.layout.transpose(axis_a, axis_b)?;
        self.validate_layout_for_storage(&self.storage, &layout)?;
        Ok(self.with_layout(layout))
    }

    pub fn contiguous(&self) -> Result<Self> {
        if self.layout.is_contiguous() && self.layout.offset_bytes() == 0 {
            return Ok(self.clone());
        }
        let parts = self.backend.copy(&self.storage, &self.layout)?;
        self.validate_layout_for_storage(&parts.storage, &parts.layout)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        self.ensure_same_backend(other)?;
        let parts = self
            .backend
            .add(&self.storage, &other.storage, &self.layout, &other.layout)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        self.ensure_same_backend(other)?;
        let parts = self
            .backend
            .sub(&self.storage, &other.storage, &self.layout, &other.layout)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn matmul(&self, other: &Self) -> Result<Self> {
        self.ensure_same_backend(other)?;
        let parts =
            self.backend
                .matmul(&self.storage, &other.storage, &self.layout, &other.layout)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn mean_f32(&self) -> Result<Tensor<B, f32>>
    where
        B: Backend<f32>,
        D: ToF32,
    {
        let parts = Backend::<D>::mean_f32(self.backend.as_ref(), &self.storage, &self.layout)?;
        Ok(Tensor::<B, f32>::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    fn ensure_same_backend(&self, other: &Self) -> Result<()> {
        if self.backend.device_kind() != other.backend.device_kind() {
            return Err(Error::Device("tensors belong to different devices".into()));
        }
        if !Arc::ptr_eq(&self.backend, &other.backend) {
            return Err(Error::Device(
                "tensors belong to different backend instances".into(),
            ));
        }
        Ok(())
    }

    fn validate_layout_for_storage(&self, storage: &B::Storage, layout: &Layout) -> Result<()> {
        let len_bytes = self.backend.storage_len_bytes(storage);
        layout.validate_bounds(D::DTYPE, len_bytes)
    }

    fn with_layout(&self, layout: Layout) -> Self {
        Self {
            backend: self.backend.clone(),
            storage: self.storage.clone(),
            layout,
            _marker: PhantomData,
        }
    }

    fn from_parts(backend: Arc<B>, storage: B::Storage, layout: Layout) -> Self {
        Self {
            backend,
            storage,
            layout,
            _marker: PhantomData,
        }
    }

    pub fn storage(&self) -> &B::Storage {
        &self.storage
    }
}
