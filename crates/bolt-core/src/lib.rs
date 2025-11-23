#![deny(unused_must_use)]

pub mod allocator;
pub mod backend;
pub mod device;
mod display;
pub mod dtype;
pub mod error;
pub mod index;
pub mod layout;
pub mod shape;
pub mod storage;
pub mod tensor;
pub(crate) mod utils;

pub use allocator::StorageAllocator;
pub use backend::{Backend, TensorParts};
pub use device::DeviceKind;
pub use dtype::{DType, NativeType, OneValue, ToF32};
pub use error::{Error, Result};
pub use index::TensorIndex;
pub use layout::{Layout, LayoutKind, TensorIndexer};
pub use storage::{BufferHandle, TensorView};
pub use tensor::Tensor;
