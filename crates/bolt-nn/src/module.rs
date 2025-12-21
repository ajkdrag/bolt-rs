use crate::Result;
use bolt_core::Backend;

pub trait Module<B: Backend>: Send + Sync {
    type Input;
    type Output;

    fn forward(&self, x: Self::Input, train: bool) -> Result<Self::Output>;
}

