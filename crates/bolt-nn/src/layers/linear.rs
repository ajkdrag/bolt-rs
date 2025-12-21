use crate::{Error, Init, Module, Param, Result, Store};
use bolt_autodiff::Tensor;
use bolt_core::Backend;

pub struct Linear<B: Backend> {
    w: Param<B>,
    b: Option<Param<B>>,
    in_f: usize,
}

impl<B: Backend> Linear<B> {
    pub fn init(store: &Store<B>, in_f: usize, out_f: usize, bias: bool) -> Result<Self> {
        let w = store.param("weight", &[in_f, out_f], Init::KaimingUniform { a: 0.0 })?;
        let b = if bias {
            Some(store.group(1).param("bias", &[out_f], Init::Zeros)?)
        } else {
            None
        };
        Ok(Self { w, b, in_f })
    }
}

impl<B: Backend> Module<B> for Linear<B> {
    type Input = Tensor<B>;
    type Output = Tensor<B>;

    fn forward(&self, x: Self::Input, _train: bool) -> Result<Self::Output> {
        let s = x.shape();
        if s.len() != 2 || s[1] != self.in_f {
            return Err(Error::Shape(format!(
                "Linear: expected [batch, {}], got {s:?}",
                self.in_f
            )));
        }

        let y = x
            .matmul(&self.w.tensor())
            .map_err(|e| Error::Shape(e.to_string()))?;
        let y = match &self.b {
            None => y,
            Some(b) => y.add(&b.tensor()).map_err(|e| Error::Shape(e.to_string()))?,
        };
        Ok(y)
    }
}
