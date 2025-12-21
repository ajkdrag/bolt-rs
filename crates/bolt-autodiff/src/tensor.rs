use crate::grad::grad_enabled;
use bolt_core::Backend;
use std::{
    collections::HashSet,
    marker::PhantomData,
    sync::{Arc, Mutex},
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("shape error: {0}")]
    Shape(String),
    #[error("backward error: {0}")]
    Backward(String),
}

type BackwardFn = Arc<dyn Fn(&[f32]) + Send + Sync>;

struct Inner<B: Backend> {
    data: Vec<f32>,
    shape: Vec<usize>,
    grad: Option<Vec<f32>>,
    requires_grad: bool,
    parents: Vec<Tensor<B>>,
    backward: Option<BackwardFn>,
}

#[derive(Clone)]
pub struct Tensor<B: Backend> {
    inner: Arc<Mutex<Inner<B>>>,
    _b: PhantomData<B>,
}

impl<B: Backend> Tensor<B> {
    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Result<Self, TensorError> {
        let numel: usize = shape.iter().product();
        if numel != data.len() {
            return Err(TensorError::Shape(format!(
                "from_vec: expected numel {numel}, got data.len {}",
                data.len()
            )));
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                data,
                shape: shape.to_vec(),
                grad: None,
                requires_grad: false,
                parents: Vec::new(),
                backward: None,
            })),
            _b: PhantomData,
        })
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        Self::from_vec(shape, vec![0.0; numel]).expect("zeros: shape valid")
    }

    pub fn shape(&self) -> Vec<usize> {
        self.inner.lock().unwrap().shape.clone()
    }

    pub fn numel(&self) -> usize {
        self.inner.lock().unwrap().data.len()
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.inner.lock().unwrap().data.clone()
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.lock().unwrap().requires_grad
    }

    pub fn set_requires_grad(&self, on: bool) {
        self.inner.lock().unwrap().requires_grad = on;
    }

    pub fn grad(&self) -> Option<Vec<f32>> {
        self.inner.lock().unwrap().grad.clone()
    }

    pub fn zero_grad(&self) {
        self.inner.lock().unwrap().grad = None;
    }

    fn add_grad(&self, g: Vec<f32>) {
        let mut inner = self.inner.lock().unwrap();
        if !inner.requires_grad {
            return;
        }
        if g.len() != inner.data.len() {
            panic!(
                "internal error: grad len mismatch (expected {}, got {})",
                inner.data.len(),
                g.len()
            );
        }
        match &mut inner.grad {
            None => inner.grad = Some(g),
            Some(acc) => {
                for (a, x) in acc.iter_mut().zip(g.iter()) {
                    *a += *x;
                }
            }
        }
    }

    fn parents(&self) -> Vec<Tensor<B>> {
        self.inner.lock().unwrap().parents.clone()
    }

    fn backward_fn(&self) -> Option<BackwardFn> {
        self.inner.lock().unwrap().backward.clone()
    }

    fn id(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    fn topo(&self, seen: &mut HashSet<usize>, out: &mut Vec<Tensor<B>>) {
        let id = self.id();
        if seen.contains(&id) {
            return;
        }
        seen.insert(id);
        for p in self.parents() {
            p.topo(seen, out);
        }
        out.push(self.clone());
    }

    pub fn backward(&self) -> Result<(), TensorError> {
        if self.numel() != 1 {
            return Err(TensorError::Backward(
                "backward: expected scalar tensor".to_string(),
            ));
        }
        if !self.requires_grad() {
            return Err(TensorError::Backward(
                "backward: expected requires_grad=true".to_string(),
            ));
        }

        self.add_grad(vec![1.0]);

        let mut seen = HashSet::new();
        let mut order = Vec::new();
        self.topo(&mut seen, &mut order);

        for t in order.into_iter().rev() {
            let g = match t.grad() {
                None => continue,
                Some(v) => v,
            };
            if let Some(bw) = t.backward_fn() {
                (bw)(&g);
            }
        }
        Ok(())
    }

    pub fn detach(&self) -> Tensor<B> {
        let inner = self.inner.lock().unwrap();
        Tensor {
            inner: Arc::new(Mutex::new(Inner {
                data: inner.data.clone(),
                shape: inner.shape.clone(),
                grad: None,
                requires_grad: false,
                parents: Vec::new(),
                backward: None,
            })),
            _b: PhantomData,
        }
    }

    pub fn mutate_data(&self, f: impl FnOnce(&mut [f32])) {
        let mut inner = self.inner.lock().unwrap();
        f(&mut inner.data);
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor<B>, TensorError> {
        let numel: usize = shape.iter().product();
        if numel != self.numel() {
            return Err(TensorError::Shape(format!(
                "reshape: expected numel {numel}, got {}",
                self.numel()
            )));
        }

        let data = self.to_vec();
        let out = Tensor::from_vec(shape, data)?;

        let req = grad_enabled() && self.requires_grad();
        if req {
            out.set_requires_grad(true);
            let parent = self.clone();
            {
                let mut inner = out.inner.lock().unwrap();
                inner.parents = vec![parent.clone()];
                inner.backward = Some(Arc::new(move |g: &[f32]| {
                    parent.add_grad(g.to_vec());
                }));
            }
        }
        Ok(out)
    }

    pub fn sum(&self) -> Result<Tensor<B>, TensorError> {
        let x = self.to_vec();
        let s: f32 = x.iter().sum();
        let out = Tensor::from_vec(&[1], vec![s])?;

        let req = grad_enabled() && self.requires_grad();
        if req {
            out.set_requires_grad(true);
            let parent = self.clone();
            let n = x.len();
            {
                let mut inner = out.inner.lock().unwrap();
                inner.parents = vec![parent.clone()];
                inner.backward = Some(Arc::new(move |g: &[f32]| {
                    let gx = vec![g[0]; n];
                    parent.add_grad(gx);
                }));
            }
        }
        Ok(out)
    }

    pub fn matmul(&self, rhs: &Tensor<B>) -> Result<Tensor<B>, TensorError> {
        let a_shape = self.shape();
        let b_shape = rhs.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(TensorError::Shape("matmul: expected rank-2 tensors".into()));
        }
        let (m, k1) = (a_shape[0], a_shape[1]);
        let (k2, n) = (b_shape[0], b_shape[1]);
        if k1 != k2 {
            return Err(TensorError::Shape(format!("matmul: k mismatch {k1} vs {k2}")));
        }

        let a = self.to_vec();
        let b = rhs.to_vec();
        let mut out = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for kk in 0..k1 {
                    s += a[i * k1 + kk] * b[kk * n + j];
                }
                out[i * n + j] = s;
            }
        }

        let y = Tensor::from_vec(&[m, n], out)?;

        let req = grad_enabled() && (self.requires_grad() || rhs.requires_grad());
        if req {
            y.set_requires_grad(true);

            let left = self.clone();
            let right = rhs.clone();
            let a_fwd = a;
            let b_fwd = b;

            {
                let mut inner = y.inner.lock().unwrap();
                inner.parents = vec![left.clone(), right.clone()];
                inner.backward = Some(Arc::new(move |gy: &[f32]| {
                    if left.requires_grad() {
                        let mut ga = vec![0.0; m * k1];
                        for i in 0..m {
                            for kk in 0..k1 {
                                let mut s = 0.0;
                                for j in 0..n {
                                    s += gy[i * n + j] * b_fwd[kk * n + j];
                                }
                                ga[i * k1 + kk] = s;
                            }
                        }
                        left.add_grad(ga);
                    }

                    if right.requires_grad() {
                        let mut gb = vec![0.0; k1 * n];
                        for kk in 0..k1 {
                            for j in 0..n {
                                let mut s = 0.0;
                                for i in 0..m {
                                    s += a_fwd[i * k1 + kk] * gy[i * n + j];
                                }
                                gb[kk * n + j] = s;
                            }
                        }
                        right.add_grad(gb);
                    }
                }));
            }
        }

        Ok(y)
    }

    pub fn add(&self, rhs: &Tensor<B>) -> Result<Tensor<B>, TensorError> {
        let a_shape = self.shape();
        let b_shape = rhs.shape();

        let a = self.to_vec();
        let b = rhs.to_vec();

        let (out_shape, out_data, broadcast_row) = if a_shape == b_shape {
            let mut out = vec![0.0; a.len()];
            for i in 0..a.len() {
                out[i] = a[i] + b[i];
            }
            (a_shape.clone(), out, false)
        } else if a_shape.len() == 2 && b_shape.len() == 1 && a_shape[1] == b_shape[0] {
            let (m, n) = (a_shape[0], a_shape[1]);
            let mut out = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    out[i * n + j] = a[i * n + j] + b[j];
                }
            }
            (a_shape.clone(), out, true)
        } else {
            return Err(TensorError::Shape(format!(
                "add: unsupported shapes {a_shape:?} + {b_shape:?}"
            )));
        };

        let y = Tensor::from_vec(&out_shape, out_data)?;

        let req = grad_enabled() && (self.requires_grad() || rhs.requires_grad());
        if req {
            y.set_requires_grad(true);
            let left = self.clone();
            let right = rhs.clone();
            {
                let mut inner = y.inner.lock().unwrap();
                inner.parents = vec![left.clone(), right.clone()];
                inner.backward = Some(Arc::new(move |gy: &[f32]| {
                    if left.requires_grad() {
                        left.add_grad(gy.to_vec());
                    }
                    if right.requires_grad() {
                        if !broadcast_row {
                            right.add_grad(gy.to_vec());
                        } else {
                            let a_shape = left.shape();
                            let (m, n) = (a_shape[0], a_shape[1]);
                            let mut gb = vec![0.0; n];
                            for i in 0..m {
                                for j in 0..n {
                                    gb[j] += gy[i * n + j];
                                }
                            }
                            right.add_grad(gb);
                        }
                    }
                }));
            }
        }

        Ok(y)
    }

    pub fn relu(&self) -> Result<Tensor<B>, TensorError> {
        let x = self.to_vec();
        let mut out = vec![0.0; x.len()];
        let mut mask = vec![0.0; x.len()];
        for i in 0..x.len() {
            if x[i] > 0.0 {
                out[i] = x[i];
                mask[i] = 1.0;
            }
        }

        let y = Tensor::from_vec(&self.shape(), out)?;

        let req = grad_enabled() && self.requires_grad();
        if req {
            y.set_requires_grad(true);
            let parent = self.clone();
            {
                let mut inner = y.inner.lock().unwrap();
                inner.parents = vec![parent.clone()];
                inner.backward = Some(Arc::new(move |gy: &[f32]| {
                    let mut gx = vec![0.0; gy.len()];
                    for i in 0..gy.len() {
                        gx[i] = gy[i] * mask[i];
                    }
                    parent.add_grad(gx);
                }));
            }
        }

        Ok(y)
    }

    pub fn cross_entropy_logits(&self, labels: &[usize]) -> Result<Tensor<B>, TensorError> {
        let s = self.shape();
        if s.len() != 2 {
            return Err(TensorError::Shape(
                "cross_entropy_logits: expected rank-2".into(),
            ));
        }
        let (bs, c) = (s[0], s[1]);
        if labels.len() != bs {
            return Err(TensorError::Shape("labels len mismatch".into()));
        }

        let logits = self.to_vec();
        let mut probs = vec![0.0; bs * c];
        let mut loss = 0.0;

        for i in 0..bs {
            let row = &logits[i * c..(i + 1) * c];
            let mut maxv = row[0];
            for &v in row.iter() {
                if v > maxv {
                    maxv = v;
                }
            }
            let mut sumexp = 0.0;
            for j in 0..c {
                sumexp += (row[j] - maxv).exp();
            }
            let logsumexp = sumexp.ln() + maxv;

            for j in 0..c {
                probs[i * c + j] = (row[j] - logsumexp).exp();
            }

            let y = labels[i];
            let p = probs[i * c + y].max(1e-12);
            loss += -p.ln();
        }

        loss /= bs as f32;
        let out = Tensor::from_vec(&[1], vec![loss])?;

        let req = grad_enabled() && self.requires_grad();
        if req {
            out.set_requires_grad(true);
            let parent = self.clone();
            let probs_fwd = probs;
            let labels_fwd = labels.to_vec();
            {
                let mut inner = out.inner.lock().unwrap();
                inner.parents = vec![parent.clone()];
                inner.backward = Some(Arc::new(move |gout: &[f32]| {
                    let g = gout[0] / (bs as f32);
                    let mut gl = probs_fwd.clone();
                    for i in 0..bs {
                        let y = labels_fwd[i];
                        gl[i * c + y] -= 1.0;
                    }
                    for v in gl.iter_mut() {
                        *v *= g;
                    }
                    parent.add_grad(gl);
                }));
            }
        }

        Ok(out)
    }
}

