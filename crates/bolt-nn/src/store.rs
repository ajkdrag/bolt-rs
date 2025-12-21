use crate::{init, Error, Init, Result};
use bolt_autodiff::Tensor;
use bolt_core::Backend;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, RwLock,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Param,
    Buffer,
}

pub struct Entry<B: Backend> {
    pub(crate) key: String,
    pub(crate) kind: Kind,
    pub(crate) group: u32,
    pub(crate) tensor: Tensor<B>,
    pub(crate) shape: Vec<usize>,
}

#[derive(Clone)]
pub struct Param<B: Backend>(Arc<Entry<B>>);

#[derive(Clone)]
pub struct Buffer<B: Backend>(Arc<Entry<B>>);

impl<B: Backend> Param<B> {
    pub fn key(&self) -> &str {
        &self.0.key
    }

    pub fn group(&self) -> u32 {
        self.0.group
    }

    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    pub fn tensor(&self) -> Tensor<B> {
        self.0.tensor.clone()
    }

    pub fn grad(&self) -> Option<Vec<f32>> {
        self.0.tensor.grad()
    }

    pub fn zero_grad(&self) {
        self.0.tensor.zero_grad();
    }
}

impl<B: Backend> Buffer<B> {
    pub fn key(&self) -> &str {
        &self.0.key
    }

    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    pub fn tensor(&self) -> Tensor<B> {
        self.0.tensor.clone()
    }

    pub fn set(&self, src: &Tensor<B>) -> Result<()> {
        let expected = self.shape();
        if src.shape() != expected {
            return Err(Error::State(format!(
                "buffer set: expected shape {expected:?}, got {:?}",
                src.shape()
            )));
        }
        let data = src.to_vec();
        self.0.tensor.mutate_data(|dst| dst.copy_from_slice(&data));
        Ok(())
    }
}

struct Inner<B: Backend> {
    params: RwLock<BTreeMap<String, Arc<Entry<B>>>>,
    buffers: RwLock<BTreeMap<String, Arc<Entry<B>>>>,
    sealed: AtomicBool,
    rng: Mutex<init::Rng>,
}

#[derive(Clone)]
pub struct Store<B: Backend> {
    inner: Arc<Inner<B>>,
    prefix: String,
    group: u32,
}

impl<B: Backend> Store<B> {
    pub fn new(seed: u64) -> Self {
        Self {
            inner: Arc::new(Inner {
                params: RwLock::new(BTreeMap::new()),
                buffers: RwLock::new(BTreeMap::new()),
                sealed: AtomicBool::new(false),
                rng: Mutex::new(init::Rng::new(seed)),
            }),
            prefix: String::new(),
            group: 0,
        }
    }

    pub fn sub(&self, name: &str) -> Self {
        validate_segment_or_panic(name);
        let mut prefix = self.prefix.clone();
        if !prefix.is_empty() {
            prefix.push('.');
        }
        prefix.push_str(name);
        Self {
            inner: self.inner.clone(),
            prefix,
            group: self.group,
        }
    }

    pub fn sub_idx(&self, idx: usize) -> Self {
        self.sub(&idx.to_string())
    }

    pub fn group(&self, group: u32) -> Self {
        Self {
            inner: self.inner.clone(),
            prefix: self.prefix.clone(),
            group,
        }
    }

    pub fn seal(&self) {
        self.inner.sealed.store(true, Ordering::Relaxed);
    }

    pub fn param(&self, name: &str, shape: &[usize], init: Init) -> Result<Param<B>> {
        validate_leaf_name(name)?;
        self.create(name, shape, init, Kind::Param).map(Param)
    }

    pub fn buffer(&self, name: &str, shape: &[usize], init: Init) -> Result<Buffer<B>> {
        validate_leaf_name(name)?;
        self.create(name, shape, init, Kind::Buffer).map(Buffer)
    }

    pub fn trainable(&self) -> Vec<Param<B>> {
        let map = self.inner.params.read().unwrap();
        map.values().cloned().map(Param).collect()
    }

    pub fn named_trainable(&self) -> Vec<(String, Param<B>)> {
        let map = self.inner.params.read().unwrap();
        map.iter()
            .map(|(k, v)| (k.clone(), Param(v.clone())))
            .collect()
    }

    pub fn zero_grad(&self) {
        for p in self.trainable() {
            p.zero_grad();
        }
    }

    fn create(&self, name: &str, shape: &[usize], initv: Init, kind: Kind) -> Result<Arc<Entry<B>>> {
        if self.inner.sealed.load(Ordering::Relaxed) {
            return Err(Error::State(
                "store is sealed; cannot create new parameters".into(),
            ));
        }

        if shape.iter().any(|&d| d == 0) {
            return Err(Error::Shape(format!("zero dimension in shape {shape:?}")));
        }

        let key = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        };

        let mut rng = self.inner.rng.lock().unwrap();
        let data = init::fill(shape, &initv, &mut rng)?;

        let t = Tensor::<B>::from_vec(shape, data)
            .map_err(|e| Error::State(format!("tensor init failed: {e}")))?;
        t.set_requires_grad(kind == Kind::Param);

        let entry = Arc::new(Entry {
            key: key.clone(),
            kind,
            group: self.group,
            tensor: t,
            shape: shape.to_vec(),
        });

        match kind {
            Kind::Param => {
                let mut map = self.inner.params.write().unwrap();
                if map.contains_key(&key) {
                    return Err(Error::State(format!("duplicate param key: {key}")));
                }
                map.insert(key, entry.clone());
            }
            Kind::Buffer => {
                let mut map = self.inner.buffers.write().unwrap();
                if map.contains_key(&key) {
                    return Err(Error::State(format!("duplicate buffer key: {key}")));
                }
                map.insert(key, entry.clone());
            }
        }

        Ok(entry)
    }

    pub(crate) fn all_entries(&self) -> (Vec<Arc<Entry<B>>>, Vec<Arc<Entry<B>>>) {
        let ps = self.inner.params.read().unwrap().values().cloned().collect();
        let bs = self.inner.buffers.read().unwrap().values().cloned().collect();
        (ps, bs)
    }

    pub(crate) fn get_entry(&self, key: &str) -> Option<Arc<Entry<B>>> {
        if let Some(e) = self.inner.params.read().unwrap().get(key) {
            return Some(e.clone());
        }
        self.inner.buffers.read().unwrap().get(key).cloned()
    }

    pub(crate) fn expected_keys(&self) -> Vec<String> {
        let mut out: Vec<String> = self.inner.params.read().unwrap().keys().cloned().collect();
        out.extend(self.inner.buffers.read().unwrap().keys().cloned());
        out.sort();
        out
    }
}

fn validate_segment(seg: &str) -> Result<()> {
    if seg.is_empty() {
        return Err(Error::State("empty name segment".into()));
    }
    if seg.contains('.') {
        return Err(Error::State(format!("'.' is reserved in name segment: {seg}")));
    }
    Ok(())
}

fn validate_segment_or_panic(seg: &str) {
    validate_segment(seg).unwrap_or_else(|e| panic!("{e}"));
}

fn validate_leaf_name(name: &str) -> Result<()> {
    validate_segment(name)
}
