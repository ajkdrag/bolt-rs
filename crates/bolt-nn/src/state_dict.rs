use crate::{Error, Result};
use crate::store::{Entry, Kind, Store};
use bolt_core::Backend;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorBlob {
    pub kind: Kind,
    pub group: u32,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateDict {
    pub format_version: u32,
    pub tensors: BTreeMap<String, TensorBlob>,
    pub meta: BTreeMap<String, String>,
}

#[derive(Clone)]
pub struct LoadOptions {
    pub strict: bool,
    pub rename: Option<Arc<dyn Fn(&str) -> String + Send + Sync>>,
}

#[derive(Clone, Debug, Default)]
pub struct LoadReport {
    pub missing: Vec<String>,
    pub unexpected: Vec<String>,
    pub mismatched: Vec<(String, Vec<usize>, Vec<usize>)>,
    pub kind_mismatched: Vec<(String, Kind, Kind)>,
}

impl StateDict {
    pub fn new() -> Self {
        Self {
            format_version: 1,
            tensors: BTreeMap::new(),
            meta: BTreeMap::new(),
        }
    }
}

impl<B: Backend> Store<B> {
    pub fn state_dict(&self) -> Result<StateDict> {
        let (ps, bs) = self.all_entries();
        let mut sd = StateDict::new();

        for e in ps.into_iter().chain(bs.into_iter()) {
            let blob = entry_to_blob(&e)?;
            sd.tensors.insert(e.key.clone(), blob);
        }

        Ok(sd)
    }

    pub fn load_state_dict(&self, sd: &StateDict, opt: LoadOptions) -> Result<LoadReport> {
        let mut used = BTreeSet::new();
        let mut report = LoadReport::default();

        for (k_in, blob) in sd.tensors.iter() {
            let k = match &opt.rename {
                None => k_in.clone(),
                Some(f) => (f)(k_in),
            };

            if used.contains(&k) {
                return Err(Error::State(format!(
                    "load_state_dict: rename produced duplicate key: {k}"
                )));
            }

            match self.get_entry(&k) {
                None => report.unexpected.push(k_in.clone()),
                Some(e) => {
                    used.insert(k.clone());

                    if e.kind != blob.kind {
                        report.kind_mismatched.push((k.clone(), e.kind, blob.kind));
                        continue;
                    }

                    if e.shape != blob.shape {
                        report
                            .mismatched
                            .push((k.clone(), e.shape.clone(), blob.shape.clone()));
                        continue;
                    }

                    let t = e.tensor.clone();
                    let data = blob.data.clone();
                    t.mutate_data(|buf| buf.copy_from_slice(&data));
                }
            }
        }

        for k in self.expected_keys() {
            if !used.contains(&k) {
                report.missing.push(k);
            }
        }

        if opt.strict
            && (!report.missing.is_empty()
                || !report.unexpected.is_empty()
                || !report.mismatched.is_empty()
                || !report.kind_mismatched.is_empty())
        {
            return Err(Error::State(format!("strict load failed: {report:?}")));
        }

        Ok(report)
    }
}

fn entry_to_blob<B: Backend>(e: &Entry<B>) -> Result<TensorBlob> {
    Ok(TensorBlob {
        kind: e.kind,
        group: e.group,
        shape: e.shape.clone(),
        data: e.tensor.to_vec(),
    })
}

