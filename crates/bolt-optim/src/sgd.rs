use bolt_core::Backend;
use bolt_nn::Param;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
pub struct SgdCfg {
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SgdGroupCfg {
    pub lr_mult: f32,
    pub weight_decay: Option<f32>,
}

pub struct Sgd<B: Backend> {
    base: SgdCfg,
    groups: BTreeMap<u32, SgdGroupCfg>,
    vel: BTreeMap<String, Vec<f32>>,
    _b: std::marker::PhantomData<B>,
}

impl<B: Backend> Sgd<B> {
    pub fn new(base: SgdCfg) -> Self {
        Self {
            base,
            groups: BTreeMap::new(),
            vel: BTreeMap::new(),
            _b: std::marker::PhantomData,
        }
    }

    pub fn set_group(&mut self, group_id: u32, cfg: SgdGroupCfg) {
        self.groups.insert(group_id, cfg);
    }

    pub fn step(&mut self, params: &[Param<B>]) {
        for p in params {
            let g = match p.grad() {
                None => continue,
                Some(v) => v,
            };

            let (lr, wd) = self.cfg_for(p.group());
            let key = p.key().to_string();
            let mom = self.base.momentum;

            let vbuf = self
                .vel
                .entry(key)
                .or_insert_with(|| vec![0.0; g.len()]);

            let t = p.tensor();
            t.mutate_data(|w| {
                for i in 0..w.len() {
                    let grad = g[i] + wd * w[i];
                    vbuf[i] = mom * vbuf[i] + grad;
                    w[i] -= lr * vbuf[i];
                }
            });
        }
    }

    fn cfg_for(&self, group_id: u32) -> (f32, f32) {
        let g = self.groups.get(&group_id).copied().unwrap_or(SgdGroupCfg {
            lr_mult: 1.0,
            weight_decay: None,
        });
        let lr = self.base.lr * g.lr_mult;
        let wd = g.weight_decay.unwrap_or(self.base.weight_decay);
        (lr, wd)
    }
}

