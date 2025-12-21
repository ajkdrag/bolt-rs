use crate::{Error, Result};
use std::f32::consts::PI;

#[derive(Clone, Debug)]
pub enum Init {
    Zeros,
    Uniform { low: f32, high: f32 },
    Normal { mean: f32, std: f32 },
    KaimingUniform { a: f32 },
}

#[derive(Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn next_f32(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as u32;
        (u as f32) / ((1u32 << 24) as f32)
    }

    pub fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }

    pub fn normal(&mut self, mean: f32, std: f32) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + std * z0
    }
}

pub fn fill(shape: &[usize], init: &Init, rng: &mut Rng) -> Result<Vec<f32>> {
    let numel: usize = shape.iter().product();
    let mut v = vec![0.0; numel];

    match init {
        Init::Zeros => {}
        Init::Uniform { low, high } => {
            for x in v.iter_mut() {
                *x = rng.uniform(*low, *high);
            }
        }
        Init::Normal { mean, std } => {
            for x in v.iter_mut() {
                *x = rng.normal(*mean, *std);
            }
        }
        Init::KaimingUniform { a } => {
            if shape.len() < 2 {
                return Err(Error::State(
                    "kaiming_uniform expects at least 2D weight".into(),
                ));
            }

            let fan_in = shape[0] as f32;
            let gain = (2.0 / (1.0 + a * a)).sqrt();
            let std = gain / fan_in.sqrt();
            let bound = (3.0f32).sqrt() * std;

            for x in v.iter_mut() {
                *x = rng.uniform(-bound, bound);
            }
        }
    }
    Ok(v)
}

