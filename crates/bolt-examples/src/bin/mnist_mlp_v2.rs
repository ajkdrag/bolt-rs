use bolt_autodiff::{no_grad, Tensor};
use bolt_cpu::Cpu;
use bolt_data::shuffle_in_place;
use bolt_datasets::mnist::MnistData;
use bolt_nn::{layers::linear::Linear, Module, Store};
use bolt_optim::{Sgd, SgdCfg, SgdGroupCfg};

struct MnistMLP {
    fc1: Linear<Cpu>,
    fc2: Linear<Cpu>,
}

impl MnistMLP {
    fn init(store: &Store<Cpu>, hidden: usize) -> bolt_nn::Result<Self> {
        let fc1 = Linear::init(&store.sub("fc1"), 784, hidden, true)?;
        let fc2 = Linear::init(&store.sub("fc2"), hidden, 10, true)?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module<Cpu> for MnistMLP {
    type Input = Tensor<Cpu>;
    type Output = Tensor<Cpu>;

    fn forward(&self, x: Self::Input, train: bool) -> bolt_nn::Result<Self::Output> {
        let x = self.fc1.forward(x, train)?;
        let x = x
            .relu()
            .map_err(|e| bolt_nn::Error::Shape(e.to_string()))?;
        self.fc2.forward(x, train)
    }
}

fn argmax_rowwise(logits: &[f32], bs: usize, c: usize) -> Vec<usize> {
    let mut out = vec![0usize; bs];
    for i in 0..bs {
        let row = &logits[i * c..(i + 1) * c];
        let mut best = 0usize;
        let mut bestv = row[0];
        for j in 1..c {
            if row[j] > bestv {
                bestv = row[j];
                best = j;
            }
        }
        out[i] = best;
    }
    out
}

fn main() -> bolt_nn::Result<()> {
    let data = MnistData::load("data/mnist").map_err(|e| bolt_nn::Error::Io(e.to_string()))?;

    let store = Store::<Cpu>::new(1337);
    let model = MnistMLP::init(&store, 128)?;
    store.seal();

    let mut opt = Sgd::<Cpu>::new(SgdCfg {
        lr: 0.1,
        momentum: 0.9,
        weight_decay: 1e-4,
    });
    opt.set_group(
        1,
        SgdGroupCfg {
            lr_mult: 1.0,
            weight_decay: Some(0.0),
        },
    );

    let params = store.trainable();
    let batch = 128usize;
    let epochs = 3usize;

    let mut idxs: Vec<usize> = (0..data.train_len()).collect();
    for ep in 0..epochs {
        shuffle_in_place(&mut idxs, (ep as u64 + 1) * 0x9E3779B97F4A7C15);

        let mut running_loss = 0.0;
        let mut steps = 0usize;

        for chunk in idxs.chunks(batch) {
            let (xv, yv) = data.get_train_batch(chunk);
            let x = Tensor::<Cpu>::from_vec(&[chunk.len(), 784], xv)
                .map_err(|e| bolt_nn::Error::State(e.to_string()))?;

            store.zero_grad();

            let logits = model.forward(x, true)?;
            let loss = logits
                .cross_entropy_logits(&yv)
                .map_err(|e| bolt_nn::Error::Shape(e.to_string()))?;

            running_loss += loss.to_vec()[0];
            steps += 1;

            loss.backward()
                .map_err(|e| bolt_nn::Error::State(e.to_string()))?;

            opt.step(&params);
        }

        let _ng = no_grad();
        let test_idxs: Vec<usize> = (0..data.test_len()).collect();

        let mut correct = 0usize;
        let mut total = 0usize;
        for chunk in test_idxs.chunks(batch) {
            let (xv, yv) = data.get_test_batch(chunk);
            let x = Tensor::<Cpu>::from_vec(&[chunk.len(), 784], xv)
                .map_err(|e| bolt_nn::Error::State(e.to_string()))?;
            let logits = model.forward(x, false)?;
            let lv = logits.to_vec();
            let pred = argmax_rowwise(&lv, chunk.len(), 10);
            for i in 0..chunk.len() {
                if pred[i] == yv[i] {
                    correct += 1;
                }
            }
            total += chunk.len();
        }

        println!(
            "epoch {ep} loss {:.4} acc {:.2}%",
            running_loss / (steps as f32),
            100.0 * (correct as f32) / (total as f32)
        );
    }

    let mut sd = store.state_dict()?;
    sd.meta.insert("arch".into(), "MnistMLP(hidden=128)".into());
    sd.meta.insert("format".into(), "bolt_nn_v2_mvp".into());

    Ok(())
}
