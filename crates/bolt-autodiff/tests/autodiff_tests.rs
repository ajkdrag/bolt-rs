use bolt_autodiff::{no_grad, Tensor};
use bolt_core::Backend;

#[derive(Clone, Debug, Default)]
struct TestBackend;
impl Backend for TestBackend {}

fn assert_allclose(a: &[f32], b: &[f32], atol: f32, rtol: f32) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        let tol = atol + rtol * b[i].abs();
        assert!(
            diff <= tol,
            "allclose failed at {i}: a={} b={} diff={} tol={}",
            a[i],
            b[i],
            diff,
            tol
        );
    }
}

fn softmax_rowwise(logits: &[f32], bs: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0; bs * c];
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
        for j in 0..c {
            out[i * c + j] = (row[j] - maxv).exp() / sumexp;
        }
    }
    out
}

#[test]
fn cross_entropy_logits_backward_matches_softmax_minus_onehot() {
    let bs = 2usize;
    let c = 3usize;

    let logits_data = vec![1.0, 2.0, 3.0, 0.0, -1.0, 1.0];
    let labels = vec![2usize, 0usize];

    let logits = Tensor::<TestBackend>::from_vec(&[bs, c], logits_data.clone()).unwrap();
    logits.set_requires_grad(true);

    let loss = logits.cross_entropy_logits(&labels).unwrap();
    loss.backward().unwrap();

    let probs = softmax_rowwise(&logits_data, bs, c);
    let mut expected = probs;
    for i in 0..bs {
        expected[i * c + labels[i]] -= 1.0;
    }
    for v in expected.iter_mut() {
        *v /= bs as f32;
    }

    let got = logits.grad().unwrap();
    assert_allclose(&got, &expected, 1e-5, 1e-5);
}

#[test]
fn no_grad_prevents_graph_building() {
    let a = Tensor::<TestBackend>::from_vec(&[2], vec![1.0, 2.0]).unwrap();
    let b = Tensor::<TestBackend>::from_vec(&[2], vec![3.0, 4.0]).unwrap();
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    let y = {
        let _g = no_grad();
        a.add(&b).unwrap()
    };

    assert!(!y.requires_grad());
    let s = y.sum().unwrap();
    assert!(!s.requires_grad());
    assert!(s.backward().is_err());
}

