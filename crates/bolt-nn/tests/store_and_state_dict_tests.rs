use bolt_cpu::Cpu;
use bolt_nn::{Init, LoadOptions, StateDict, Store};

fn assert_vec_eq(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert!(a[i].to_bits() == b[i].to_bits(), "mismatch at {i}");
    }
}

#[test]
fn seal_prevents_param_creation() {
    let store = Store::<Cpu>::new(0);
    store.seal();
    let r = store.param("p", &[2, 2], Init::Zeros);
    assert!(r.is_err());
}

#[test]
fn state_dict_roundtrip_loads_tensor_data() {
    let s1 = Store::<Cpu>::new(0);
    let p1 = s1.param("p", &[2], Init::Zeros).unwrap();
    let b1 = s1.buffer("b", &[2], Init::Zeros).unwrap();

    p1.tensor().mutate_data(|w| {
        w.copy_from_slice(&[1.0, 2.0]);
    });
    b1.tensor().mutate_data(|w| {
        w.copy_from_slice(&[3.0, 4.0]);
    });

    let sd = s1.state_dict().unwrap();

    let s2 = Store::<Cpu>::new(123);
    let p2 = s2.param("p", &[2], Init::Zeros).unwrap();
    let b2 = s2.buffer("b", &[2], Init::Zeros).unwrap();

    let report = s2
        .load_state_dict(
            &sd,
            LoadOptions {
                strict: true,
                rename: None,
            },
        )
        .unwrap();
    assert!(report.missing.is_empty());
    assert!(report.unexpected.is_empty());
    assert!(report.mismatched.is_empty());
    assert!(report.kind_mismatched.is_empty());

    assert_vec_eq(&p2.tensor().to_vec(), &[1.0, 2.0]);
    assert_vec_eq(&b2.tensor().to_vec(), &[3.0, 4.0]);
}

#[test]
fn load_state_dict_supports_rename_mapping() {
    let s_src = Store::<Cpu>::new(0);
    let p_src = s_src.param("old", &[2], Init::Zeros).unwrap();
    p_src.tensor().mutate_data(|w| {
        w.copy_from_slice(&[9.0, 10.0]);
    });
    let sd = s_src.state_dict().unwrap();

    let s_dst = Store::<Cpu>::new(1);
    let p_dst = s_dst.param("new", &[2], Init::Zeros).unwrap();

    let report = s_dst
        .load_state_dict(
            &sd,
            LoadOptions {
                strict: true,
                rename: Some(std::sync::Arc::new(|k: &str| {
                    if k == "old" {
                        "new".to_string()
                    } else {
                        k.to_string()
                    }
                })),
            },
        )
        .unwrap();

    assert!(report.missing.is_empty());
    assert!(report.unexpected.is_empty());
    assert!(report.mismatched.is_empty());
    assert!(report.kind_mismatched.is_empty());

    assert_vec_eq(&p_dst.tensor().to_vec(), &[9.0, 10.0]);
}

#[test]
fn load_state_dict_fails_on_rename_collision() {
    let s_src = Store::<Cpu>::new(0);
    let _a = s_src.param("a", &[1], Init::Zeros).unwrap();
    let _b = s_src.param("b", &[1], Init::Zeros).unwrap();
    let sd = s_src.state_dict().unwrap();

    let s_dst = Store::<Cpu>::new(0);
    let _x = s_dst.param("x", &[1], Init::Zeros).unwrap();

    let err = s_dst
        .load_state_dict(
            &sd,
            LoadOptions {
                strict: false,
                rename: Some(std::sync::Arc::new(|_k: &str| "x".to_string())),
            },
        )
        .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("duplicate key"), "unexpected error: {msg}");
}

#[test]
fn load_state_dict_reports_mismatch_without_strict() {
    let s_src = Store::<Cpu>::new(0);
    let _p = s_src.param("p", &[2], Init::Zeros).unwrap();
    let sd = s_src.state_dict().unwrap();

    let s_dst = Store::<Cpu>::new(0);
    let _p2 = s_dst.param("p", &[3], Init::Zeros).unwrap();

    let report = s_dst
        .load_state_dict(
            &sd,
            LoadOptions {
                strict: false,
                rename: None,
            },
        )
        .unwrap();
    assert_eq!(report.mismatched.len(), 1);
}

#[test]
fn state_dict_is_serde_roundtripable() {
    let s = Store::<Cpu>::new(0);
    let _p = s.param("p", &[2], Init::Zeros).unwrap();
    let mut sd = s.state_dict().unwrap();
    sd.meta.insert("k".into(), "v".into());

    let bytes = bincode::serialize(&sd).unwrap();
    let sd2: StateDict = bincode::deserialize(&bytes).unwrap();
    assert_eq!(sd2.meta.get("k").unwrap(), "v");
}

