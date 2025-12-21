use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("format error: {0}")]
    Format(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub struct MnistData {
    pub train_images: Vec<f32>,
    pub train_labels: Vec<u8>,
    pub test_images: Vec<f32>,
    pub test_labels: Vec<u8>,
}

impl MnistData {
    pub fn load(base_dir: impl AsRef<Path>) -> Result<Self> {
        let base_dir = base_dir.as_ref();

        let train_images_path = base_dir.join("train-images-idx3-ubyte");
        let train_labels_path = base_dir.join("train-labels-idx1-ubyte");
        let test_images_path = base_dir.join("t10k-images-idx3-ubyte");
        let test_labels_path = base_dir.join("t10k-labels-idx1-ubyte");

        let train_images = load_images(&train_images_path)?;
        let train_labels = load_labels(&train_labels_path)?;
        let test_images = load_images(&test_images_path)?;
        let test_labels = load_labels(&test_labels_path)?;

        if train_images.len() != train_labels.len() * 784 {
            return Err(Error::Format(
                "train images/labels length mismatch".into(),
            ));
        }
        if test_images.len() != test_labels.len() * 784 {
            return Err(Error::Format("test images/labels length mismatch".into()));
        }

        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }

    pub fn train_len(&self) -> usize {
        self.train_labels.len()
    }

    pub fn test_len(&self) -> usize {
        self.test_labels.len()
    }

    pub fn get_train_batch(&self, idxs: &[usize]) -> (Vec<f32>, Vec<usize>) {
        get_batch(&self.train_images, &self.train_labels, idxs)
    }

    pub fn get_test_batch(&self, idxs: &[usize]) -> (Vec<f32>, Vec<usize>) {
        get_batch(&self.test_images, &self.test_labels, idxs)
    }
}

fn get_batch(images: &[f32], labels: &[u8], idxs: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let bs = idxs.len();
    let mut x = vec![0.0; bs * 784];
    let mut y = vec![0usize; bs];

    for (bi, &idx) in idxs.iter().enumerate() {
        let off = idx * 784;
        x[bi * 784..(bi + 1) * 784].copy_from_slice(&images[off..off + 784]);
        y[bi] = labels[idx] as usize;
    }

    (x, y)
}

fn load_images(path: &Path) -> Result<Vec<f32>> {
    let buf = fs::read(path)?;
    let mut cur = Cursor::new(path.to_path_buf(), buf);
    let magic = cur.read_u32_be()?;
    if magic != 2051 {
        return Err(Error::Format(format!(
            "bad images magic {magic} in {}",
            path.display()
        )));
    }

    let n = cur.read_u32_be()? as usize;
    let rows = cur.read_u32_be()? as usize;
    let cols = cur.read_u32_be()? as usize;
    if rows != 28 || cols != 28 {
        return Err(Error::Format(format!(
            "expected 28x28 images, got {rows}x{cols}"
        )));
    }

    let expected = 16usize + n * rows * cols;
    if cur.buf.len() != expected {
        return Err(Error::Format(format!(
            "unexpected images file size: expected {expected}, got {}",
            cur.buf.len()
        )));
    }

    let mut out = vec![0.0; n * 784];
    for i in 0..(n * 784) {
        out[i] = (cur.buf[16 + i] as f32) / 255.0;
    }
    Ok(out)
}

fn load_labels(path: &Path) -> Result<Vec<u8>> {
    let buf = fs::read(path)?;
    let mut cur = Cursor::new(path.to_path_buf(), buf);
    let magic = cur.read_u32_be()?;
    if magic != 2049 {
        return Err(Error::Format(format!(
            "bad labels magic {magic} in {}",
            path.display()
        )));
    }
    let n = cur.read_u32_be()? as usize;

    let expected = 8usize + n;
    if cur.buf.len() != expected {
        return Err(Error::Format(format!(
            "unexpected labels file size: expected {expected}, got {}",
            cur.buf.len()
        )));
    }

    Ok(cur.buf[8..].to_vec())
}

struct Cursor {
    path: PathBuf,
    buf: Vec<u8>,
    off: usize,
}

impl Cursor {
    fn new(path: PathBuf, buf: Vec<u8>) -> Self {
        Self { path, buf, off: 0 }
    }

    fn read_u32_be(&mut self) -> Result<u32> {
        if self.off + 4 > self.buf.len() {
            return Err(Error::Format(format!(
                "unexpected eof reading u32 in {}",
                self.path.display()
            )));
        }
        let b = &self.buf[self.off..self.off + 4];
        self.off += 4;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }
}

