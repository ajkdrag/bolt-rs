pub fn shuffle_in_place(idxs: &mut [usize], mut seed: u64) {
    for i in (1..idxs.len()).rev() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (seed as usize) % (i + 1);
        idxs.swap(i, j);
    }
}

