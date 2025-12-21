use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

pub fn grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

pub struct NoGradGuard {
    prev: bool,
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|g| g.set(self.prev));
    }
}

pub fn no_grad() -> NoGradGuard {
    let prev = GRAD_ENABLED.with(|g| {
        let prev = g.get();
        g.set(false);
        prev
    });
    NoGradGuard { prev }
}

