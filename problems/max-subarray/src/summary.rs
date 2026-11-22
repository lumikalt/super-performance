//! What is known about a stretch of the input, enough to join it with its neighbours.

/// What is known about a stretch of the input, enough to join it with its neighbours.
#[derive(Clone, Copy)]
pub(crate) struct Summary {
    /// The sum of the whole stretch.
    pub(crate) sum: i64,
    /// The best sum of a run that starts at the beginning of the stretch.
    pub(crate) prefix: i64,
    /// The best sum of a run that ends at the end of the stretch.
    pub(crate) suffix: i64,
    /// The best sum of any run inside the stretch.
    pub(crate) best: i64,
}

impl Summary {
    pub(crate) fn of(x: i32) -> Self {
        let x = x as i64;

        Self { sum: x, prefix: x, suffix: x, best: x }
    }

    /// Takes in the next value, this is the Kadane step.
    #[inline(always)]
    pub(crate) fn push(&mut self, x: i32) {
        let x = x as i64;

        self.sum += x;
        self.prefix = self.prefix.max(self.sum);
        self.suffix = x.max(self.suffix + x);
        self.best = self.best.max(self.suffix);
    }

    /// The summary of this stretch followed by `next`, a run can now cross the join.
    pub(crate) fn then(self, next: Self) -> Self {
        Self {
            sum: self.sum + next.sum,
            prefix: self.prefix.max(self.sum + next.prefix),
            suffix: next.suffix.max(next.sum + self.suffix),
            best: self.best.max(next.best).max(self.suffix + next.prefix),
        }
    }
}
