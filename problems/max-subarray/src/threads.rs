//! The input cut in parts, each part summarised on a thread.

use crate::summary::Summary;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The summary of a stretch with the fastest single-thread code available.
fn fast_summary(input: &[i32]) -> Summary {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        crate::avx2::summary(input)
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut summary = Summary::of(input[0]);

        for &x in &input[1..] {
            summary.push(x);
        }

        summary
    }
}

/// The fewest values worth giving to a thread, below this the cost of starting it is larger than the work.
const MIN_PART: usize = 1 << 16;

/// Cuts the input in one equal part per thread and joins the parts at the end.
pub fn threads_equal(input: &[i32]) -> i64 {
    assert!(!input.is_empty(), "the subarray has to be non-empty");

    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let parts = threads.min(input.len() / MIN_PART).max(1);

    // A single part is not worth a thread.
    if parts == 1 {
        return fast_summary(input).best;
    }

    let len = input.len().div_ceil(parts);

    let summaries: Vec<Summary> = std::thread::scope(|scope| {
        let handles: Vec<_> = input.chunks(len).map(|part| scope.spawn(|| fast_summary(part))).collect();

        handles.into_iter().map(|handle| handle.join().unwrap()).collect()
    });

    summaries.into_iter().reduce(Summary::then).unwrap().best
}

/// The size of the pieces threads take from the queue, a multiple of the SIMD block.
const PIECE: usize = 1 << 18;

/// Cuts the input in many small pieces and lets every thread take the next one when it is free.
/// A thread on a slow core simply takes fewer pieces, so nobody waits for it at the end.
pub fn threads_queue(input: &[i32]) -> i64 {
    assert!(!input.is_empty(), "the subarray has to be non-empty");

    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let pieces: Vec<&[i32]> = input.chunks(PIECE).collect();

    if threads == 1 || pieces.len() < 2 {
        return fast_summary(input).best;
    }

    let next = AtomicUsize::new(0);
    let done: Vec<OnceLock<Summary>> = pieces.iter().map(|_| OnceLock::new()).collect();

    std::thread::scope(|scope| {
        for _ in 0..threads.min(pieces.len()) {
            scope.spawn(|| {
                loop {
                    let piece = next.fetch_add(1, Ordering::Relaxed);

                    if piece >= pieces.len() {
                        break;
                    }

                    // Every piece is taken once, so this never fails.
                    done[piece].set(fast_summary(pieces[piece])).ok();
                }
            });
        }
    });

    done.into_iter().map(|piece| piece.into_inner().unwrap()).reduce(Summary::then).unwrap().best
}
