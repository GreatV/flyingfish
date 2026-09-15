use anyhow::{Context, Result, bail};
use std::{
    collections::HashMap,
    sync::{Condvar, Mutex, MutexGuard},
    thread::ScopedJoinHandle,
    time::{Duration, Instant},
};

pub(super) struct FillQueue<T> {
    state: Mutex<State<T>>,
    free_cv: Condvar,
    ready_cv: Condvar,
}

struct State<T> {
    free: Vec<usize>,
    ready: HashMap<usize, T>,
    next: usize,
    total: usize,
    cancelled: bool,
}

pub(super) struct CancelOnDrop<'a, T> {
    queue: &'a FillQueue<T>,
    armed: bool,
}

impl<T> Drop for CancelOnDrop<'_, T> {
    fn drop(&mut self) {
        if self.armed {
            self.queue.cancel();
        }
    }
}

impl<T> FillQueue<T> {
    pub(super) fn new(buffers: usize, total: usize) -> Self {
        Self {
            state: Mutex::new(State {
                free: (0..buffers).collect(),
                ready: HashMap::new(),
                next: 0,
                total,
                cancelled: false,
            }),
            free_cv: Condvar::new(),
            ready_cv: Condvar::new(),
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, State<T>>> {
        self.state
            .lock()
            .map_err(|_| anyhow::anyhow!("GLM fill-ahead queue lock poisoned"))
    }

    pub(super) fn guard(&self) -> CancelOnDrop<'_, T> {
        CancelOnDrop {
            queue: self,
            armed: true,
        }
    }

    fn cancel(&self) {
        // Use the waiters' mutex even during panic cleanup to avoid lost wakeups.
        let mut state = self.state.lock().unwrap_or_else(|err| err.into_inner());
        state.cancelled = true;
        self.free_cv.notify_all();
        self.ready_cv.notify_all();
    }

    pub(super) fn run_worker(&self, work: impl FnOnce() -> Result<()>) -> Result<()> {
        let mut guard = self.guard();
        let result = work();
        guard.armed = result.is_err();
        result
    }

    pub(super) fn claim(&self) -> Result<Option<(usize, usize)>> {
        let mut state = self.lock()?;
        loop {
            if state.cancelled || state.next == state.total {
                return Ok(None);
            }
            // Reserve the buffer and ordinal together so later fills cannot starve the consumer.
            if let Some(buffer) = state.free.pop() {
                let ordinal = state.next;
                state.next += 1;
                return Ok(Some((ordinal, buffer)));
            }
            state = self
                .free_cv
                .wait(state)
                .map_err(|_| anyhow::anyhow!("GLM fill-ahead queue lock poisoned"))?;
        }
    }

    pub(super) fn publish(&self, ordinal: usize, value: T) -> Result<()> {
        self.lock()?.ready.insert(ordinal, value);
        self.ready_cv.notify_all();
        Ok(())
    }

    pub(super) fn release(&self, buffer: usize) -> Result<()> {
        self.lock()?.free.push(buffer);
        self.free_cv.notify_one();
        Ok(())
    }

    pub(super) fn take(&self, ordinal: usize) -> Result<T> {
        let start = Instant::now();
        let timeout = Duration::from_secs(60);
        let mut state = self.lock()?;
        loop {
            if let Some(value) = state.ready.remove(&ordinal) {
                return Ok(value);
            }
            if state.cancelled {
                bail!("GLM fill-ahead aborted before miss {ordinal}");
            }
            let remaining = timeout.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                bail!("GLM fill-ahead timed out 60s waiting for miss {ordinal}");
            }
            (state, _) = self
                .ready_cv
                .wait_timeout(state, remaining)
                .map_err(|_| anyhow::anyhow!("GLM fill-ahead queue lock poisoned"))?;
        }
    }

    pub(super) fn finish<'scope>(
        &self,
        walk: Result<()>,
        workers: impl IntoIterator<Item = ScopedJoinHandle<'scope, Result<()>>>,
    ) -> Result<()> {
        self.cancel();
        let mut failure = None;
        for worker in workers {
            let result = worker.join().unwrap_or_else(|panic| {
                let message = panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown panic");
                Err(anyhow::anyhow!("GLM fill-ahead worker panicked: {message}"))
            });
            if let Err(error) = result {
                failure.get_or_insert(error);
            }
        }
        if let Some(error) = failure {
            return Err(error).context("GLM fill-ahead worker failed");
        }
        walk
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    };

    #[test]
    fn cancellation_before_wait_does_not_block() -> Result<()> {
        let queue = FillQueue::<()>::new(0, 1);
        queue.cancel();
        assert_eq!(queue.claim()?, None);
        assert!(queue.take(0).unwrap_err().to_string().contains("aborted"));
        Ok(())
    }

    #[test]
    fn cancellation_synchronizes_with_wait_predicate() {
        let queue = FillQueue::<()>::new(0, 1);
        let (checked_tx, checked_rx) = mpsc::channel();
        let (cancel_tx, cancel_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            let queue = &queue;
            let worker = scope.spawn(move || {
                let state = queue.state.lock().unwrap();
                assert!(!state.cancelled);
                checked_tx.send(()).unwrap();
                cancel_rx.recv().unwrap();
                let (state, timeout) = queue
                    .free_cv
                    .wait_timeout_while(state, Duration::from_secs(2), |s| !s.cancelled)
                    .unwrap();
                assert!(state.cancelled);
                assert!(!timeout.timed_out());
            });
            checked_rx.recv().unwrap();
            cancel_tx.send(()).unwrap();
            queue.cancel();
            worker.join().unwrap();
        });
    }

    #[test]
    fn ordered_consumer_recycles_out_of_order_fills() -> Result<()> {
        let queue = FillQueue::new(2, 3);
        let (first, first_buffer) = queue.claim()?.unwrap();
        let (second, second_buffer) = queue.claim()?.unwrap();
        queue.publish(second, second_buffer)?;
        queue.publish(first, first_buffer)?;
        queue.release(queue.take(first)?)?;
        assert_eq!(queue.claim()?, Some((2, first_buffer)));
        queue.release(queue.take(second)?)?;
        assert_eq!(queue.claim()?, None);
        Ok(())
    }

    #[test]
    fn worker_failure_preserves_cause_and_joins_every_worker() {
        let queue = FillQueue::<()>::new(0, 1);
        let exited = AtomicUsize::new(0);
        let error = std::thread::scope(|scope| {
            let workers = vec![
                scope.spawn(|| queue.run_worker(|| bail!("event synchronization failed"))),
                scope.spawn(|| {
                    queue.run_worker(|| {
                        assert_eq!(queue.claim()?, None);
                        exited.fetch_add(1, Ordering::Relaxed);
                        Ok(())
                    })
                }),
            ];
            queue.finish(queue.take(0), workers)
        })
        .unwrap_err();
        assert!(format!("{error:#}").contains("event synchronization failed"));
        assert_eq!(exited.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn uploader_failure_cancels_workers_and_preserves_error() {
        let queue = FillQueue::<()>::new(0, 1);
        let error = std::thread::scope(|scope| {
            let worker = scope.spawn(|| {
                queue.run_worker(|| {
                    assert_eq!(queue.claim()?, None);
                    Ok(())
                })
            });
            queue.finish(Err(anyhow::anyhow!("upload failed")), [worker])
        })
        .unwrap_err();
        assert_eq!(error.to_string(), "upload failed");
    }

    #[test]
    fn worker_panics_cancel_and_are_all_joined() {
        let queue = FillQueue::<()>::new(0, 1);
        let error = std::thread::scope(|scope| {
            let workers = (0..2)
                .map(|_| scope.spawn(|| queue.run_worker(|| panic!("fill panic"))))
                .collect::<Vec<_>>();
            queue.finish(queue.take(0), workers)
        })
        .unwrap_err();
        assert!(format!("{error:#}").contains("fill panic"));
    }

    #[test]
    fn cancellation_guard_recovers_poisoned_queue() {
        let queue = FillQueue::<()>::new(0, 1);
        let _ = std::panic::catch_unwind(|| {
            let _guard = queue.guard();
            let _state = queue.state.lock().unwrap();
            panic!("poison queue");
        });
        assert!(queue.state.lock().err().unwrap().into_inner().cancelled);
    }
}
