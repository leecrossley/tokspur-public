//! The serving loop around the session, mirrored from `tokspur-server/src/serve.rs`: the one-slot job queue (`Serving::enqueue` and the availability refusal in `route`), `generation_loop` and `run_job`, with `std` threads, mutex, condvar and atomics replaced by Shuttle's.
//!
//! Not mirrored: sockets and HTTP framing (a model `Connection` keeps the disconnect check every write makes), signals (`stop_requested`), random response IDs and seeds, and time. The request deadline is one more way a connection becomes cancelled, so it is folded into the disconnect flag.
//!
//! Shuttle's timed condvar waits never time out, but the real loop relies on its 100 ms timeout twice: for the idle liveness tick, and when the shutdown notify (sent without the queue lock) lands between the predicate check and the wait. The model's wait may time out a bounded number of times (chosen nondeterministically) for the first, and the harness repeats the shutdown notify until the generation thread ends for the second.

use std::cell::{Cell, RefCell};
use std::fmt;
use std::sync::Arc;

use shuttle::rand::{Rng, thread_rng};
use shuttle::sync::atomic::{AtomicBool, Ordering};
use shuttle::sync::mpsc::{Sender, channel};
use shuttle::sync::{Condvar, Mutex, MutexGuard};
use shuttle::thread;
use tokspur_json::Json;
use tokspur_server::admission::admit_chat;
use tokspur_server::session::{ServeError, ServeSession, ServeSink};
use tokspur_wire::worker::CallbackFailed;

use crate::log::{Event, Log};
use crate::wire::Wire;

/// A failed write; `Display` is the real transport's text.
#[derive(Debug)]
pub struct Disconnected;

impl fmt::Display for Disconnected {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("HTTP peer disconnected or sent unexpected bytes")
    }
}

impl std::error::Error for Disconnected {}

/// The session sees transport failures as `ServeError::Http`; the model has no `HttpError` constructor, and every sink error is handled alike.
impl From<Disconnected> for ServeError {
    fn from(error: Disconnected) -> Self {
        ServeError::Text(Box::new(error))
    }
}

/// One accepted connection: the disconnect flag every write checks (`Connection::write_all`), and the response the client waits for.
pub struct Connection {
    disconnected: Arc<AtomicBool>,
    response: Sender<u16>,
}

impl Connection {
    /// `Connection::disconnected`.
    pub fn disconnected(&mut self) -> bool {
        self.disconnected.load(Ordering::SeqCst)
    }

    /// One `write_all`: refused once the peer is gone.
    fn write(&mut self) -> Result<(), Disconnected> {
        if self.disconnected() {
            Err(Disconnected)
        } else {
            Ok(())
        }
    }

    fn respond(&self, status: u16) {
        if self.response.send(status).is_err() {
            // The client stopped waiting.
        }
    }

    /// Header and body writes.
    pub fn send_json(&mut self, status: u16) -> Result<(), Disconnected> {
        self.write()?;
        self.write()?;
        self.respond(status);
        Ok(())
    }

    pub fn begin_sse(&mut self) -> Result<(), Disconnected> {
        self.write()
    }

    /// Chunk size, event and chunk terminator writes.
    pub fn send_sse_data(&mut self) -> Result<(), Disconnected> {
        self.write()?;
        self.write()?;
        self.write()
    }

    pub fn finish_sse(&mut self) -> Result<(), Disconnected> {
        self.write()?;
        self.respond(200);
        Ok(())
    }
}

pub struct Job {
    pub request: usize,
    pub connection: Connection,
    pub body: Json,
}

/// The state `server_cli` shares between the accept loop and the generation thread.
pub struct Shared {
    pub pending: Mutex<Option<Job>>,
    pub condition: Condvar,
    pub shutdown: AtomicBool,
    pub available: AtomicBool,
}

impl Shared {
    pub fn new() -> Self {
        Self {
            pending: Mutex::new(None),
            condition: Condvar::new(),
            shutdown: AtomicBool::new(false),
            available: AtomicBool::new(true),
        }
    }
}

/// `condition.wait_timeout_while(guard, 100 ms, condition)`: while the condition holds, the wait either times out (at most `ticks` times per execution, chosen nondeterministically) or waits for a notification.
fn wait_timeout_while<'a, T>(
    condvar: &Condvar,
    mut guard: MutexGuard<'a, T>,
    mut condition: impl FnMut(&mut T) -> bool,
    ticks: &mut usize,
) -> MutexGuard<'a, T> {
    if *ticks > 0 && condition(&mut guard) && thread_rng().gen_bool(0.5) {
        *ticks -= 1;
        return guard;
    }
    condvar.wait_while(guard, condition).unwrap()
}

/// `generation_loop`.
pub fn generation_loop(
    wire: &mut Wire,
    session: &mut ServeSession,
    shared: &Shared,
    log: &Log,
    mut ticks: usize,
) {
    loop {
        let job;
        {
            let guard = shared.pending.lock().unwrap();
            let mut guard = wait_timeout_while(
                &shared.condition,
                guard,
                |slot| !shared.shutdown.load(Ordering::SeqCst) && slot.is_none(),
                &mut ticks,
            );
            if shared.shutdown.load(Ordering::SeqCst) {
                *guard = None;
                break;
            }
            if guard.is_none() {
                let live = wire.alive();
                shared.available.store(live, Ordering::SeqCst);
                if !live {
                    session.invalidate();
                    log.push(Event::Invalidated);
                }
                continue;
            }
            job = guard.take();
        }
        if let Some(job) = job {
            run_job(wire, session, job, shared, log);
        }
    }
}

/// `run_job`.
fn run_job(wire: &mut Wire, session: &mut ServeSession, job: Job, shared: &Shared, log: &Log) {
    log.push(Event::JobStart {
        request: job.request,
    });
    wire.request = job.request;
    let streaming = Cell::new(false);
    let target_started = Cell::new(false);
    let connection = RefCell::new(job.connection);
    // `shutdown || stop_requested() || now >= deadline || disconnected()`.
    let cancelled =
        || shared.shutdown.load(Ordering::SeqCst) || connection.borrow_mut().disconnected();
    let outcome = (|| -> Result<(), String> {
        if cancelled() {
            return Err("queued request cancelled or expired".to_owned());
        }
        let response_id = b"chatcmpl-00000000000000000000000000000000";
        let mut target =
            |request: &Json,
             on_token: &mut dyn FnMut(&Json) -> std::result::Result<(), ServeError>| {
                target_started.set(true);
                let mut probe = || Ok::<bool, CallbackFailed>(cancelled());
                let mut sink = |event: &Json| on_token(event).map_err(|_| CallbackFailed);
                wire.request(request, Some(&mut probe), Some(&mut sink))
                    .map_err(ServeError::from)
            };
        let mut chunk = |event: &Json| -> std::result::Result<(), ServeError> {
            let mut connection = connection.borrow_mut();
            if !streaming.get() {
                connection.begin_sse()?;
                streaming.set(true);
            }
            event.dump_limited(1024 * 1024)?;
            connection.send_sse_data()?;
            Ok(())
        };
        let mut is_cancelled = || cancelled();
        let mut new_seed = || 42;
        let mut sink = ServeSink {
            chunk: Some(&mut chunk),
            cancelled: &mut is_cancelled,
            new_seed: &mut new_seed,
            prefix_trace: None,
        };
        let result = session
            .complete(&job.body, response_id, &mut target, &mut sink)
            .map_err(|error| error.to_string())?;
        let mut connection = connection.borrow_mut();
        let sent = if streaming.get() {
            connection
                .send_sse_data()
                .and_then(|()| connection.finish_sse())
        } else {
            result
                .dump_limited(8 * 1024 * 1024)
                .map_err(|error| error.to_string())?;
            connection.send_json(200)
        };
        sent.map_err(|error| error.to_string())
    })();
    let Err(error) = outcome else {
        log.push(Event::JobEnd {
            request: job.request,
            error: None,
        });
        return;
    };
    session.invalidate();
    shared.available.store(wire.alive(), Ordering::SeqCst);
    log.push(Event::JobEnd {
        request: job.request,
        error: Some(error),
    });
    let mut connection = connection.borrow_mut();
    if !connection.disconnected() && !shared.shutdown.load(Ordering::SeqCst) {
        let sent = if streaming.get() {
            connection
                .send_sse_data()
                .and_then(|()| connection.finish_sse())
        } else {
            connection.send_json(if target_started.get() { 502 } else { 400 })
        };
        if sent.is_err() {
            // Broken clients cannot waive worker draining/cleanup.
        }
    }
}

/// The longest pause, in yields, before a client disconnects: long enough to reach the last command of a branch8 job under random scheduling.
const MAX_PAUSE: usize = 192;

/// The clients and the accept loop: each request is admitted (or answered 503/429 as `route` and `enqueue` do), then either waited for or abandoned after a nondeterministic pause. An abandoning client sends the next request at once, or once the server has dropped the connection. Ends like `server_cli`'s accept loop: shutdown, then wake the generation thread.
pub fn clients(script: Vec<Json>, shared: Arc<Shared>, log: Log, cancel_rate: f64) {
    let mut rng = thread_rng();
    for (request, body) in script.into_iter().enumerate() {
        if !shared.available.load(Ordering::SeqCst) {
            log.push(Event::Answered {
                request,
                status: 503,
            });
            continue;
        }
        admit_chat(&body, b"tokspur").unwrap(); // No mutation on invalid admission.
        let flag = Arc::new(AtomicBool::new(false));
        let (response, responses) = channel();
        {
            let mut slot = shared.pending.lock().unwrap();
            if slot.is_some() {
                drop(slot);
                log.push(Event::Answered {
                    request,
                    status: 429,
                });
                continue;
            }
            *slot = Some(Job {
                request,
                connection: Connection {
                    disconnected: Arc::clone(&flag),
                    response,
                },
                body,
            });
            log.push(Event::Queued { request });
            shared.condition.notify_one();
        }
        let abandon = rng.gen_bool(cancel_rate);
        if abandon {
            let pause = if rng.gen_bool(0.25) {
                0
            } else {
                rng.gen_range(1..=MAX_PAUSE)
            };
            for _ in 0..pause {
                thread::yield_now();
            }
            log.push(Event::Disconnected { request });
            flag.store(true, Ordering::SeqCst);
        }
        if (!abandon || rng.gen_bool(0.5)) && responses.recv().is_err() {
            // The job ended without a response: the connection was dropped.
        }
    }
    shared.shutdown.store(true, Ordering::SeqCst);
    shared.condition.notify_one();
}
