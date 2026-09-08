use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ResilienceChannel {
    Llm,
    Thymos,
    Dcmdb,
}

impl ResilienceChannel {
    const ALL: [Self; 3] = [Self::Llm, Self::Thymos, Self::Dcmdb];

    fn name(self) -> &'static str {
        match self {
            Self::Llm => "llm",
            Self::Thymos => "thymos",
            Self::Dcmdb => "dcmdb",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BreakerState {
    Closed,
    Open,
    HalfOpen,
}

impl BreakerState {
    fn name(self) -> &'static str {
        match self {
            Self::Closed => "CLOSED",
            Self::Open => "OPEN",
            Self::HalfOpen => "HALF_OPEN",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChannelStatus {
    pub state: BreakerState,
    pub consecutive_failures: u32,
    pub maximum_failures: u32,
    pub total_failures: u64,
}

struct Channel {
    state: BreakerState,
    consecutive_failures: u32,
    maximum_failures: u32,
    total_failures: u64,
    cooldown: Duration,
    last_failure: Option<Instant>,
    probe_in_flight: bool,
}

impl Channel {
    fn new(maximum_failures: u32, cooldown: Duration) -> Self {
        Self {
            state: BreakerState::Closed,
            consecutive_failures: 0,
            maximum_failures,
            total_failures: 0,
            cooldown,
            last_failure: None,
            probe_in_flight: false,
        }
    }

    fn allow(&mut self, now: Instant) -> bool {
        match self.state {
            BreakerState::Closed => true,
            BreakerState::Open => {
                let cooled_down = self.last_failure.is_some_and(|failed_at| {
                    now.checked_duration_since(failed_at)
                        .is_some_and(|elapsed| elapsed >= self.cooldown)
                });
                if !cooled_down {
                    return false;
                }
                self.state = BreakerState::HalfOpen;
                self.probe_in_flight = true;
                true
            }
            BreakerState::HalfOpen if !self.probe_in_flight => {
                self.probe_in_flight = true;
                true
            }
            BreakerState::HalfOpen => false,
        }
    }

    fn available(&self, now: Instant) -> bool {
        match self.state {
            BreakerState::Closed => true,
            BreakerState::Open => self.last_failure.is_some_and(|failed_at| {
                now.checked_duration_since(failed_at)
                    .is_some_and(|elapsed| elapsed >= self.cooldown)
            }),
            BreakerState::HalfOpen => !self.probe_in_flight,
        }
    }

    fn record_success(&mut self) {
        self.state = BreakerState::Closed;
        self.consecutive_failures = 0;
        self.last_failure = None;
        self.probe_in_flight = false;
    }

    fn record_failure(&mut self, now: Instant) {
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        self.total_failures = self.total_failures.saturating_add(1);
        self.last_failure = Some(now);
        self.probe_in_flight = false;
        if self.state == BreakerState::HalfOpen
            || self.consecutive_failures >= self.maximum_failures
        {
            self.state = BreakerState::Open;
        }
    }

    fn record_cancelled(&mut self, now: Instant) {
        if self.state == BreakerState::HalfOpen && self.probe_in_flight {
            self.state = BreakerState::Open;
            self.last_failure = Some(now);
            self.probe_in_flight = false;
        }
    }

    fn reset(&mut self) {
        self.state = BreakerState::Closed;
        self.consecutive_failures = 0;
        self.last_failure = None;
        self.probe_in_flight = false;
    }

    fn status(&self) -> ChannelStatus {
        ChannelStatus {
            state: self.state,
            consecutive_failures: self.consecutive_failures,
            maximum_failures: self.maximum_failures,
            total_failures: self.total_failures,
        }
    }
}

pub struct CircuitBreaker {
    channels: BTreeMap<ResilienceChannel, Channel>,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self {
            channels: BTreeMap::from([
                (
                    ResilienceChannel::Llm,
                    Channel::new(3, Duration::from_secs(30)),
                ),
                (
                    ResilienceChannel::Thymos,
                    Channel::new(1, Duration::from_secs(60)),
                ),
                (
                    ResilienceChannel::Dcmdb,
                    Channel::new(2, Duration::from_secs(30)),
                ),
            ]),
        }
    }
}

impl CircuitBreaker {
    pub fn available(&self, channel: ResilienceChannel, now: Instant) -> bool {
        self.channels
            .get(&channel)
            .expect("all resilience channels are initialized")
            .available(now)
    }

    pub fn allow(&mut self, channel: ResilienceChannel, now: Instant) -> bool {
        self.channel_mut(channel).allow(now)
    }

    pub fn record_success(&mut self, channel: ResilienceChannel) {
        self.channel_mut(channel).record_success();
    }

    pub fn record_failure(&mut self, channel: ResilienceChannel, now: Instant) {
        self.channel_mut(channel).record_failure(now);
    }

    pub fn record_cancelled(&mut self, channel: ResilienceChannel, now: Instant) {
        self.channel_mut(channel).record_cancelled(now);
    }

    pub fn reset(&mut self, channel: ResilienceChannel) {
        self.channel_mut(channel).reset();
    }

    pub fn reset_all(&mut self) {
        for channel in self.channels.values_mut() {
            channel.reset();
        }
    }

    pub fn status(&self, channel: ResilienceChannel) -> ChannelStatus {
        self.channels
            .get(&channel)
            .expect("all resilience channels are initialized")
            .status()
    }

    pub fn status_summary(&self) -> String {
        ResilienceChannel::ALL
            .into_iter()
            .map(|channel| {
                let status = self.status(channel);
                format!(
                    "{}: {} (fails={}/{}, total={})",
                    channel.name(),
                    status.state.name(),
                    status.consecutive_failures,
                    status.maximum_failures,
                    status.total_failures,
                )
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }

    fn channel_mut(&mut self, channel: ResilienceChannel) -> &mut Channel {
        self.channels
            .get_mut(&channel)
            .expect("all resilience channels are initialized")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channels_open_at_their_independent_thresholds() {
        let now = Instant::now();
        let mut breaker = CircuitBreaker::default();

        breaker.record_failure(ResilienceChannel::Llm, now);
        breaker.record_failure(ResilienceChannel::Llm, now);
        assert_eq!(
            breaker.status(ResilienceChannel::Llm).state,
            BreakerState::Closed
        );
        breaker.record_failure(ResilienceChannel::Llm, now);
        assert_eq!(
            breaker.status(ResilienceChannel::Llm).state,
            BreakerState::Open
        );

        breaker.record_failure(ResilienceChannel::Thymos, now);
        assert_eq!(
            breaker.status(ResilienceChannel::Thymos).state,
            BreakerState::Open
        );

        breaker.record_failure(ResilienceChannel::Dcmdb, now);
        assert_eq!(
            breaker.status(ResilienceChannel::Dcmdb).state,
            BreakerState::Closed
        );
        breaker.record_failure(ResilienceChannel::Dcmdb, now);
        assert_eq!(
            breaker.status(ResilienceChannel::Dcmdb).state,
            BreakerState::Open
        );
    }

    #[test]
    fn success_resets_only_the_consecutive_failure_count() {
        let now = Instant::now();
        let mut breaker = CircuitBreaker::default();
        breaker.record_failure(ResilienceChannel::Llm, now);
        breaker.record_success(ResilienceChannel::Llm);
        let status = breaker.status(ResilienceChannel::Llm);
        assert_eq!(status.state, BreakerState::Closed);
        assert_eq!(status.consecutive_failures, 0);
        assert_eq!(status.total_failures, 1);
    }

    #[test]
    fn cooldown_allows_exactly_one_half_open_probe() {
        let now = Instant::now();
        let mut breaker = CircuitBreaker::default();
        breaker.record_failure(ResilienceChannel::Thymos, now);
        assert!(!breaker.allow(ResilienceChannel::Thymos, now + Duration::from_secs(59)));
        assert!(breaker.allow(ResilienceChannel::Thymos, now + Duration::from_secs(60)));
        assert_eq!(
            breaker.status(ResilienceChannel::Thymos).state,
            BreakerState::HalfOpen
        );
        assert!(!breaker.allow(ResilienceChannel::Thymos, now + Duration::from_secs(61)));
    }

    #[test]
    fn availability_check_does_not_consume_the_half_open_probe() {
        let now = Instant::now();
        let mut breaker = CircuitBreaker::default();
        breaker.record_failure(ResilienceChannel::Thymos, now);
        let cooled_down = now + Duration::from_secs(60);
        assert!(breaker.available(ResilienceChannel::Thymos, cooled_down));
        assert!(breaker.available(ResilienceChannel::Thymos, cooled_down));
        assert!(breaker.allow(ResilienceChannel::Thymos, cooled_down));
        assert!(!breaker.available(ResilienceChannel::Thymos, cooled_down));
    }

    #[test]
    fn half_open_probe_resolves_to_open_or_closed() {
        let now = Instant::now();
        let mut breaker = CircuitBreaker::default();
        breaker.record_failure(ResilienceChannel::Thymos, now);
        assert!(breaker.allow(ResilienceChannel::Thymos, now + Duration::from_secs(60)));
        breaker.record_failure(ResilienceChannel::Thymos, now + Duration::from_secs(60));
        assert_eq!(
            breaker.status(ResilienceChannel::Thymos).state,
            BreakerState::Open
        );
        assert!(!breaker.allow(ResilienceChannel::Thymos, now + Duration::from_secs(119)));
        assert!(breaker.allow(ResilienceChannel::Thymos, now + Duration::from_secs(120)));
        breaker.record_success(ResilienceChannel::Thymos);
        assert_eq!(
            breaker.status(ResilienceChannel::Thymos).state,
            BreakerState::Closed
        );
    }

    #[test]
    fn cancelled_half_open_probe_reopens_without_counting_a_failure() {
        let now = Instant::now();
        let mut breaker = CircuitBreaker::default();
        breaker.record_failure(ResilienceChannel::Thymos, now);
        assert!(breaker.allow(ResilienceChannel::Thymos, now + Duration::from_secs(60)));
        breaker.record_cancelled(ResilienceChannel::Thymos, now + Duration::from_secs(61));
        let status = breaker.status(ResilienceChannel::Thymos);
        assert_eq!(status.state, BreakerState::Open);
        assert_eq!(status.total_failures, 1);
        assert!(!breaker.available(ResilienceChannel::Thymos, now + Duration::from_secs(120)));
        assert!(breaker.available(ResilienceChannel::Thymos, now + Duration::from_secs(121)));
    }

    #[test]
    fn reset_preserves_diagnostics_and_summary_order_is_stable() {
        let now = Instant::now();
        let mut breaker = CircuitBreaker::default();
        breaker.record_failure(ResilienceChannel::Thymos, now);
        breaker.record_failure(ResilienceChannel::Dcmdb, now);
        assert_eq!(
            breaker.status_summary(),
            "llm: CLOSED (fails=0/3, total=0) | thymos: OPEN (fails=1/1, total=1) | dcmdb: CLOSED (fails=1/2, total=1)"
        );
        breaker.reset(ResilienceChannel::Thymos);
        breaker.reset_all();
        assert_eq!(breaker.status(ResilienceChannel::Thymos).total_failures, 1);
        assert_eq!(breaker.status(ResilienceChannel::Dcmdb).total_failures, 1);
        assert_eq!(
            breaker.status(ResilienceChannel::Llm).state,
            BreakerState::Closed
        );
    }
}
