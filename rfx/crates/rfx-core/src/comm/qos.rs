//! QoS profiles for transport publish/subscribe.
//!
//! Maps conceptually to Zenoh QoS settings with sensible presets
//! for common robotics use-cases.

/// Message delivery reliability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reliability {
    /// Fire-and-forget — lowest latency, may drop messages.
    BestEffort,
    /// Acknowledged delivery — retransmit on loss.
    Reliable,
}

/// Behavior when the outgoing buffer is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CongestionControl {
    /// Drop newest message when buffer is full.
    Drop,
    /// Block the publisher until space is available.
    Block,
}

/// Message priority (1 = real-time, 7 = background).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Priority(pub u8);

impl Priority {
    pub const REAL_TIME: Self = Self(1);
    pub const INTERACTIVE_HIGH: Self = Self(2);
    pub const INTERACTIVE_LOW: Self = Self(3);
    pub const DATA_HIGH: Self = Self(4);
    pub const DATA: Self = Self(5);
    pub const DATA_LOW: Self = Self(6);
    pub const BACKGROUND: Self = Self(7);
}

impl Default for Priority {
    fn default() -> Self {
        Self::DATA
    }
}

/// Message durability — whether late-joining subscribers see past messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durability {
    /// Messages are only delivered to subscribers present at publish time.
    Volatile,
    /// The last N messages are replayed to new subscribers (latch).
    TransientLocal,
}

/// Quality-of-service profile for transport operations.
#[derive(Debug, Clone)]
pub struct QoSProfile {
    pub reliability: Reliability,
    pub congestion_control: CongestionControl,
    pub priority: Priority,
    pub durability: Durability,
    /// Skip Zenoh routing overhead for LAN-only scenarios.
    pub express: bool,
    /// Number of historical samples to keep for TransientLocal durability.
    pub history_depth: usize,
}

impl Default for QoSProfile {
    fn default() -> Self {
        Self {
            reliability: Reliability::Reliable,
            congestion_control: CongestionControl::Block,
            priority: Priority::DATA,
            durability: Durability::Volatile,
            express: false,
            history_depth: 1,
        }
    }
}

impl QoSProfile {
    /// High-frequency sensor data: best-effort, drop on backpressure, express.
    pub fn sensor_data() -> Self {
        Self {
            reliability: Reliability::BestEffort,
            congestion_control: CongestionControl::Drop,
            priority: Priority::REAL_TIME,
            durability: Durability::Volatile,
            express: true,
            history_depth: 1,
        }
    }

    /// Reliable command/control messages: reliable, block on backpressure.
    pub fn reliable() -> Self {
        Self {
            reliability: Reliability::Reliable,
            congestion_control: CongestionControl::Block,
            priority: Priority::INTERACTIVE_HIGH,
            durability: Durability::Volatile,
            express: false,
            history_depth: 1,
        }
    }

    /// Parameter/config data: reliable, transient-local, background priority.
    pub fn parameters() -> Self {
        Self {
            reliability: Reliability::Reliable,
            congestion_control: CongestionControl::Block,
            priority: Priority::BACKGROUND,
            durability: Durability::TransientLocal,
            express: false,
            history_depth: 1,
        }
    }

    /// System events (discovery, lifecycle): reliable, data priority.
    pub fn system_events() -> Self {
        Self {
            reliability: Reliability::Reliable,
            congestion_control: CongestionControl::Drop,
            priority: Priority::DATA,
            durability: Durability::Volatile,
            express: false,
            history_depth: 1,
        }
    }

    /// Builder: set history depth.
    pub fn with_history_depth(mut self, depth: usize) -> Self {
        self.history_depth = depth;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_data_preset() {
        let qos = QoSProfile::sensor_data();
        assert_eq!(qos.reliability, Reliability::BestEffort);
        assert_eq!(qos.congestion_control, CongestionControl::Drop);
        assert!(qos.express);
        assert_eq!(qos.priority, Priority::REAL_TIME);
    }

    #[test]
    fn test_reliable_preset() {
        let qos = QoSProfile::reliable();
        assert_eq!(qos.reliability, Reliability::Reliable);
        assert_eq!(qos.congestion_control, CongestionControl::Block);
    }

    #[test]
    fn test_parameters_preset() {
        let qos = QoSProfile::parameters();
        assert_eq!(qos.durability, Durability::TransientLocal);
        assert_eq!(qos.priority, Priority::BACKGROUND);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::REAL_TIME < Priority::BACKGROUND);
        assert!(Priority::INTERACTIVE_HIGH < Priority::DATA);
    }

    #[test]
    fn test_history_depth_builder() {
        let qos = QoSProfile::parameters().with_history_depth(10);
        assert_eq!(qos.history_depth, 10);
    }
}
