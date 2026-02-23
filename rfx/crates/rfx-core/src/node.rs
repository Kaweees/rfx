//! Universal robot node — bridges any `Robot` to the transport layer.
//!
//! `RobotNode` is the ONE pipeline for all robots. It wraps any `impl Robot`
//! and:
//! - Publishes `RobotState` to `rfx/{name}/state` at a configurable rate
//! - Subscribes to `rfx/{name}/cmd` and forwards `Command` to the robot
//! - Declares the node via `DiscoveryBackend` for auto-discovery
//!
//! Works with both `ZenohTransport` (cross-process) and `InprocTransport`
//! (same-process). Callers pick the backend; the node doesn't care.
//!
//! # Example
//!
//! ```no_run
//! use rfx_core::node::{RobotNode, RobotNodeConfig};
//! use rfx_core::comm::InprocTransport;
//! use rfx_core::hardware::so101::{So101, So101Config};
//! use std::sync::Arc;
//!
//! let transport = Arc::new(InprocTransport::new());
//! let arm = So101::connect(So101Config::follower("/dev/ttyACM0")).unwrap();
//!
//! let config = RobotNodeConfig {
//!     name: "my-arm".into(),
//!     publish_rate_hz: 50.0,
//! };
//!
//! let node = RobotNode::spawn(arm, transport, config).unwrap();
//! // State is now being published. Commands can be received.
//! // node.stop() to shut down.
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::comm::{DiscoveryBackend, LivelinessToken, TransportBackend, TransportSubscription};
use crate::hardware::{Command, Robot, RobotState};
use crate::Error;

/// Configuration for a robot node.
#[derive(Debug, Clone)]
pub struct RobotNodeConfig {
    /// Node name — used for topic keys: `rfx/{name}/state`, `rfx/{name}/cmd`.
    pub name: String,
    /// State publish rate in Hz. Default: 50.0.
    pub publish_rate_hz: f64,
}

impl Default for RobotNodeConfig {
    fn default() -> Self {
        Self {
            name: "robot".into(),
            publish_rate_hz: 50.0,
        }
    }
}

/// Universal robot node that bridges any `Robot` to the transport layer.
pub struct RobotNode {
    running: Arc<AtomicBool>,
    publish_handle: Option<JoinHandle<()>>,
    cmd_handle: Option<JoinHandle<()>>,
    _discovery_token: Option<LivelinessToken>,
    _pub_token: Option<LivelinessToken>,
    config: RobotNodeConfig,
}

impl RobotNode {
    /// Spawn a robot node that publishes state and accepts commands.
    ///
    /// The robot is moved into the node and owned by it. State is published
    /// on a background thread at `publish_rate_hz`. Commands received on the
    /// transport are forwarded to `robot.send_command()`.
    pub fn spawn<R: Robot + 'static>(
        robot: R,
        transport: Arc<dyn TransportBackend>,
        config: RobotNodeConfig,
    ) -> crate::Result<Self> {
        Self::spawn_with_discovery(robot, transport, None, config)
    }

    /// Spawn with optional discovery backend for auto-registration.
    pub fn spawn_with_discovery<R: Robot + 'static>(
        robot: R,
        transport: Arc<dyn TransportBackend>,
        discovery: Option<Arc<dyn DiscoveryBackend>>,
        config: RobotNodeConfig,
    ) -> crate::Result<Self> {
        if config.publish_rate_hz <= 0.0 || !config.publish_rate_hz.is_finite() {
            return Err(Error::Config(format!(
                "publish_rate_hz must be a positive finite number, got {}",
                config.publish_rate_hz
            )));
        }
        if config.name.is_empty() {
            return Err(Error::Config("node name must not be empty".into()));
        }

        let running = Arc::new(AtomicBool::new(true));
        let robot = Arc::new(robot);

        let state_key = format!("rfx/{}/state", config.name);
        let cmd_key = format!("rfx/{}/cmd", config.name);

        // Register with discovery
        let discovery_token = if let Some(ref disc) = discovery {
            Some(disc.declare_node(&config.name)?)
        } else {
            None
        };
        let pub_token = if let Some(ref disc) = discovery {
            Some(disc.declare_publisher(&config.name, &state_key)?)
        } else {
            None
        };

        // Subscribe to commands
        let cmd_sub = transport.subscribe(&cmd_key, 64);

        // Spawn state publisher thread
        let publish_handle = {
            let robot = Arc::clone(&robot);
            let transport = Arc::clone(&transport);
            let running = Arc::clone(&running);
            let state_key = state_key.clone();
            let interval = Duration::from_secs_f64(1.0 / config.publish_rate_hz);

            thread::Builder::new()
                .name(format!("rfx-node-{}-pub", config.name))
                .spawn(move || {
                    Self::publish_loop(robot, transport, &state_key, interval, running);
                })
                .map_err(|e| Error::Hardware(format!("Failed to spawn publish thread: {e}")))?
        };

        // Spawn command receiver thread
        let cmd_handle = {
            let robot = Arc::clone(&robot);
            let running = Arc::clone(&running);

            thread::Builder::new()
                .name(format!("rfx-node-{}-cmd", config.name))
                .spawn(move || {
                    Self::cmd_loop(robot, cmd_sub, running);
                })
                .map_err(|e| Error::Hardware(format!("Failed to spawn cmd thread: {e}")))?
        };

        tracing::info!(
            "RobotNode '{}' started (pub: {state_key}, cmd: {cmd_key})",
            config.name
        );

        Ok(Self {
            running,
            publish_handle: Some(publish_handle),
            cmd_handle: Some(cmd_handle),
            _discovery_token: discovery_token,
            _pub_token: pub_token,
            config,
        })
    }

    /// State publish loop — serializes RobotState and publishes at fixed rate.
    fn publish_loop(
        robot: Arc<dyn Robot>,
        transport: Arc<dyn TransportBackend>,
        state_key: &str,
        interval: Duration,
        running: Arc<AtomicBool>,
    ) {
        while running.load(Ordering::Relaxed) {
            let loop_start = Instant::now();

            let state = robot.state();
            match serde_json::to_vec(&state) {
                Ok(payload) => {
                    let metadata = serde_json::json!({
                        "num_joints": state.joint_positions.len(),
                        "timestamp": state.timestamp,
                    });
                    let metadata_str = metadata.to_string();

                    transport.publish(
                        state_key,
                        Arc::from(payload.into_boxed_slice()),
                        Some(Arc::from(metadata_str.into_boxed_str())),
                    );
                }
                Err(e) => {
                    tracing::warn!("Failed to serialize state: {e}");
                }
            }

            // Always sleep to maintain rate, even on error.
            let elapsed = loop_start.elapsed();
            if elapsed < interval {
                thread::sleep(interval - elapsed);
            }
        }
    }

    /// Command receive loop — deserializes Command and forwards to robot.
    fn cmd_loop(
        robot: Arc<dyn Robot>,
        subscription: TransportSubscription,
        running: Arc<AtomicBool>,
    ) {
        while running.load(Ordering::Relaxed) {
            let envelope = match subscription.recv_timeout(Duration::from_millis(100)) {
                Some(env) => env,
                None => continue,
            };

            let cmd: Command = match serde_json::from_slice(&envelope.payload) {
                Ok(cmd) => cmd,
                Err(e) => {
                    tracing::warn!("Invalid command payload: {e}");
                    continue;
                }
            };

            if let Err(e) = robot.send_command(cmd) {
                tracing::warn!("Failed to send command: {e}");
            }
        }
    }

    /// Get the node name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Check if the node is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Stop the node and join threads.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);

        if let Some(handle) = self.publish_handle.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.cmd_handle.take() {
            let _ = handle.join();
        }

        tracing::info!("RobotNode '{}' stopped", self.config.name);
    }
}

impl Drop for RobotNode {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Helpers for creating nodes with standard topic key conventions.
pub mod keys {
    /// State topic key for a robot.
    pub fn state_key(name: &str) -> String {
        format!("rfx/{name}/state")
    }

    /// Command topic key for a robot.
    pub fn cmd_key(name: &str) -> String {
        format!("rfx/{name}/cmd")
    }

    /// Wildcard for all state topics.
    pub fn all_states() -> &'static str {
        "rfx/*/state"
    }

    /// Wildcard for everything under a robot name.
    pub fn robot_all(name: &str) -> String {
        format!("rfx/{name}/**")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comm::InprocTransport;
    use crate::hardware::MockRobot;

    #[test]
    fn test_robot_node_publish_subscribe() {
        let transport = Arc::new(InprocTransport::new());
        let robot = MockRobot::new(6);

        let config = RobotNodeConfig {
            name: "test-arm".into(),
            publish_rate_hz: 100.0,
        };

        let mut node = RobotNode::spawn(robot, transport.clone(), config).unwrap();

        // Subscribe to state
        let sub = transport.subscribe("rfx/test-arm/state", 16);

        // Wait for at least one publish
        let env = sub.recv_timeout(Duration::from_secs(1));
        assert!(env.is_some(), "expected to receive a state message");

        let env = env.unwrap();
        assert_eq!(env.key.as_ref(), "rfx/test-arm/state");

        // Deserialize the state
        let state: RobotState = serde_json::from_slice(&env.payload).unwrap();
        assert_eq!(state.joint_positions.len(), 6);

        node.stop();
    }

    #[test]
    fn test_robot_node_command() {
        let transport = Arc::new(InprocTransport::new());
        let robot = MockRobot::new(6);

        let config = RobotNodeConfig {
            name: "test-cmd".into(),
            publish_rate_hz: 100.0,
        };

        let mut node = RobotNode::spawn(robot, transport.clone(), config).unwrap();

        // Send a command via transport
        let cmd = Command::position(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let cmd_bytes = serde_json::to_vec(&cmd).unwrap();
        transport.publish(
            "rfx/test-cmd/cmd",
            Arc::from(cmd_bytes.into_boxed_slice()),
            None,
        );

        // Give the command loop time to process
        thread::sleep(Duration::from_millis(200));

        // Read state back — MockRobot should reflect the command
        let sub = transport.subscribe("rfx/test-cmd/state", 16);
        let env = sub.recv_timeout(Duration::from_secs(1));
        assert!(env.is_some());

        node.stop();
    }

    #[test]
    fn test_key_helpers() {
        assert_eq!(keys::state_key("my-arm"), "rfx/my-arm/state");
        assert_eq!(keys::cmd_key("my-arm"), "rfx/my-arm/cmd");
        assert_eq!(keys::all_states(), "rfx/*/state");
        assert_eq!(keys::robot_all("go2"), "rfx/go2/**");
    }

    #[test]
    fn test_invalid_publish_rate_zero() {
        let transport = Arc::new(InprocTransport::new());
        let robot = MockRobot::new(6);
        let config = RobotNodeConfig {
            name: "bad-rate".into(),
            publish_rate_hz: 0.0,
        };
        match RobotNode::spawn(robot, transport, config) {
            Ok(_) => panic!("expected error for publish_rate_hz=0"),
            Err(e) => {
                let msg = e.to_string();
                assert!(msg.contains("positive finite"), "got: {msg}");
            }
        }
    }

    #[test]
    fn test_invalid_publish_rate_negative() {
        let transport = Arc::new(InprocTransport::new());
        let robot = MockRobot::new(6);
        let config = RobotNodeConfig {
            name: "neg-rate".into(),
            publish_rate_hz: -10.0,
        };
        assert!(RobotNode::spawn(robot, transport, config).is_err());
    }

    #[test]
    fn test_invalid_publish_rate_nan() {
        let transport = Arc::new(InprocTransport::new());
        let robot = MockRobot::new(6);
        let config = RobotNodeConfig {
            name: "nan-rate".into(),
            publish_rate_hz: f64::NAN,
        };
        assert!(RobotNode::spawn(robot, transport, config).is_err());
    }

    #[test]
    fn test_empty_node_name() {
        let transport = Arc::new(InprocTransport::new());
        let robot = MockRobot::new(6);
        let config = RobotNodeConfig {
            name: "".into(),
            publish_rate_hz: 50.0,
        };
        match RobotNode::spawn(robot, transport, config) {
            Ok(_) => panic!("expected error for empty name"),
            Err(e) => {
                let msg = e.to_string();
                assert!(msg.contains("name"), "got: {msg}");
            }
        }
    }

    #[test]
    fn test_node_stop_is_idempotent() {
        let transport = Arc::new(InprocTransport::new());
        let robot = MockRobot::new(6);
        let config = RobotNodeConfig {
            name: "stop-twice".into(),
            publish_rate_hz: 100.0,
        };
        let mut node = RobotNode::spawn(robot, transport, config).unwrap();
        assert!(node.is_running());
        node.stop();
        assert!(!node.is_running());
        // Second stop should not panic.
        node.stop();
        assert!(!node.is_running());
    }
}
