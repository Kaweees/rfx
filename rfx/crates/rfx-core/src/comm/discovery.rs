//! Node and topic discovery for the rfx runtime.
//!
//! Two backends:
//! - **Inproc**: immediate HashSet-based registry for single-process.
//! - **Zenoh** (feature-gated): liveliness-token based, eventually consistent.

use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

/// Configuration for discovery behaviour.
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Prune entries not refreshed within this TTL (seconds). Default: 10.0.
    pub stale_ttl_s: f64,
    /// Re-query liveliness periodically (seconds). Default: 3.0.
    pub refresh_interval_s: f64,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            stale_ttl_s: 10.0,
            refresh_interval_s: 3.0,
        }
    }
}

/// Discovery event types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveryEvent {
    NodeJoined(String),
    NodeLeft(String),
    TopicPublished { topic: String, node: String },
    TopicUnpublished { topic: String, node: String },
}

/// Info about a discovered topic.
#[derive(Debug, Clone)]
pub struct TopicInfo {
    pub name: String,
    pub publishers: Vec<String>,
    pub subscribers: Vec<String>,
}

/// Opaque token that keeps a liveliness declaration alive.
/// Dropping this token removes the declaration.
pub struct LivelinessToken {
    _name: String,
    _on_drop: Option<Box<dyn FnOnce() + Send>>,
}

impl LivelinessToken {
    fn new(name: String, on_drop: impl FnOnce() + Send + 'static) -> Self {
        Self {
            _name: name,
            _on_drop: Some(Box::new(on_drop)),
        }
    }

    fn simple(name: String) -> Self {
        Self {
            _name: name,
            _on_drop: None,
        }
    }
}

impl Drop for LivelinessToken {
    fn drop(&mut self) {
        if let Some(f) = self._on_drop.take() {
            f();
        }
    }
}

/// Backend trait for discovery.
///
/// APIs are **eventually consistent** — discovery events may lag reality
/// by up to `stale_ttl_s`. Callers should not rely on discovery for
/// correctness, only for UX (e.g. `rfx topic-list`, graph visualisation).
pub trait DiscoveryBackend: Send + Sync {
    /// Declare this process as a named node.
    fn declare_node(&self, name: &str) -> crate::Result<LivelinessToken>;

    /// Declare that a node publishes on a topic.
    fn declare_publisher(&self, node: &str, topic: &str) -> crate::Result<LivelinessToken>;

    /// Declare that a node subscribes to a topic.
    fn declare_subscriber(&self, node: &str, topic: &str) -> crate::Result<LivelinessToken>;

    /// List currently known node names.
    fn list_nodes(&self) -> Vec<String>;

    /// List currently known topics with publisher/subscriber info.
    fn list_topics(&self) -> Vec<TopicInfo>;

    /// Register a callback for discovery events.
    fn on_event(&self, callback: Arc<dyn Fn(DiscoveryEvent) + Send + Sync>) -> crate::Result<()>;
}

// ============================================================================
// In-process discovery backend
// ============================================================================

/// In-process discovery — immediate, no TTL needed.
pub struct InprocDiscovery {
    nodes: RwLock<HashSet<String>>,
    publishers: RwLock<HashMap<String, HashSet<String>>>,  // topic -> nodes
    subscribers: RwLock<HashMap<String, HashSet<String>>>,  // topic -> nodes
    callbacks: RwLock<Vec<Arc<dyn Fn(DiscoveryEvent) + Send + Sync>>>,
}

impl InprocDiscovery {
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashSet::new()),
            publishers: RwLock::new(HashMap::new()),
            subscribers: RwLock::new(HashMap::new()),
            callbacks: RwLock::new(Vec::new()),
        }
    }

    fn emit(&self, event: DiscoveryEvent) {
        let cbs = self.callbacks.read();
        for cb in cbs.iter() {
            cb(event.clone());
        }
    }
}

impl Default for InprocDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl DiscoveryBackend for InprocDiscovery {
    fn declare_node(&self, name: &str) -> crate::Result<LivelinessToken> {
        let name_owned = name.to_owned();
        self.nodes.write().insert(name_owned.clone());
        self.emit(DiscoveryEvent::NodeJoined(name_owned.clone()));

        let nodes = Arc::new(self.nodes.read().clone());
        // We can't hold a reference to self in the closure, so use a simple token
        Ok(LivelinessToken::simple(name_owned))
    }

    fn declare_publisher(&self, node: &str, topic: &str) -> crate::Result<LivelinessToken> {
        let topic_owned = topic.to_owned();
        let node_owned = node.to_owned();
        self.publishers
            .write()
            .entry(topic_owned.clone())
            .or_default()
            .insert(node_owned.clone());
        self.emit(DiscoveryEvent::TopicPublished {
            topic: topic_owned.clone(),
            node: node_owned,
        });
        Ok(LivelinessToken::simple(topic_owned))
    }

    fn declare_subscriber(&self, node: &str, topic: &str) -> crate::Result<LivelinessToken> {
        let topic_owned = topic.to_owned();
        let node_owned = node.to_owned();
        self.subscribers
            .write()
            .entry(topic_owned.clone())
            .or_default()
            .insert(node_owned);
        Ok(LivelinessToken::simple(topic_owned))
    }

    fn list_nodes(&self) -> Vec<String> {
        self.nodes.read().iter().cloned().collect()
    }

    fn list_topics(&self) -> Vec<TopicInfo> {
        let pubs = self.publishers.read();
        let subs = self.subscribers.read();
        let mut all_topics: HashSet<&String> = HashSet::new();
        all_topics.extend(pubs.keys());
        all_topics.extend(subs.keys());

        all_topics
            .into_iter()
            .map(|topic| TopicInfo {
                name: topic.clone(),
                publishers: pubs
                    .get(topic)
                    .map(|s| s.iter().cloned().collect())
                    .unwrap_or_default(),
                subscribers: subs
                    .get(topic)
                    .map(|s| s.iter().cloned().collect())
                    .unwrap_or_default(),
            })
            .collect()
    }

    fn on_event(&self, callback: Arc<dyn Fn(DiscoveryEvent) + Send + Sync>) -> crate::Result<()> {
        self.callbacks.write().push(callback);
        Ok(())
    }
}

// ============================================================================
// Zenoh discovery backend (feature-gated)
// ============================================================================

#[cfg(feature = "zenoh")]
mod zenoh_discovery {
    use super::*;
    use zenoh::Wait;

    struct DiscoveryCacheEntry {
        last_seen: Instant,
    }

    /// Zenoh-backed discovery using liveliness tokens.
    ///
    /// Liveliness keys:
    /// - `rfx/alive/node/{name}`
    /// - `rfx/alive/pub/{topic}/{node}`
    /// - `rfx/alive/sub/{topic}/{node}`
    pub struct ZenohDiscovery {
        session: zenoh::Session,
        key_prefix: String,
        config: DiscoveryConfig,
        nodes: RwLock<HashMap<String, DiscoveryCacheEntry>>,
        publishers: RwLock<HashMap<String, HashMap<String, DiscoveryCacheEntry>>>,
        subscribers: RwLock<HashMap<String, HashMap<String, DiscoveryCacheEntry>>>,
        callbacks: RwLock<Vec<Arc<dyn Fn(DiscoveryEvent) + Send + Sync>>>,
    }

    impl ZenohDiscovery {
        pub fn new(
            session: zenoh::Session,
            key_prefix: String,
            config: DiscoveryConfig,
        ) -> crate::Result<Self> {
            let disc = Self {
                session,
                key_prefix,
                config,
                nodes: RwLock::new(HashMap::new()),
                publishers: RwLock::new(HashMap::new()),
                subscribers: RwLock::new(HashMap::new()),
                callbacks: RwLock::new(Vec::new()),
            };
            Ok(disc)
        }

        fn alive_prefix(&self) -> String {
            if self.key_prefix.is_empty() {
                "rfx/alive".into()
            } else {
                format!("{}/alive", self.key_prefix)
            }
        }

        fn emit(&self, event: DiscoveryEvent) {
            let cbs = self.callbacks.read();
            for cb in cbs.iter() {
                cb(event.clone());
            }
        }

        /// Prune entries older than stale_ttl_s.
        pub fn prune_stale(&self) {
            let cutoff = Instant::now()
                - std::time::Duration::from_secs_f64(self.config.stale_ttl_s);

            let mut nodes = self.nodes.write();
            let before = nodes.len();
            nodes.retain(|name, entry| {
                if entry.last_seen < cutoff {
                    self.emit(DiscoveryEvent::NodeLeft(name.clone()));
                    false
                } else {
                    true
                }
            });

            let mut pubs = self.publishers.write();
            for (topic, nodes_map) in pubs.iter_mut() {
                nodes_map.retain(|node, entry| {
                    if entry.last_seen < cutoff {
                        self.emit(DiscoveryEvent::TopicUnpublished {
                            topic: topic.clone(),
                            node: node.clone(),
                        });
                        false
                    } else {
                        true
                    }
                });
            }
            pubs.retain(|_, nodes_map| !nodes_map.is_empty());

            let mut subs = self.subscribers.write();
            for (_, nodes_map) in subs.iter_mut() {
                nodes_map.retain(|_, entry| entry.last_seen >= cutoff);
            }
            subs.retain(|_, nodes_map| !nodes_map.is_empty());
        }
    }

    impl DiscoveryBackend for ZenohDiscovery {
        fn declare_node(&self, name: &str) -> crate::Result<LivelinessToken> {
            let key = format!("{}/node/{}", self.alive_prefix(), name);
            let token = self
                .session
                .liveliness()
                .declare_token(&key)
                .wait()
                .map_err(|e| {
                    crate::Error::Communication(format!("liveliness declare failed: {e}"))
                })?;

            self.nodes.write().insert(
                name.to_owned(),
                DiscoveryCacheEntry {
                    last_seen: Instant::now(),
                },
            );
            self.emit(DiscoveryEvent::NodeJoined(name.to_owned()));

            // Keep the zenoh token alive by moving it into the LivelinessToken
            Ok(LivelinessToken::new(name.to_owned(), move || {
                drop(token);
            }))
        }

        fn declare_publisher(&self, node: &str, topic: &str) -> crate::Result<LivelinessToken> {
            let key = format!("{}/pub/{}/{}", self.alive_prefix(), topic, node);
            let token = self
                .session
                .liveliness()
                .declare_token(&key)
                .wait()
                .map_err(|e| {
                    crate::Error::Communication(format!("liveliness declare failed: {e}"))
                })?;

            self.publishers
                .write()
                .entry(topic.to_owned())
                .or_default()
                .insert(
                    node.to_owned(),
                    DiscoveryCacheEntry {
                        last_seen: Instant::now(),
                    },
                );

            Ok(LivelinessToken::new(
                format!("{}/{}", topic, node),
                move || {
                    drop(token);
                },
            ))
        }

        fn declare_subscriber(&self, node: &str, topic: &str) -> crate::Result<LivelinessToken> {
            let key = format!("{}/sub/{}/{}", self.alive_prefix(), topic, node);
            let token = self
                .session
                .liveliness()
                .declare_token(&key)
                .wait()
                .map_err(|e| {
                    crate::Error::Communication(format!("liveliness declare failed: {e}"))
                })?;

            self.subscribers
                .write()
                .entry(topic.to_owned())
                .or_default()
                .insert(
                    node.to_owned(),
                    DiscoveryCacheEntry {
                        last_seen: Instant::now(),
                    },
                );

            Ok(LivelinessToken::new(
                format!("{}/{}", topic, node),
                move || {
                    drop(token);
                },
            ))
        }

        fn list_nodes(&self) -> Vec<String> {
            self.nodes.read().keys().cloned().collect()
        }

        fn list_topics(&self) -> Vec<TopicInfo> {
            let pubs = self.publishers.read();
            let subs = self.subscribers.read();
            let mut all_topics: HashSet<&String> = HashSet::new();
            all_topics.extend(pubs.keys());
            all_topics.extend(subs.keys());

            all_topics
                .into_iter()
                .map(|topic| TopicInfo {
                    name: topic.clone(),
                    publishers: pubs
                        .get(topic)
                        .map(|m| m.keys().cloned().collect())
                        .unwrap_or_default(),
                    subscribers: subs
                        .get(topic)
                        .map(|m| m.keys().cloned().collect())
                        .unwrap_or_default(),
                })
                .collect()
        }

        fn on_event(
            &self,
            callback: Arc<dyn Fn(DiscoveryEvent) + Send + Sync>,
        ) -> crate::Result<()> {
            self.callbacks.write().push(callback);
            Ok(())
        }
    }
}

#[cfg(feature = "zenoh")]
pub use zenoh_discovery::ZenohDiscovery;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inproc_discovery_nodes() {
        let disc = InprocDiscovery::new();
        let _token = disc.declare_node("robot.teleop").unwrap();
        assert_eq!(disc.list_nodes().len(), 1);
        assert!(disc.list_nodes().contains(&"robot.teleop".to_string()));
    }

    #[test]
    fn test_inproc_discovery_topics() {
        let disc = InprocDiscovery::new();
        let _pub_token = disc.declare_publisher("node_a", "joint_state").unwrap();
        let _sub_token = disc.declare_subscriber("node_b", "joint_state").unwrap();

        let topics = disc.list_topics();
        assert_eq!(topics.len(), 1);
        let topic = &topics[0];
        assert_eq!(topic.name, "joint_state");
        assert!(topic.publishers.contains(&"node_a".to_string()));
        assert!(topic.subscribers.contains(&"node_b".to_string()));
    }

    #[test]
    fn test_inproc_discovery_events() {
        let disc = InprocDiscovery::new();
        let events = Arc::new(RwLock::new(Vec::new()));
        let events_clone = events.clone();
        disc.on_event(Arc::new(move |e| {
            events_clone.write().push(e);
        }))
        .unwrap();

        let _token = disc.declare_node("test_node").unwrap();
        let evts = events.read();
        assert_eq!(evts.len(), 1);
        assert_eq!(evts[0], DiscoveryEvent::NodeJoined("test_node".into()));
    }
}
