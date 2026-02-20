//! Communication primitives for rfx framework
//!
//! Provides channels, topics, streams, transport, QoS, services,
//! discovery, and parameters for inter-component communication.
//! Designed for both same-process (lock-free) and multi-process scenarios.

mod channel;
mod stream;
mod topic;
mod transport;
mod qos;
mod schema;
mod service;
mod discovery;
mod params;

#[cfg(feature = "zenoh")]
mod zenoh;

pub use channel::{bounded_channel, unbounded_channel, Channel, Receiver, Sender};
pub use stream::{Stream, StreamConfig, StreamHandle};
pub use topic::{Topic, TopicConfig};
pub use transport::{InprocTransport, TransportBackend, TransportEnvelope, TransportSubscription};
pub use qos::*;
pub use schema::{envelope_type_name, publish_typed, MessageSchema, SchemaRegistry};
pub use service::{
    InprocServiceBackend, ServiceArity, ServiceBackend, ServiceHandler, ServiceRequest,
    ServiceResponse, ServiceServerHandle, ServiceStatus,
};
pub use discovery::{
    DiscoveryBackend, DiscoveryConfig, DiscoveryEvent, InprocDiscovery, LivelinessToken, TopicInfo,
};
pub use params::{ParamValue, ParameterClient, ParameterServer};

#[cfg(feature = "zenoh")]
pub use self::zenoh::{ZenohContext, ZenohTransport, ZenohTransportConfig};

#[cfg(feature = "zenoh")]
pub use service::ZenohServiceBackend;

#[cfg(feature = "zenoh")]
pub use discovery::ZenohDiscovery;
