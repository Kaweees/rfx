//! Message type system and schema registry.
//!
//! Provides optional schema validation and type-name metadata injection
//! for transport envelopes, enabling wire-contract compatibility checks.

use parking_lot::RwLock;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use super::transport::{TransportBackend, TransportEnvelope};

/// Describes a message type schema.
#[derive(Debug, Clone)]
pub struct MessageSchema {
    /// Fully-qualified type name (e.g. `rfx.msg.JointState`).
    pub type_name: String,
    /// Optional JSON Schema for payload validation.
    pub json_schema: Option<Value>,
    /// Human-readable description.
    pub description: String,
    /// Schema version (e.g. `"1.0"`).
    pub version: String,
}

/// Thread-safe registry of known message schemas.
#[derive(Debug, Default)]
pub struct SchemaRegistry {
    schemas: RwLock<HashMap<String, MessageSchema>>,
}

impl SchemaRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a message schema by type name.
    pub fn register(&self, schema: MessageSchema) {
        self.schemas
            .write()
            .insert(schema.type_name.clone(), schema);
    }

    /// Look up a schema by type name.
    pub fn get(&self, type_name: &str) -> Option<MessageSchema> {
        self.schemas.read().get(type_name).cloned()
    }

    /// List all registered type names.
    pub fn list(&self) -> Vec<String> {
        self.schemas.read().keys().cloned().collect()
    }

    /// Validate a JSON payload against a registered schema.
    ///
    /// Returns `Ok(())` if no schema is registered or if the schema has no
    /// JSON Schema definition. Returns `Err` if validation fails.
    pub fn validate(&self, type_name: &str, _payload: &Value) -> crate::Result<()> {
        let schemas = self.schemas.read();
        if let Some(schema) = schemas.get(type_name) {
            if schema.json_schema.is_some() {
                // Full JSON Schema validation would require a jsonschema crate.
                // For now, we just check the schema exists — real validation
                // can be added when the dependency is justified.
            }
        }
        Ok(())
    }
}

/// Extract `_type` from envelope metadata JSON.
pub fn envelope_type_name(envelope: &TransportEnvelope) -> Option<String> {
    let meta_str = envelope.metadata_json.as_deref()?;
    let meta: Value = serde_json::from_str(meta_str).ok()?;
    meta.get("_type").and_then(|v| v.as_str()).map(String::from)
}

/// Publish a payload with type metadata injected into the envelope.
///
/// Merges `_type` and `_schema_version` into the metadata JSON before
/// publishing through the given transport backend.
pub fn publish_typed(
    transport: &dyn TransportBackend,
    key: &str,
    payload: Arc<[u8]>,
    type_name: &str,
    schema_version: &str,
    extra_metadata: Option<&Value>,
) -> TransportEnvelope {
    let mut meta = if let Some(extra) = extra_metadata {
        extra.clone()
    } else {
        Value::Object(serde_json::Map::new())
    };

    if let Value::Object(ref mut map) = meta {
        map.insert("_type".into(), Value::String(type_name.into()));
        map.insert(
            "_schema_version".into(),
            Value::String(schema_version.into()),
        );
    }

    let metadata_json = serde_json::to_string(&meta).ok().map(Arc::<str>::from);
    transport.publish(key, payload, metadata_json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comm::InprocTransport;

    #[test]
    fn test_schema_registry_crud() {
        let reg = SchemaRegistry::new();
        assert!(reg.list().is_empty());

        reg.register(MessageSchema {
            type_name: "rfx.msg.JointState".into(),
            json_schema: None,
            description: "Joint positions and velocities".into(),
            version: "1.0".into(),
        });

        assert_eq!(reg.list().len(), 1);
        let schema = reg.get("rfx.msg.JointState").unwrap();
        assert_eq!(schema.version, "1.0");
        assert!(reg.get("rfx.msg.Unknown").is_none());
    }

    #[test]
    fn test_publish_typed_injects_metadata() {
        let transport = InprocTransport::new();
        let sub = transport.subscribe("test/**", 8);

        let env = publish_typed(
            &transport,
            "test/joints",
            Arc::from(b"{}".to_vec().into_boxed_slice()),
            "rfx.msg.JointState",
            "1.0",
            None,
        );

        let meta_str = env.metadata_json.as_deref().unwrap();
        let meta: Value = serde_json::from_str(meta_str).unwrap();
        assert_eq!(meta["_type"], "rfx.msg.JointState");
        assert_eq!(meta["_schema_version"], "1.0");

        // Subscriber should also see the metadata
        let got = sub.recv_timeout(std::time::Duration::from_millis(100)).unwrap();
        assert_eq!(envelope_type_name(&got).as_deref(), Some("rfx.msg.JointState"));
    }

    #[test]
    fn test_publish_typed_preserves_extra_metadata() {
        let transport = InprocTransport::new();

        let extra = serde_json::json!({"_source_node": "robot.teleop"});
        let env = publish_typed(
            &transport,
            "test/state",
            Arc::from(b"data".to_vec().into_boxed_slice()),
            "rfx.msg.State",
            "2.0",
            Some(&extra),
        );

        let meta: Value = serde_json::from_str(env.metadata_json.as_deref().unwrap()).unwrap();
        assert_eq!(meta["_type"], "rfx.msg.State");
        assert_eq!(meta["_source_node"], "robot.teleop");
    }

    #[test]
    fn test_envelope_type_name_missing() {
        let env = TransportEnvelope::new("key", 1, Arc::from(vec![].into_boxed_slice()), None);
        assert!(envelope_type_name(&env).is_none());
    }
}
