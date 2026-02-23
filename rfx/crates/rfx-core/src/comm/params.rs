//! Parameter server and client for dynamic node configuration.
//!
//! **Server** (per-node): stores parameters, exposes get/set/list via services,
//! publishes change notifications via transport.
//!
//! **Client**: calls remote parameter services, watches for change notifications.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use super::service::{
    ServiceArity, ServiceBackend, ServiceHandler, ServiceRequest, ServiceResponse,
};
use super::transport::{TransportBackend, TransportSubscription};

/// A parameter value — JSON-compatible.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Json(serde_json::Value),
}

impl ParamValue {
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            ParamValue::Bool(v) => serde_json::Value::Bool(*v),
            ParamValue::Int(v) => serde_json::json!(*v),
            ParamValue::Float(v) => serde_json::json!(*v),
            ParamValue::String(v) => serde_json::Value::String(v.clone()),
            ParamValue::Json(v) => v.clone(),
        }
    }

    pub fn from_json(v: &serde_json::Value) -> Self {
        match v {
            serde_json::Value::Bool(b) => ParamValue::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    ParamValue::Int(i)
                } else {
                    ParamValue::Float(n.as_f64().unwrap_or(0.0))
                }
            }
            serde_json::Value::String(s) => ParamValue::String(s.clone()),
            other => ParamValue::Json(other.clone()),
        }
    }
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Per-node parameter server.
///
/// Registers get/set/list services and publishes change notifications.
pub struct ParameterServer {
    node_name: String,
    store: Arc<RwLock<HashMap<String, ParamValue>>>,
    transport: Arc<dyn TransportBackend>,
}

impl ParameterServer {
    /// Create a new parameter server and register services.
    pub fn new(
        node_name: &str,
        service_backend: &dyn ServiceBackend,
        transport: Arc<dyn TransportBackend>,
    ) -> crate::Result<Self> {
        let store = Arc::new(RwLock::new(HashMap::<String, ParamValue>::new()));

        // Register GET service
        let store_get = store.clone();
        let get_handler: ServiceHandler = Arc::new(move |req: ServiceRequest| {
            let payload: serde_json::Value =
                serde_json::from_slice(&req.payload).unwrap_or_default();
            let key = payload.get("key").and_then(|v| v.as_str()).unwrap_or("");

            let params = store_get.read();
            if let Some(value) = params.get(key) {
                let resp = serde_json::json!({"value": value.to_json()});
                ServiceResponse::ok(req.request_id, resp.to_string().into_bytes())
            } else {
                ServiceResponse::error(req.request_id, 404, format!("parameter not found: {key}"))
            }
        });
        service_backend.serve(
            &format!("param/{node_name}/get"),
            get_handler,
            ServiceArity::Unary,
        )?;

        // Register SET service
        let store_set = store.clone();
        let transport_set = transport.clone();
        let node_for_set = node_name.to_owned();
        let set_handler: ServiceHandler = Arc::new(move |req: ServiceRequest| {
            let payload: serde_json::Value =
                serde_json::from_slice(&req.payload).unwrap_or_default();
            let key = payload
                .get("key")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned();
            let value = payload.get("value");

            if key.is_empty() || value.is_none() {
                return ServiceResponse::error(req.request_id, 400, "missing key or value");
            }

            let param_value = ParamValue::from_json(value.unwrap());
            store_set.write().insert(key.clone(), param_value.clone());

            // Publish change notification
            let notification = serde_json::json!({
                "value": param_value.to_json(),
                "timestamp_ns": now_ns(),
                "source_node": node_for_set,
            });
            let topic = format!("rfx/param/{}/{}", node_for_set, key);
            let payload_bytes = notification.to_string().into_bytes();
            transport_set.publish(&topic, Arc::from(payload_bytes.into_boxed_slice()), None);

            let resp = serde_json::json!({"status": "ok"});
            ServiceResponse::ok(req.request_id, resp.to_string().into_bytes())
        });
        service_backend.serve(
            &format!("param/{node_name}/set"),
            set_handler,
            ServiceArity::Unary,
        )?;

        // Register LIST service
        let store_list = store.clone();
        let list_handler: ServiceHandler = Arc::new(move |req: ServiceRequest| {
            let params = store_list.read();
            let keys: Vec<&str> = params.keys().map(|k| k.as_str()).collect();
            let resp = serde_json::json!({"keys": keys});
            ServiceResponse::ok(req.request_id, resp.to_string().into_bytes())
        });
        service_backend.serve(
            &format!("param/{node_name}/list"),
            list_handler,
            ServiceArity::Unary,
        )?;

        Ok(Self {
            node_name: node_name.to_owned(),
            store,
            transport,
        })
    }

    /// Declare a parameter with a default value.
    pub fn declare(&self, key: &str, default: ParamValue) {
        self.store.write().entry(key.to_owned()).or_insert(default);
    }

    /// Get a parameter value (local read).
    pub fn get(&self, key: &str) -> Option<ParamValue> {
        self.store.read().get(key).cloned()
    }

    /// Set a parameter value (local write + publish notification).
    pub fn set(&self, key: &str, value: ParamValue) -> crate::Result<()> {
        self.store.write().insert(key.to_owned(), value.clone());

        let notification = serde_json::json!({
            "value": value.to_json(),
            "timestamp_ns": now_ns(),
            "source_node": self.node_name,
        });
        let topic = format!("rfx/param/{}/{}", self.node_name, key);
        let payload_bytes = notification.to_string().into_bytes();
        self.transport
            .publish(&topic, Arc::from(payload_bytes.into_boxed_slice()), None);
        Ok(())
    }

    /// List all parameter keys.
    pub fn list(&self) -> Vec<String> {
        self.store.read().keys().cloned().collect()
    }
}

/// Client for reading/writing parameters on remote nodes.
pub struct ParameterClient {
    service_backend: Arc<dyn ServiceBackend>,
    transport: Arc<dyn TransportBackend>,
}

impl ParameterClient {
    pub fn new(
        service_backend: Arc<dyn ServiceBackend>,
        transport: Arc<dyn TransportBackend>,
    ) -> Self {
        Self {
            service_backend,
            transport,
        }
    }

    /// Get a parameter from a remote node via service call.
    pub fn get(&self, node: &str, key: &str) -> crate::Result<ParamValue> {
        let req_payload = serde_json::json!({"key": key}).to_string().into_bytes();
        let request = ServiceRequest {
            request_id: now_ns(),
            timeout_ms: 5000,
            payload: req_payload,
        };
        let response = self.service_backend.call(
            &format!("param/{node}/get"),
            request,
            Duration::from_secs(5),
        )?;

        if response.status != super::service::ServiceStatus::Ok {
            return Err(crate::Error::Communication(response.error_message));
        }

        let resp_json: serde_json::Value = serde_json::from_slice(&response.payload)
            .map_err(|e| crate::Error::Communication(format!("invalid response: {e}")))?;

        resp_json
            .get("value")
            .map(ParamValue::from_json)
            .ok_or_else(|| crate::Error::Communication("missing value in response".into()))
    }

    /// Set a parameter on a remote node via service call.
    pub fn set(&self, node: &str, key: &str, value: ParamValue) -> crate::Result<()> {
        let req_payload = serde_json::json!({"key": key, "value": value.to_json()})
            .to_string()
            .into_bytes();
        let request = ServiceRequest {
            request_id: now_ns(),
            timeout_ms: 5000,
            payload: req_payload,
        };
        let response = self.service_backend.call(
            &format!("param/{node}/set"),
            request,
            Duration::from_secs(5),
        )?;

        if response.status != super::service::ServiceStatus::Ok {
            return Err(crate::Error::Communication(response.error_message));
        }
        Ok(())
    }

    /// List parameters on a remote node via service call.
    pub fn list(&self, node: &str) -> crate::Result<Vec<String>> {
        let request = ServiceRequest {
            request_id: now_ns(),
            timeout_ms: 5000,
            payload: b"{}".to_vec(),
        };
        let response = self.service_backend.call(
            &format!("param/{node}/list"),
            request,
            Duration::from_secs(5),
        )?;

        if response.status != super::service::ServiceStatus::Ok {
            return Err(crate::Error::Communication(response.error_message));
        }

        let resp_json: serde_json::Value = serde_json::from_slice(&response.payload)
            .map_err(|e| crate::Error::Communication(format!("invalid response: {e}")))?;

        Ok(resp_json
            .get("keys")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default())
    }

    /// Watch for parameter changes on a remote node via pub/sub.
    pub fn watch(&self, node: &str, key: &str, capacity: usize) -> TransportSubscription {
        let pattern = format!("rfx/param/{node}/{key}");
        self.transport.subscribe(&pattern, capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comm::service::InprocServiceBackend;
    use crate::comm::InprocTransport;

    #[test]
    fn test_parameter_server_declare_get() {
        let service = InprocServiceBackend::new();
        let transport = Arc::new(InprocTransport::new());
        let server = ParameterServer::new("test_node", &service, transport.clone()).unwrap();

        server.declare("rate_hz", ParamValue::Float(50.0));
        assert_eq!(server.get("rate_hz"), Some(ParamValue::Float(50.0)));
        assert_eq!(server.get("missing"), None);
    }

    #[test]
    fn test_parameter_server_set_publishes_notification() {
        let service = InprocServiceBackend::new();
        let transport = Arc::new(InprocTransport::new());
        let sub = transport.subscribe("rfx/param/test_node/rate_hz", 8);

        let server = ParameterServer::new("test_node", &service, transport.clone()).unwrap();
        server.set("rate_hz", ParamValue::Float(100.0)).unwrap();

        let env = sub.recv_timeout(Duration::from_millis(100)).unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&env.payload).unwrap();
        assert_eq!(payload["value"], 100.0);
        assert_eq!(payload["source_node"], "test_node");
    }

    #[test]
    fn test_parameter_server_list() {
        let service = InprocServiceBackend::new();
        let transport = Arc::new(InprocTransport::new());
        let server = ParameterServer::new("test_node", &service, transport).unwrap();

        server.declare("a", ParamValue::Int(1));
        server.declare("b", ParamValue::Int(2));
        let mut keys = server.list();
        keys.sort();
        assert_eq!(keys, vec!["a", "b"]);
    }

    #[test]
    fn test_parameter_client_get_set_list() {
        let service = Arc::new(InprocServiceBackend::new());
        let transport = Arc::new(InprocTransport::new());

        // Set up server
        let _server = ParameterServer::new("node_a", service.as_ref(), transport.clone()).unwrap();

        // Set up client
        let client = ParameterClient::new(service.clone(), transport.clone());

        // Set via client
        client
            .set("node_a", "kp", ParamValue::Float(std::f64::consts::PI))
            .unwrap();

        // Get via client
        let value = client.get("node_a", "kp").unwrap();
        assert_eq!(value, ParamValue::Float(std::f64::consts::PI));

        // List via client
        let keys = client.list("node_a").unwrap();
        assert!(keys.contains(&"kp".to_string()));
    }

    #[test]
    fn test_parameter_client_watch() {
        let service = Arc::new(InprocServiceBackend::new());
        let transport = Arc::new(InprocTransport::new());

        let server = ParameterServer::new("node_b", service.as_ref(), transport.clone()).unwrap();
        let client = ParameterClient::new(service.clone(), transport.clone());

        let watch = client.watch("node_b", "rate", 8);
        server.set("rate", ParamValue::Float(200.0)).unwrap();

        let env = watch.recv_timeout(Duration::from_millis(100)).unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&env.payload).unwrap();
        assert_eq!(payload["value"], 200.0);
    }

    #[test]
    fn test_param_value_roundtrip() {
        for val in [
            ParamValue::Bool(true),
            ParamValue::Int(42),
            ParamValue::Float(std::f64::consts::PI),
            ParamValue::String("hello".into()),
        ] {
            let json = val.to_json();
            let back = ParamValue::from_json(&json);
            assert_eq!(val, back);
        }
    }
}
