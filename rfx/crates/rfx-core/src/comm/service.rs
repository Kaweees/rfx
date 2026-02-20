//! Service (request/response) pattern for RPC-style communication.
//!
//! Two backends:
//! - **Inproc**: direct function call within the same process.
//! - **Zenoh** (feature-gated): queryable/get pattern over the network.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// Whether a service expects a single reply or a stream of replies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceArity {
    /// Single request → single response (default).
    Unary,
    /// Single request → multiple responses (used by actions).
    Streaming,
}

impl Default for ServiceArity {
    fn default() -> Self {
        Self::Unary
    }
}

/// Serialized service request.
#[derive(Debug, Clone)]
pub struct ServiceRequest {
    pub request_id: u64,
    pub timeout_ms: u64,
    pub payload: Vec<u8>,
}

/// Serialized service response.
#[derive(Debug, Clone)]
pub struct ServiceResponse {
    pub request_id: u64,
    pub status: ServiceStatus,
    pub error_code: i32,
    pub error_message: String,
    pub payload: Vec<u8>,
}

/// Response status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceStatus {
    Ok,
    Error,
    Timeout,
    Canceled,
}

impl ServiceStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Error => "error",
            Self::Timeout => "timeout",
            Self::Canceled => "canceled",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "ok" => Self::Ok,
            "error" => Self::Error,
            "timeout" => Self::Timeout,
            "canceled" => Self::Canceled,
            _ => Self::Error,
        }
    }
}

impl ServiceResponse {
    pub fn ok(request_id: u64, payload: Vec<u8>) -> Self {
        Self {
            request_id,
            status: ServiceStatus::Ok,
            error_code: 0,
            error_message: String::new(),
            payload,
        }
    }

    pub fn error(request_id: u64, code: i32, message: impl Into<String>) -> Self {
        Self {
            request_id,
            status: ServiceStatus::Error,
            error_code: code,
            error_message: message.into(),
            payload: Vec::new(),
        }
    }

    pub fn timeout(request_id: u64) -> Self {
        Self {
            request_id,
            status: ServiceStatus::Timeout,
            error_code: -1,
            error_message: "service call timed out".into(),
            payload: Vec::new(),
        }
    }
}

/// Handler function type for service servers.
pub type ServiceHandler = Arc<dyn Fn(ServiceRequest) -> ServiceResponse + Send + Sync>;

/// Opaque handle returned when registering a service server.
/// Dropping this handle unregisters the service.
pub struct ServiceServerHandle {
    name: String,
    backend: Arc<dyn ServiceBackend>,
}

impl ServiceServerHandle {
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for ServiceServerHandle {
    fn drop(&mut self) {
        self.backend.unserve(&self.name);
    }
}

/// Backend trait for service (request/response) communication.
pub trait ServiceBackend: Send + Sync {
    /// Register a service handler.
    fn serve(
        &self,
        name: &str,
        handler: ServiceHandler,
        arity: ServiceArity,
    ) -> crate::Result<()>;

    /// Unregister a service handler.
    fn unserve(&self, name: &str);

    /// Call a remote service and wait for response.
    fn call(
        &self,
        name: &str,
        request: ServiceRequest,
        timeout: Duration,
    ) -> crate::Result<ServiceResponse>;

    /// List registered service names.
    fn list_services(&self) -> Vec<String>;
}

// ============================================================================
// In-process service backend
// ============================================================================

struct InprocServiceEntry {
    handler: ServiceHandler,
    _arity: ServiceArity,
}

/// In-process service backend — direct function call, no serialization overhead.
pub struct InprocServiceBackend {
    services: RwLock<HashMap<String, InprocServiceEntry>>,
}

impl InprocServiceBackend {
    pub fn new() -> Self {
        Self {
            services: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InprocServiceBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ServiceBackend for InprocServiceBackend {
    fn serve(
        &self,
        name: &str,
        handler: ServiceHandler,
        arity: ServiceArity,
    ) -> crate::Result<()> {
        self.services.write().insert(
            name.to_owned(),
            InprocServiceEntry {
                handler,
                _arity: arity,
            },
        );
        Ok(())
    }

    fn unserve(&self, name: &str) {
        self.services.write().remove(name);
    }

    fn call(
        &self,
        name: &str,
        request: ServiceRequest,
        _timeout: Duration,
    ) -> crate::Result<ServiceResponse> {
        let services = self.services.read();
        let entry = services
            .get(name)
            .ok_or_else(|| crate::Error::Communication(format!("service not found: {name}")))?;
        Ok((entry.handler)(request))
    }

    fn list_services(&self) -> Vec<String> {
        self.services.read().keys().cloned().collect()
    }
}

// ============================================================================
// Zenoh service backend (feature-gated)
// ============================================================================

#[cfg(feature = "zenoh")]
mod zenoh_service {
    use super::*;
    use zenoh::bytes::ZBytes;
    use zenoh::query::Queryable;
    use zenoh::Wait;

    struct ZenohServiceEntry {
        _queryable: Queryable<()>,
        _arity: ServiceArity,
    }

    /// Zenoh-backed service backend using queryable/get pattern.
    pub struct ZenohServiceBackend {
        session: zenoh::Session,
        key_prefix: String,
        services: RwLock<HashMap<String, ZenohServiceEntry>>,
    }

    impl ZenohServiceBackend {
        pub fn new(session: zenoh::Session, key_prefix: String) -> Self {
            Self {
                session,
                key_prefix,
                services: RwLock::new(HashMap::new()),
            }
        }

        fn full_key(&self, name: &str) -> String {
            if self.key_prefix.is_empty() {
                format!("rfx/srv/{name}")
            } else {
                format!("{}/srv/{name}", self.key_prefix)
            }
        }
    }

    impl ServiceBackend for ZenohServiceBackend {
        fn serve(
            &self,
            name: &str,
            handler: ServiceHandler,
            arity: ServiceArity,
        ) -> crate::Result<()> {
            let key = self.full_key(name);
            let handler = handler.clone();

            let queryable = self
                .session
                .declare_queryable(&key)
                .callback(move |query| {
                    let req_bytes: Vec<u8> = query
                        .payload()
                        .map(|p| p.to_bytes().to_vec())
                        .unwrap_or_default();

                    let request: ServiceRequest =
                        match serde_json::from_slice(&req_bytes) {
                            Ok(r) => r,
                            Err(_) => ServiceRequest {
                                request_id: 0,
                                timeout_ms: 5000,
                                payload: req_bytes,
                            },
                        };

                    let response = handler(request);
                    let resp_bytes = serde_json::to_vec(&ServiceResponseWire::from(&response))
                        .unwrap_or_default();

                    let _ = query
                        .reply(query.key_expr(), ZBytes::from(resp_bytes))
                        .wait();
                })
                .wait()
                .map_err(|e| {
                    crate::Error::Communication(format!("failed to declare queryable: {e}"))
                })?;

            self.services.write().insert(
                name.to_owned(),
                ZenohServiceEntry {
                    _queryable: queryable,
                    _arity: arity,
                },
            );
            Ok(())
        }

        fn unserve(&self, name: &str) {
            // Dropping the entry drops the queryable, undeclaring it.
            self.services.write().remove(name);
        }

        fn call(
            &self,
            name: &str,
            request: ServiceRequest,
            timeout: Duration,
        ) -> crate::Result<ServiceResponse> {
            let key = self.full_key(name);
            let req_bytes = serde_json::to_vec(&ServiceRequestWire::from(&request))
                .map_err(|e| crate::Error::Communication(format!("serialize request: {e}")))?;

            let replies = self
                .session
                .get(&key)
                .payload(ZBytes::from(req_bytes))
                .timeout(timeout)
                .wait()
                .map_err(|e| {
                    crate::Error::Communication(format!("service call failed: {e}"))
                })?;

            // Unary: take first successful reply.
            let mut first_response: Option<ServiceResponse> = None;
            while let Ok(reply) = replies.recv() {
                if let Ok(sample) = reply.into_result() {
                    let resp_bytes: Vec<u8> = sample.payload().to_bytes().to_vec();
                    if let Ok(wire) = serde_json::from_slice::<ServiceResponseWire>(&resp_bytes)
                    {
                        let resp = wire.into_response();
                        if first_response.is_none() {
                            first_response = Some(resp);
                        } else {
                            tracing::warn!(
                                "service '{}': discarding additional reply (unary mode)",
                                name
                            );
                        }
                    }
                }
            }

            first_response.ok_or_else(|| {
                crate::Error::Timeout(format!("service '{}' timed out", name))
            })
        }

        fn list_services(&self) -> Vec<String> {
            self.services.read().keys().cloned().collect()
        }
    }

    // Wire format for JSON serialization
    #[derive(serde::Serialize, serde::Deserialize)]
    struct ServiceRequestWire {
        request_id: u64,
        timeout_ms: u64,
        payload: String, // base64
    }

    impl From<&ServiceRequest> for ServiceRequestWire {
        fn from(r: &ServiceRequest) -> Self {
            use base64_encode;
            Self {
                request_id: r.request_id,
                timeout_ms: r.timeout_ms,
                payload: base64_encode(&r.payload),
            }
        }
    }

    #[derive(serde::Serialize, serde::Deserialize)]
    struct ServiceResponseWire {
        request_id: u64,
        status: String,
        error_code: i32,
        error_message: String,
        payload: String, // base64
    }

    impl From<&ServiceResponse> for ServiceResponseWire {
        fn from(r: &ServiceResponse) -> Self {
            Self {
                request_id: r.request_id,
                status: r.status.as_str().into(),
                error_code: r.error_code,
                error_message: r.error_message.clone(),
                payload: base64_encode(&r.payload),
            }
        }
    }

    impl ServiceResponseWire {
        fn into_response(self) -> ServiceResponse {
            ServiceResponse {
                request_id: self.request_id,
                status: ServiceStatus::from_str(&self.status),
                error_code: self.error_code,
                error_message: self.error_message,
                payload: base64_decode(&self.payload),
            }
        }
    }

    fn base64_encode(data: &[u8]) -> String {
        use std::io::Write;
        let mut buf = Vec::with_capacity(data.len() * 4 / 3 + 4);
        // Simple hex encoding as a stand-in — no base64 crate dependency.
        for byte in data {
            let _ = write!(buf, "{:02x}", byte);
        }
        String::from_utf8(buf).unwrap_or_default()
    }

    fn base64_decode(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .filter_map(|i| {
                s.get(i..i + 2)
                    .and_then(|hex| u8::from_str_radix(hex, 16).ok())
            })
            .collect()
    }
}

#[cfg(feature = "zenoh")]
pub use zenoh_service::ZenohServiceBackend;

// ============================================================================
// JSON serialization support for ServiceRequest/ServiceResponse
// ============================================================================

impl serde::Serialize for ServiceRequest {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("ServiceRequest", 3)?;
        s.serialize_field("request_id", &self.request_id)?;
        s.serialize_field("timeout_ms", &self.timeout_ms)?;
        // Encode payload as hex string for JSON compat
        let hex: String = self.payload.iter().map(|b| format!("{:02x}", b)).collect();
        s.serialize_field("payload", &hex)?;
        s.end()
    }
}

impl<'de> serde::Deserialize<'de> for ServiceRequest {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Helper {
            request_id: u64,
            timeout_ms: u64,
            payload: String,
        }
        let h = Helper::deserialize(deserializer)?;
        let payload: Vec<u8> = (0..h.payload.len())
            .step_by(2)
            .filter_map(|i| {
                h.payload
                    .get(i..i + 2)
                    .and_then(|hex| u8::from_str_radix(hex, 16).ok())
            })
            .collect();
        Ok(ServiceRequest {
            request_id: h.request_id,
            timeout_ms: h.timeout_ms,
            payload,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inproc_service_roundtrip() {
        let backend = InprocServiceBackend::new();

        // Register an echo service
        let handler: ServiceHandler = Arc::new(|req| ServiceResponse::ok(req.request_id, req.payload));
        backend
            .serve("echo", handler, ServiceArity::Unary)
            .unwrap();

        assert_eq!(backend.list_services().len(), 1);

        let request = ServiceRequest {
            request_id: 42,
            timeout_ms: 5000,
            payload: b"hello".to_vec(),
        };
        let response = backend
            .call("echo", request, Duration::from_secs(5))
            .unwrap();
        assert_eq!(response.request_id, 42);
        assert_eq!(response.status, ServiceStatus::Ok);
        assert_eq!(response.payload, b"hello");
    }

    #[test]
    fn test_service_not_found() {
        let backend = InprocServiceBackend::new();
        let request = ServiceRequest {
            request_id: 1,
            timeout_ms: 1000,
            payload: vec![],
        };
        let result = backend.call("missing", request, Duration::from_secs(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_unserve() {
        let backend = InprocServiceBackend::new();
        let handler: ServiceHandler = Arc::new(|req| ServiceResponse::ok(req.request_id, vec![]));
        backend.serve("temp", handler, ServiceArity::Unary).unwrap();
        assert_eq!(backend.list_services().len(), 1);
        backend.unserve("temp");
        assert!(backend.list_services().is_empty());
    }

    #[test]
    fn test_service_response_constructors() {
        let ok = ServiceResponse::ok(1, b"data".to_vec());
        assert_eq!(ok.status, ServiceStatus::Ok);
        assert_eq!(ok.error_code, 0);

        let err = ServiceResponse::error(2, 404, "not found");
        assert_eq!(err.status, ServiceStatus::Error);
        assert_eq!(err.error_code, 404);

        let timeout = ServiceResponse::timeout(3);
        assert_eq!(timeout.status, ServiceStatus::Timeout);
    }

    #[test]
    fn test_service_status_roundtrip() {
        for status in [
            ServiceStatus::Ok,
            ServiceStatus::Error,
            ServiceStatus::Timeout,
            ServiceStatus::Canceled,
        ] {
            assert_eq!(ServiceStatus::from_str(status.as_str()), status);
        }
    }
}
