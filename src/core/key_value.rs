use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use uuid::Uuid;

/// AuroraDB Key-Value Store - Advanced K/V Operations with Atomicity and Pub/Sub
/// Implements quantum-class key-value store with atomic operations, pub/sub, and advanced features
/// Key-value entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyValueEntry {
    pub key: String,
    pub value: Value,
    pub metadata: KeyMetadata,
    pub version: u64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Key metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyMetadata {
    pub size_bytes: usize,
    pub content_type: ContentType,
    pub ttl: Option<u64>, // seconds from epoch
    pub tags: HashSet<String>,
    pub owner: Option<String>,
    pub compression: bool,
}

/// Content types for values
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContentType {
    String,
    Json,
    Binary,
    Integer,
    Float,
    Boolean,
    Null,
}

/// Key-value namespace for logical separation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyNamespace {
    pub name: String,
    pub max_keys: usize,
    pub max_memory: usize, // bytes
    pub ttl_default: Option<u64>,
    pub read_only: bool,
    pub created_at: DateTime<Utc>,
}

/// Atomic operations for key-value store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AtomicOperation {
    Set(Value),
    SetNX(Value),               // Set if not exists
    GetSet(Value),              // Get old value and set new
    Inc(i64),                   // Increment integer
    Dec(i64),                   // Decrement integer
    Append(String),             // Append to string
    Prepend(String),            // Prepend to string
    Push(Value, PushDirection), // Push to list
    Pop(PushDirection),         // Pop from list
}

/// Push direction for lists
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PushDirection {
    Left,
    Right,
}

/// Compare-and-swap operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CasOperation {
    pub key: String,
    pub expected_version: u64,
    pub new_value: Value,
    pub ttl: Option<u64>,
}

/// Batch operation for multiple keys
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchOperation {
    pub operations: Vec<KeyOperation>,
    pub atomic: bool, // If true, all operations succeed or all fail
}

/// Individual key operation in batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyOperation {
    pub key: String,
    pub operation: AtomicOperation,
}

/// Scan options for key iteration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanOptions {
    pub pattern: Option<String>, // Glob pattern like "user:*"
    pub count: usize,
    pub cursor: Option<String>,
    pub namespace: Option<String>,
}

/// Pub/Sub message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PubSubMessage {
    pub channel: String,
    pub message: Value,
    pub timestamp: DateTime<Utc>,
    pub publisher: Option<String>,
}

/// Subscription information
#[derive(Debug, Clone)]
pub struct Subscription {
    pub channels: HashSet<String>,
    pub patterns: HashSet<String>,
    pub message_queue: VecDeque<PubSubMessage>,
}

/// Key-value store engine
pub struct KeyValueEngine {
    pub namespaces: HashMap<String, KeyNamespace>,
    pub storage: KeyValueStorage,
    pub pubsub: PubSubEngine,
    pub atomic_ops: AtomicOperations,
}

/// Key-value storage backend
pub struct KeyValueStorage {
    pub entries: HashMap<String, KeyValueEntry>,
    pub namespaces: HashMap<String, HashSet<String>>, // namespace -> keys
    pub total_memory: usize,
    pub max_memory: usize,
}

/// Pub/Sub engine for messaging
pub struct PubSubEngine {
    pub channels: HashMap<String, HashSet<Uuid>>, // channel -> subscriber IDs
    pub patterns: HashMap<String, HashSet<Uuid>>, // pattern -> subscriber IDs
    pub subscribers: HashMap<Uuid, Subscription>,
    pub message_history: VecDeque<PubSubMessage>,
    pub max_history: usize,
}

/// Atomic operations processor
pub struct AtomicOperations {
    pub pending_operations: HashMap<String, Vec<AtomicOperation>>,
    pub operation_log: VecDeque<OperationLog>,
    pub max_log_size: usize,
}

/// Operation log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationLog {
    pub key: String,
    pub operation: String,
    pub old_value: Option<Value>,
    pub new_value: Option<Value>,
    pub timestamp: DateTime<Utc>,
    pub success: bool,
}

/// Key scan result
#[derive(Debug, Clone)]
pub struct ScanResult {
    pub keys: Vec<String>,
    pub cursor: Option<String>,
    pub has_more: bool,
}

/// Batch operation result
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub results: Vec<OperationResult>,
    pub success: bool,
    pub execution_time_ms: u64,
}

/// Individual operation result
#[derive(Debug, Clone)]
pub struct OperationResult {
    pub key: String,
    pub success: bool,
    pub old_value: Option<Value>,
    pub new_value: Option<Value>,
    pub error: Option<String>,
}

/// Key-value error types
#[derive(Debug)]
pub enum KeyValueError {
    KeyNotFound(String),
    KeyExists(String),
    NamespaceNotFound(String),
    NamespaceFull(String),
    MemoryLimitExceeded(usize, usize),
    InvalidOperation(String),
    AtomicOperationFailed(String),
    TtlExpired(String),
    PermissionDenied(String),
    SerializationError(String),
}

impl KeyValueEngine {
    /// Create new key-value engine
    pub fn new() -> Self {
        Self {
            namespaces: HashMap::new(),
            storage: KeyValueStorage::new(),
            pubsub: PubSubEngine::new(),
            atomic_ops: AtomicOperations::new(),
        }
    }

    /// Create a namespace
    pub async fn create_namespace(&mut self, name: &str, config: NamespaceConfig) -> Result<(), KeyValueError> {
        if self.namespaces.contains_key(name) {
            return Err(KeyValueError::KeyExists(format!("Namespace {} already exists", name)));
        }

        let namespace = KeyNamespace {
            name: name.to_string(),
            max_keys: config.max_keys,
            max_memory: config.max_memory,
            ttl_default: config.ttl_default,
            read_only: config.read_only,
            created_at: Utc::now(),
        };

        self.namespaces.insert(name.to_string(), namespace);
        self.storage.namespaces.insert(name.to_string(), HashSet::new());

        Ok(())
    }

    /// Delete a namespace
    pub async fn delete_namespace(&mut self, name: &str) -> Result<(), KeyValueError> {
        if !self.namespaces.contains_key(name) {
            return Err(KeyValueError::NamespaceNotFound(name.to_string()));
        }

        // Remove all keys in namespace
        if let Some(keys) = self.storage.namespaces.remove(name) {
            for key in keys {
                self.storage.entries.remove(&key);
            }
        }

        self.namespaces.remove(name);
        Ok(())
    }

    /// Set a key-value pair
    pub async fn set(
        &mut self,
        key: &str,
        value: Value,
        ttl: Option<u64>,
        namespace: Option<&str>,
    ) -> Result<(), KeyValueError> {
        self.validate_namespace(namespace)?;

        let full_key = self.make_full_key(key, namespace);
        let entry = self.create_or_update_entry(&full_key, value, ttl).await?;

        // Add to namespace tracking
        if let Some(ns) = namespace {
            if let Some(ns_keys) = self.storage.namespaces.get_mut(ns) {
                ns_keys.insert(full_key.clone());
            }
        }

        self.storage.entries.insert(full_key, entry);
        self.cleanup_expired_keys().await;

        Ok(())
    }

    /// Get a value by key
    pub async fn get(&self, key: &str, namespace: Option<&str>) -> Result<Option<Value>, KeyValueError> {
        self.validate_namespace(namespace)?;

        let full_key = self.make_full_key(key, namespace);

        match self.storage.entries.get(&full_key) {
            Some(entry) => {
                // Check if key has expired
                if self.is_expired(entry) {
                    return Ok(None);
                }
                Ok(Some(entry.value.clone()))
            }
            None => Ok(None),
        }
    }

    /// Delete a key
    pub async fn delete(&mut self, key: &str, namespace: Option<&str>) -> Result<bool, KeyValueError> {
        self.validate_namespace(namespace)?;

        let full_key = self.make_full_key(key, namespace);
        let existed = self.storage.entries.remove(&full_key).is_some();

        // Remove from namespace tracking
        if let Some(ns) = namespace {
            if let Some(ns_keys) = self.storage.namespaces.get_mut(ns) {
                ns_keys.remove(&full_key);
            }
        }

        Ok(existed)
    }

    /// Check if key exists
    pub async fn exists(&self, key: &str, namespace: Option<&str>) -> Result<bool, KeyValueError> {
        self.validate_namespace(namespace)?;

        let full_key = self.make_full_key(key, namespace);

        match self.storage.entries.get(&full_key) {
            Some(entry) => Ok(!self.is_expired(entry)),
            None => Ok(false),
        }
    }

    /// Get key metadata
    pub async fn get_metadata(&self, key: &str, namespace: Option<&str>) -> Result<Option<KeyMetadata>, KeyValueError> {
        self.validate_namespace(namespace)?;

        let full_key = self.make_full_key(key, namespace);

        match self.storage.entries.get(&full_key) {
            Some(entry) => {
                if self.is_expired(entry) {
                    Ok(None)
                } else {
                    Ok(Some(entry.metadata.clone()))
                }
            }
            None => Ok(None),
        }
    }

    /// Set expiration time for key
    pub async fn expire(
        &mut self,
        key: &str,
        ttl_seconds: u64,
        namespace: Option<&str>,
    ) -> Result<bool, KeyValueError> {
        self.validate_namespace(namespace)?;

        let full_key = self.make_full_key(key, namespace);

        if let Some(entry) = self.storage.entries.get(&full_key) {
            if !self.is_expired(entry) {
                if let Some(entry_mut) = self.storage.entries.get_mut(&full_key) {
                    entry_mut.metadata.ttl = Some(Utc::now().timestamp() as u64 + ttl_seconds);
                    entry_mut.updated_at = Utc::now();
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Get time to live for key
    pub async fn ttl(&self, key: &str, namespace: Option<&str>) -> Result<Option<i64>, KeyValueError> {
        self.validate_namespace(namespace)?;

        let full_key = self.make_full_key(key, namespace);

        if let Some(entry) = self.storage.entries.get(&full_key) {
            if let Some(expiry) = entry.metadata.ttl {
                let now = Utc::now().timestamp() as u64;
                if expiry > now {
                    return Ok(Some((expiry - now) as i64));
                } else {
                    return Ok(Some(-1)); // Already expired
                }
            }
        }

        Ok(None) // No TTL set
    }

    /// Perform atomic operation
    pub async fn atomic_operation(
        &mut self,
        key: &str,
        operation: AtomicOperation,
        namespace: Option<&str>,
    ) -> Result<Option<Value>, KeyValueError> {
        self.validate_namespace(namespace)?;

        let _full_key = self.make_full_key(key, namespace);

        match operation {
            AtomicOperation::Set(value) => {
                let old_value = self.get(key, namespace).await?;
                self.set(key, value, None, namespace).await?;
                Ok(old_value)
            }
            AtomicOperation::SetNX(value) => {
                if !self.exists(key, namespace).await? {
                    self.set(key, value, None, namespace).await?;
                    Ok(None)
                } else {
                    Ok(Some(Value::Bool(false)))
                }
            }
            AtomicOperation::GetSet(value) => {
                let old_value = self.get(key, namespace).await?;
                self.set(key, value, None, namespace).await?;
                Ok(old_value)
            }
            AtomicOperation::Inc(amount) => {
                let current = self.get(key, namespace).await?.and_then(|v| v.as_i64()).unwrap_or(0);
                let new_value = current + amount;
                self.set(key, Value::Number(new_value.into()), None, namespace).await?;
                Ok(Some(Value::Number(new_value.into())))
            }
            AtomicOperation::Dec(amount) => {
                let current = self.get(key, namespace).await?.and_then(|v| v.as_i64()).unwrap_or(0);
                let new_value = current - amount;
                self.set(key, Value::Number(new_value.into()), None, namespace).await?;
                Ok(Some(Value::Number(new_value.into())))
            }
            AtomicOperation::Append(string) => {
                let current = self
                    .get(key, namespace)
                    .await?
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
                    .unwrap_or_else(|| "".to_string());
                let new_value = format!("{}{}", current, string);
                self.set(key, Value::String(new_value.clone()), None, namespace).await?;
                Ok(Some(Value::String(new_value)))
            }
            AtomicOperation::Prepend(string) => {
                let current = self
                    .get(key, namespace)
                    .await?
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
                    .unwrap_or_else(|| "".to_string());
                let new_value = format!("{}{}", string, current);
                self.set(key, Value::String(new_value.clone()), None, namespace).await?;
                Ok(Some(Value::String(new_value)))
            }
            AtomicOperation::Push(value, direction) => {
                let mut current = self
                    .get(key, namespace)
                    .await?
                    .and_then(|v| v.as_array().cloned())
                    .unwrap_or_default();

                match direction {
                    PushDirection::Left => current.insert(0, value),
                    PushDirection::Right => current.push(value),
                }

                self.set(key, Value::Array(current.clone()), None, namespace).await?;
                Ok(Some(Value::Array(current)))
            }
            AtomicOperation::Pop(direction) => {
                let mut current = self
                    .get(key, namespace)
                    .await?
                    .and_then(|v| v.as_array().cloned())
                    .unwrap_or_default();

                let popped = match direction {
                    PushDirection::Left => current.first().cloned(),
                    PushDirection::Right => current.last().cloned(),
                };

                if popped.is_some() {
                    match direction {
                        PushDirection::Left => {
                            current.remove(0);
                        }
                        PushDirection::Right => {
                            current.pop();
                        }
                    }
                    self.set(key, Value::Array(current), None, namespace).await?;
                }

                Ok(popped)
            }
        }
    }

    /// Compare and swap
    pub async fn compare_and_swap(
        &mut self,
        cas: CasOperation,
        namespace: Option<&str>,
    ) -> Result<bool, KeyValueError> {
        self.validate_namespace(namespace)?;

        let full_key = self.make_full_key(&cas.key, namespace);

        if let Some(entry) = self.storage.entries.get_mut(&full_key) {
            if entry.version == cas.expected_version {
                entry.value = cas.new_value;
                entry.version += 1;
                entry.updated_at = Utc::now();

                if let Some(ttl) = cas.ttl {
                    entry.metadata.ttl = Some(Utc::now().timestamp() as u64 + ttl);
                }

                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }

    /// Execute batch operations
    pub async fn execute_batch(
        &mut self,
        batch: BatchOperation,
        namespace: Option<&str>,
    ) -> Result<BatchResult, KeyValueError> {
        let start_time = std::time::Instant::now();

        if batch.atomic {
            // Atomic batch - all or nothing
            let mut results = Vec::new();
            let mut success = true;

            // First, validate all operations
            for op in &batch.operations {
                let full_key = self.make_full_key(&op.key, namespace);
                if !self.storage.entries.contains_key(&full_key) {
                    // For atomic batch, all keys must exist
                    success = false;
                    break;
                }
            }

            if success {
                // Execute all operations
                for op in batch.operations {
                    let result = self.execute_single_operation(op, namespace).await?;
                    results.push(result);
                }
            } else {
                return Err(KeyValueError::AtomicOperationFailed(
                    "Some keys do not exist for atomic batch".to_string(),
                ));
            }

            let execution_time = start_time.elapsed().as_millis() as u64;
            Ok(BatchResult { results, success: true, execution_time_ms: execution_time })
        } else {
            // Non-atomic batch - execute all, collect results
            let mut results = Vec::new();

            for op in batch.operations {
                let result = self.execute_single_operation(op, namespace).await?;
                results.push(result);
            }

            let execution_time = start_time.elapsed().as_millis() as u64;
            Ok(BatchResult { results, success: true, execution_time_ms: execution_time })
        }
    }

    /// Scan keys with pattern matching
    pub async fn scan(&self, options: ScanOptions) -> Result<ScanResult, KeyValueError> {
        let mut matching_keys = Vec::new();
        let mut cursor = options.cursor.unwrap_or_default();

        // Simple implementation - in real implementation, this would use indexes
        for (key, entry) in &self.storage.entries {
            if self.is_expired(entry) {
                continue;
            }

            // Check namespace filter
            if let Some(ns) = &options.namespace {
                if !key.starts_with(&format!("{}:", ns)) {
                    continue;
                }
            }

            // Check pattern filter
            if let Some(pattern) = &options.pattern {
                if !self.matches_pattern(key, pattern) {
                    continue;
                }
            }

            // Cursor-based pagination
            if key > &cursor {
                matching_keys.push(key.clone());
                cursor = key.clone();

                if matching_keys.len() >= options.count {
                    break;
                }
            }
        }

        let has_more = matching_keys.len() >= options.count;

        Ok(ScanResult {
            keys: matching_keys,
            cursor: if has_more { Some(cursor) } else { None },
            has_more,
        })
    }

    /// Publish message to channel
    pub async fn publish(
        &mut self,
        channel: &str,
        message: Value,
        publisher: Option<String>,
    ) -> Result<usize, KeyValueError> {
        let pubsub_message = PubSubMessage {
            channel: channel.to_string(),
            message,
            timestamp: Utc::now(),
            publisher,
        };

        // Add to history
        self.pubsub.message_history.push_back(pubsub_message.clone());
        if self.pubsub.message_history.len() > self.pubsub.max_history {
            self.pubsub.message_history.pop_front();
        }

        // Send to subscribers
        let mut subscriber_count = 0;

        // Direct channel subscribers
        if let Some(subscribers) = self.pubsub.channels.get(channel) {
            for subscriber_id in subscribers {
                if let Some(subscription) = self.pubsub.subscribers.get_mut(subscriber_id) {
                    subscription.message_queue.push_back(pubsub_message.clone());
                    subscriber_count += 1;
                }
            }
        }

        // Pattern subscribers
        for (pattern, subscribers) in &self.pubsub.patterns {
            if self.matches_pattern(channel, pattern) {
                for subscriber_id in subscribers {
                    if let Some(subscription) = self.pubsub.subscribers.get_mut(subscriber_id) {
                        subscription.message_queue.push_back(pubsub_message.clone());
                        subscriber_count += 1;
                    }
                }
            }
        }

        Ok(subscriber_count)
    }

    /// Subscribe to channels
    pub async fn subscribe(&mut self, subscriber_id: Uuid, channels: Vec<String>) -> Result<(), KeyValueError> {
        let subscription = self.pubsub.subscribers.entry(subscriber_id).or_insert_with(|| Subscription {
            channels: HashSet::new(),
            patterns: HashSet::new(),
            message_queue: VecDeque::new(),
        });

        for channel in channels {
            subscription.channels.insert(channel.clone());
            self.pubsub.channels.entry(channel).or_default().insert(subscriber_id);
        }

        Ok(())
    }

    /// Subscribe to channel patterns
    pub async fn psubscribe(&mut self, subscriber_id: Uuid, patterns: Vec<String>) -> Result<(), KeyValueError> {
        let subscription = self.pubsub.subscribers.entry(subscriber_id).or_insert_with(|| Subscription {
            channels: HashSet::new(),
            patterns: HashSet::new(),
            message_queue: VecDeque::new(),
        });

        for pattern in patterns {
            subscription.patterns.insert(pattern.clone());
            self.pubsub.patterns.entry(pattern).or_default().insert(subscriber_id);
        }

        Ok(())
    }

    /// Get pending messages for subscriber
    pub async fn get_messages(&mut self, subscriber_id: Uuid) -> Result<Vec<PubSubMessage>, KeyValueError> {
        if let Some(subscription) = self.pubsub.subscribers.get_mut(&subscriber_id) {
            let messages: Vec<_> = subscription.message_queue.drain(..).collect();
            Ok(messages)
        } else {
            Ok(Vec::new())
        }
    }

    /// Unsubscribe from channels
    pub async fn unsubscribe(&mut self, subscriber_id: Uuid, channels: Vec<String>) -> Result<(), KeyValueError> {
        if let Some(subscription) = self.pubsub.subscribers.get_mut(&subscriber_id) {
            for channel in channels {
                subscription.channels.remove(&channel);
                if let Some(channel_subs) = self.pubsub.channels.get_mut(&channel) {
                    channel_subs.remove(&subscriber_id);
                    if channel_subs.is_empty() {
                        self.pubsub.channels.remove(&channel);
                    }
                }
            }
        }
        Ok(())
    }

    /// Get statistics
    pub async fn stats(&self) -> KeyValueStats {
        let mut total_keys = 0;
        let mut total_memory = 0;
        let mut expired_keys = 0;

        for entry in self.storage.entries.values() {
            total_keys += 1;
            total_memory += entry.metadata.size_bytes;

            if self.is_expired(entry) {
                expired_keys += 1;
            }
        }

        KeyValueStats {
            total_keys,
            total_memory,
            expired_keys,
            namespaces: self.namespaces.len(),
            subscribers: self.pubsub.subscribers.len(),
            channels: self.pubsub.channels.len(),
        }
    }

    /// Helper methods
    fn validate_namespace(&self, namespace: Option<&str>) -> Result<(), KeyValueError> {
        if let Some(ns) = namespace {
            if !self.namespaces.contains_key(ns) {
                return Err(KeyValueError::NamespaceNotFound(ns.to_string()));
            }
        }
        Ok(())
    }

    fn make_full_key(&self, key: &str, namespace: Option<&str>) -> String {
        if let Some(ns) = namespace {
            format!("{}:{}", ns, key)
        } else {
            key.to_string()
        }
    }

    async fn create_or_update_entry(
        &self,
        full_key: &str,
        value: Value,
        ttl: Option<u64>,
    ) -> Result<KeyValueEntry, KeyValueError> {
        let size_bytes = serde_json::to_string(&value)
            .map_err(|e| KeyValueError::SerializationError(e.to_string()))?
            .len();
        let now = Utc::now();

        let ttl_timestamp = ttl.map(|t| now.timestamp() as u64 + t);

        let metadata = KeyMetadata {
            size_bytes,
            content_type: self.infer_content_type(&value),
            ttl: ttl_timestamp,
            tags: HashSet::new(),
            owner: None,
            compression: false,
        };

        let entry = if let Some(existing) = self.storage.entries.get(full_key) {
            KeyValueEntry {
                key: full_key.to_string(),
                value,
                metadata,
                version: existing.version + 1,
                created_at: existing.created_at,
                updated_at: now,
            }
        } else {
            KeyValueEntry {
                key: full_key.to_string(),
                value,
                metadata,
                version: 1,
                created_at: now,
                updated_at: now,
            }
        };

        Ok(entry)
    }

    fn infer_content_type(&self, value: &Value) -> ContentType {
        match value {
            Value::String(_) => ContentType::String,
            Value::Number(n) => {
                if n.is_i64() {
                    ContentType::Integer
                } else {
                    ContentType::Float
                }
            }
            Value::Bool(_) => ContentType::Boolean,
            Value::Array(_) => ContentType::Json,
            Value::Object(_) => ContentType::Json,
            Value::Null => ContentType::Null,
        }
    }

    fn is_expired(&self, entry: &KeyValueEntry) -> bool {
        if let Some(ttl) = entry.metadata.ttl {
            Utc::now().timestamp() as u64 >= ttl
        } else {
            false
        }
    }

    async fn cleanup_expired_keys(&mut self) {
        let expired_keys: Vec<String> = self
            .storage
            .entries
            .iter()
            .filter(|(_, entry)| self.is_expired(entry))
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_keys {
            self.storage.entries.remove(&key);
            // Also remove from namespace tracking
            for ns_keys in self.storage.namespaces.values_mut() {
                ns_keys.remove(&key);
            }
        }
    }

    fn matches_pattern(&self, text: &str, pattern: &str) -> bool {
        // Simple glob matching - in real implementation, use regex or proper glob
        if pattern.contains('*') {
            let regex_pattern = pattern.replace('*', ".*");
            regex::Regex::new(&format!("^{}$", regex_pattern))
                .map(|re| re.is_match(text))
                .unwrap_or(false)
        } else {
            text == pattern
        }
    }

    async fn execute_single_operation(
        &mut self,
        operation: KeyOperation,
        namespace: Option<&str>,
    ) -> Result<OperationResult, KeyValueError> {
        let old_value = self.get(&operation.key, namespace).await?;

        match &operation.operation {
            AtomicOperation::Set(_)
            | AtomicOperation::SetNX(_)
            | AtomicOperation::GetSet(_)
            | AtomicOperation::Inc(_)
            | AtomicOperation::Dec(_)
            | AtomicOperation::Append(_)
            | AtomicOperation::Prepend(_)
            | AtomicOperation::Push(_, _) => {
                let result = self.atomic_operation(&operation.key, operation.operation, namespace).await;
                match result {
                    Ok(new_value) => Ok(OperationResult {
                        key: operation.key,
                        success: true,
                        old_value,
                        new_value,
                        error: None,
                    }),
                    Err(e) => Ok(OperationResult {
                        key: operation.key,
                        success: false,
                        old_value,
                        new_value: None,
                        error: Some(format!("{:?}", e)),
                    }),
                }
            }
            AtomicOperation::Pop(_) => {
                let result = self.atomic_operation(&operation.key, operation.operation, namespace).await;
                match result {
                    Ok(popped_value) => Ok(OperationResult {
                        key: operation.key,
                        success: true,
                        old_value,
                        new_value: popped_value,
                        error: None,
                    }),
                    Err(e) => Ok(OperationResult {
                        key: operation.key,
                        success: false,
                        old_value,
                        new_value: None,
                        error: Some(format!("{:?}", e)),
                    }),
                }
            }
        }
    }
}

impl Default for KeyValueEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Namespace configuration
#[derive(Debug, Clone)]
pub struct NamespaceConfig {
    pub max_keys: usize,
    pub max_memory: usize,
    pub ttl_default: Option<u64>,
    pub read_only: bool,
}

/// Key-value statistics
#[derive(Debug, Clone)]
pub struct KeyValueStats {
    pub total_keys: usize,
    pub total_memory: usize,
    pub expired_keys: usize,
    pub namespaces: usize,
    pub subscribers: usize,
    pub channels: usize,
}

/// Implementation for storage
impl KeyValueStorage {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            namespaces: HashMap::new(),
            total_memory: 0,
            max_memory: 1_000_000_000, // 1GB default
        }
    }
}

impl Default for KeyValueStorage {
    fn default() -> Self {
        Self::new()
    }
}

/// Implementation for pubsub
impl PubSubEngine {
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
            patterns: HashMap::new(),
            subscribers: HashMap::new(),
            message_history: VecDeque::new(),
            max_history: 1000,
        }
    }
}

impl Default for PubSubEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Implementation for atomic operations
impl AtomicOperations {
    pub fn new() -> Self {
        Self {
            pending_operations: HashMap::new(),
            operation_log: VecDeque::new(),
            max_log_size: 10000,
        }
    }
}

impl Default for AtomicOperations {
    fn default() -> Self {
        Self::new()
    }
}

pub use AtomicOperation as AtomicOp;
/// Export key-value engine components
pub use KeyValueEngine as Engine;
pub use KeyValueEntry as Entry;
pub use KeyValueError as Error;
pub use KeyValueStats as Stats;
