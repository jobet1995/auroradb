use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json;
use std::{
    collections::{BTreeMap, HashMap},
    sync::atomic::AtomicBool,
    time::Duration,
};
use uuid::Uuid;

/// Database version and metadata constants
pub const DATABASE_VERSION: &str = "0.1.0";
pub const DATABASE_NAME: &str = "AuroraDB";
pub const COMPATIBILITY_VERSION: u32 = 1;

/// Default configuration values
pub const DEFAULT_MAX_CONNECTIONS: usize = 1000;
pub const DEFAULT_CACHE_SIZE_MB: usize = 512;
pub const DEFAULT_TEMPORAL_RETENTION_DAYS: u32 = 365;
pub const DEFAULT_QUERY_TIMEOUT_SECS: u64 = 30;
pub const DEFAULT_TRANSACTION_TIMEOUT_SECS: u64 = 300;

/// Storage and performance constants
pub const MAX_TABLE_NAME_LENGTH: usize = 63;
pub const MAX_COLUMN_NAME_LENGTH: usize = 63;
pub const MAX_DOCUMENT_SIZE_BYTES: usize = 16 * 1024 * 1024; // 16MB
pub const DEFAULT_PAGE_SIZE: usize = 4096;
pub const MAX_BATCH_SIZE: usize = 10000;

/// Temporal database constants
pub const TEMPORAL_HISTORY_TABLE_SUFFIX: &str = "_history";
pub const TEMPORAL_VALID_FROM_COLUMN: &str = "valid_from";
pub const TEMPORAL_VALID_TO_COLUMN: &str = "valid_to";

/// Common type aliases for database operations
pub type DatabaseId = u64;
pub type TableId = u64;
pub type ColumnId = u32;
pub type RowId = u64;
pub type TransactionId = u64;
pub type QueryId = u64;
pub type Timestamp = DateTime<Utc>;

/// Engine runtime variables
pub const AURORA_VERSION: &str = "0.1.0";

/// Relational database type aliases
pub type SchemaDef = Vec<ColumnDefinition>;
pub type KeyDef = Vec<ColumnId>;
pub type IndexName = String;
pub type Index = BTreeMap<String, Vec<RowId>>;
pub type Table = String;
pub type Relation = Vec<(ColumnId, ColumnId)>;

/// Document/NoSQL type aliases
pub type Document = serde_json::Value;

/// Memory management type aliases
pub type BumpArena = (); // Placeholder - would need bumpalo crate
pub type DropGuard = (); // Placeholder - custom implementation needed
pub type Bytes = Vec<u8>; // Using Vec<u8> as Bytes placeholder

/// Transaction type aliases
pub type TxId = u128;

/// Authorization type aliases
pub type BitFlags = u64; // Using u64 for bit flags - could be replaced with bitflags crate
pub type Predicate = String; // SQL predicate string for row-level security

/// Security type aliases
pub type KeyMaterial = Vec<u8>; // Raw key bytes for encryption

/// Query optimization type aliases
pub type ExecutionPlan = Vec<String>; // Sequence of execution steps
pub type CostMetrics = f64; // Query cost estimation
pub type ExprTree = String; // Expression tree representation

/// View type aliases
pub type QueryAST = String; // Abstract syntax tree representation of query

/// Function type aliases
pub type Type = DataType; // Function parameter/return types using DataType

/// Column definition for table schemas
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ColumnDefinition {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub default_value: Option<String>,
}

/// User authentication context for permissions and session management
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UserContext {
    pub user_id: Uuid,
    pub username: String,
    pub roles: Vec<String>,
    pub permissions: Vec<String>,
    pub tenant_id: Option<Uuid>,
    pub session_expires_at: Option<u64>,
}

// Global runtime variables - initialized at startup
lazy_static::lazy_static! {
    /// Engine boot time in milliseconds since epoch
    pub static ref START_TIMESTAMP: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    /// Unique database instance identifier
    pub static ref INSTANCE_ID: Uuid = Uuid::new_v4();

    /// Graceful termination flag
    pub static ref SHUTDOWN_SIGNAL: AtomicBool = AtomicBool::new(false);

    /// Prevents Rc/Arc reference count leaks
    pub static ref REFCOUNT_LIMIT: usize = 1_000_000;

    /// Unique identifier for ownership boundary
    pub static ref LIFETIME_SCOPE_ID: Uuid = Uuid::new_v4();

    /// Current engine execution mode (set at runtime)
    pub static ref ENGINE_MODE: std::sync::RwLock<EngineMode> = std::sync::RwLock::new(EngineMode::Cli);

    /// Current node role in distributed setup (set at runtime)
    pub static ref NODE_ROLE: std::sync::RwLock<NodeRole> = std::sync::RwLock::new(NodeRole::Primary);

    /// Current table schema definitions (runtime managed)
    pub static ref TABLE_SCHEMA: std::sync::RwLock<HashMap<TableId, SchemaDef>> = std::sync::RwLock::new(HashMap::new());

    /// Global row ID counter
    pub static ref ROW_ID: std::sync::RwLock<u128> = std::sync::RwLock::new(1);

    /// Primary key definitions (runtime managed)
    pub static ref PRIMARY_KEY: std::sync::RwLock<HashMap<TableId, KeyDef>> = std::sync::RwLock::new(HashMap::new());

    /// Foreign key constraint mappings
    pub static ref FOREIGN_KEY_MAP: std::sync::RwLock<HashMap<Table, Relation>> = std::sync::RwLock::new(HashMap::new());

    /// Index mappings for fast lookups
    pub static ref INDEX_MAP: std::sync::RwLock<BTreeMap<IndexName, Index>> = std::sync::RwLock::new(BTreeMap::new());

    /// Current temporal query timestamp
    pub static ref AS_OF_TIMESTAMP: std::sync::RwLock<Option<u64>> = std::sync::RwLock::new(None);

    /// Current document ID for NoSQL operations
    pub static ref DOCUMENT_ID: std::sync::RwLock<Option<Uuid>> = std::sync::RwLock::new(None);

    /// Current collection name for document operations
    pub static ref COLLECTION_NAME: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

    /// Schema-optional flag for flexible document storage
    pub static ref SCHEMA_OPTIONAL: std::sync::RwLock<bool> = std::sync::RwLock::new(false);

    /// Current key-value namespace
    pub static ref KV_NAMESPACE: std::sync::RwLock<String> = std::sync::RwLock::new("default".to_string());

    /// Embedded documents for nested operations
    pub static ref EMBEDDED_DOCS: std::sync::RwLock<Vec<Document>> = std::sync::RwLock::new(Vec::new());

    /// Arena allocator for scoped memory management
    pub static ref ARENA_ALLOCATOR: std::sync::RwLock<BumpArena> = std::sync::RwLock::new(());

    /// Memory guard for enforced cleanup
    pub static ref MEMORY_GUARD: std::sync::RwLock<DropGuard> = std::sync::RwLock::new(());

    /// Zero-copy buffer for efficient data transfer
    pub static ref ZERO_COPY_BUFFER: std::sync::RwLock<Bytes> = std::sync::RwLock::new(Vec::new());

    /// Current transaction identifier
    pub static ref TX_ID: std::sync::RwLock<TxId> = std::sync::RwLock::new(1);

    /// Current transaction state
    pub static ref TX_STATE: std::sync::RwLock<TransactionState> = std::sync::RwLock::new(TransactionState::Committed);

    /// Current isolation level for transactions
    pub static ref ISOLATION_LEVEL: std::sync::RwLock<TransactionIsolationLevel> = std::sync::RwLock::new(TransactionIsolationLevel::ReadCommitted);

    /// Current MVCC snapshot version
    pub static ref MVCC_VERSION: std::sync::RwLock<u64> = std::sync::RwLock::new(1);

    /// Current lock mode for concurrency control
    pub static ref LOCK_MODE: std::sync::RwLock<LockMode> = std::sync::RwLock::new(LockMode::Shared);

    /// JWT secret key for token signing
    pub static ref JWT_SECRET: std::sync::RwLock<String> = std::sync::RwLock::new("your-secret-key-change-in-production".to_string());

    /// JWT token issuer identifier
    pub static ref JWT_ISSUER: std::sync::RwLock<String> = std::sync::RwLock::new("auroradb".to_string());

    /// JWT token expiration time in seconds
    pub static ref JWT_EXPIRY: std::sync::RwLock<u64> = std::sync::RwLock::new(3600); // 1 hour

    /// Current authentication session identifier
    pub static ref SESSION_ID: std::sync::RwLock<Option<Uuid>> = std::sync::RwLock::new(None);

    /// Current user authentication context
    pub static ref AUTH_CONTEXT: std::sync::RwLock<Option<UserContext>> = std::sync::RwLock::new(None);

    /// Current user role identifier
    pub static ref ROLE_ID: std::sync::RwLock<Option<Uuid>> = std::sync::RwLock::new(None);

    /// Current permission bit flags for access control
    pub static ref PERMISSION_SET: std::sync::RwLock<BitFlags> = std::sync::RwLock::new(0);

    /// Current tenant identifier for multi-tenant isolation
    pub static ref TENANT_ID: std::sync::RwLock<Option<Uuid>> = std::sync::RwLock::new(None);

    /// Current row-level security filter predicate
    pub static ref ROW_LEVEL_FILTER: std::sync::RwLock<Option<Predicate>> = std::sync::RwLock::new(None);

    /// Data encryption key for at-rest encryption
    pub static ref DATA_ENCRYPTION_KEY: std::sync::RwLock<KeyMaterial> = std::sync::RwLock::new(vec![0u8; 32]); // 256-bit key placeholder

    /// TLS encryption enabled for in-transit security
    pub static ref TLS_ENABLED: std::sync::RwLock<bool> = std::sync::RwLock::new(false);

    /// Key rotation epoch timestamp for security hygiene
    pub static ref KEY_ROTATION_EPOCH: std::sync::RwLock<u64> = std::sync::RwLock::new(0);

    /// Current query execution plan
    pub static ref QUERY_PLAN: std::sync::RwLock<Option<ExecutionPlan>> = std::sync::RwLock::new(None);

    /// Current query cost model metrics
    pub static ref COST_MODEL: std::sync::RwLock<CostMetrics> = std::sync::RwLock::new(0.0);

    /// Current filter predicate expression tree
    pub static ref FILTER_PREDICATE: std::sync::RwLock<Option<ExprTree>> = std::sync::RwLock::new(None);

    /// Current join strategy for query optimization
    pub static ref JOIN_STRATEGY: std::sync::RwLock<JoinStrategy> = std::sync::RwLock::new(JoinStrategy::Hash);

    /// Current view name for view operations
    pub static ref VIEW_NAME: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

    /// Current view query AST representation
    pub static ref VIEW_QUERY: std::sync::RwLock<Option<QueryAST>> = std::sync::RwLock::new(None);

    /// Whether current view is materialized
    pub static ref MATERIALIZED: std::sync::RwLock<bool> = std::sync::RwLock::new(false);

    /// Current function name for function operations
    pub static ref FUNCTION_NAME: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

    /// Current function argument types
    pub static ref FUNCTION_ARGS: std::sync::RwLock<Vec<Type>> = std::sync::RwLock::new(Vec::new());

    /// Current function return type
    pub static ref RETURN_TYPE: std::sync::RwLock<Option<Type>> = std::sync::RwLock::new(None);

    /// Whether current function is immutable (side-effect free)
    pub static ref IMMUTABLE: std::sync::RwLock<bool> = std::sync::RwLock::new(true);

    /// Current CLI command being executed
    pub static ref CLI_COMMAND: std::sync::RwLock<Option<CliCommand>> = std::sync::RwLock::new(None);

    /// Current CLI command flags and arguments
    pub static ref CLI_FLAGS: std::sync::RwLock<HashMap<String, String>> = std::sync::RwLock::new(HashMap::new());

    /// Current output format for CLI results
    pub static ref STDOUT_FORMAT: std::sync::RwLock<StdoutFormat> = std::sync::RwLock::new(StdoutFormat::Table);

    /// Whether CLI is in interactive mode
    pub static ref INTERACTIVE_MODE: std::sync::RwLock<bool> = std::sync::RwLock::new(false);

    /// HTTP server port for web interface
    pub static ref HTTP_PORT: std::sync::RwLock<u16> = std::sync::RwLock::new(8080);

    /// API version for web interface
    pub static ref API_VERSION: std::sync::RwLock<String> = std::sync::RwLock::new("v1".to_string());

    /// Session storage backend for web interface
    pub static ref SESSION_STORE: std::sync::RwLock<SessionStore> = std::sync::RwLock::new(SessionStore::Memory);

    /// UI adapter for frontend bridge (trait object)
    pub static ref UI_ADAPTER: std::sync::RwLock<Option<Box<dyn UiAdapter + Send + Sync>>> = std::sync::RwLock::new(None);

    /// Current logging level for system observability
    pub static ref LOG_LEVEL: std::sync::RwLock<LogLevel> = std::sync::RwLock::new(LogLevel::Info);

    /// Current trace identifier for distributed tracing
    pub static ref TRACE_ID: std::sync::RwLock<Option<Uuid>> = std::sync::RwLock::new(None);

    /// Whether metrics collection is enabled
    pub static ref METRICS_ENABLED: std::sync::RwLock<bool> = std::sync::RwLock::new(true);

    /// Current system health status for monitoring
    pub static ref HEALTH_STATUS: std::sync::RwLock<HealthStatus> = std::sync::RwLock::new(HealthStatus::Healthy);

    /// Current temporal query timestamp (AS_OF)
    pub static ref TEMPORAL_TIMESTAMP: std::sync::RwLock<Option<u64>> = std::sync::RwLock::new(None);

    /// Current document operation mode
    pub static ref DOCUMENT_MODE: std::sync::RwLock<bool> = std::sync::RwLock::new(false);

    /// Current streaming window configuration
    pub static ref STREAM_WINDOW: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

    /// Current ML model for PREDICT operations
    pub static ref ACTIVE_ML_MODEL: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

    /// Current tenant context for multi-tenant operations
    pub static ref TENANT_CONTEXT: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

    /// Query profiling enabled flag
    pub static ref QUERY_PROFILING: std::sync::RwLock<bool> = std::sync::RwLock::new(false);

    /// Change stream enabled for real-time operations
    pub static ref CHANGE_STREAM_ENABLED: std::sync::RwLock<bool> = std::sync::RwLock::new(false);

    /// Graph traversal depth limit
    pub static ref GRAPH_DEPTH_LIMIT: std::sync::RwLock<usize> = std::sync::RwLock::new(10);

    /// Hybrid query mode (relational + document)
    pub static ref HYBRID_MODE: std::sync::RwLock<bool> = std::sync::RwLock::new(false);

    /// Current streaming window type
    pub static ref STREAM_WINDOW_TYPE: std::sync::RwLock<Option<StreamWindowType>> = std::sync::RwLock::new(None);

    /// Current temporal operation mode
    pub static ref TEMPORAL_OPERATION: std::sync::RwLock<TemporalQueryMode> = std::sync::RwLock::new(TemporalQueryMode::Latest);

    /// Current graph algorithm for traversals
    pub static ref GRAPH_ALGORITHM: std::sync::RwLock<GraphAlgorithm> = std::sync::RwLock::new(GraphAlgorithm::BreadthFirst);

    /// Current ML model type for AI operations
    pub static ref CURRENT_MODEL_TYPE: std::sync::RwLock<Option<ModelType>> = std::sync::RwLock::new(None);

    /// Real-time processing enabled
    pub static ref REAL_TIME_ENABLED: std::sync::RwLock<bool> = std::sync::RwLock::new(false);

    /// Document flattening depth for nested operations
    pub static ref DOCUMENT_FLATTEN_DEPTH: std::sync::RwLock<usize> = std::sync::RwLock::new(3);

    /// Schema isolation mode for multi-tenant operations
    pub static ref SCHEMA_ISOLATION: std::sync::RwLock<bool> = std::sync::RwLock::new(true);

    /// Quantum-resistant encryption algorithm
    pub static ref QUANTUM_ALGORITHM: std::sync::RwLock<QuantumAlgorithm> = std::sync::RwLock::new(QuantumAlgorithm::CrystalsKyber);

    /// Blockchain consensus type for immutable operations
    pub static ref CONSENSUS_TYPE: std::sync::RwLock<ConsensusType> = std::sync::RwLock::new(ConsensusType::ProofOfStake);

    /// Edge computing execution mode
    pub static ref EDGE_MODE: std::sync::RwLock<EdgeMode> = std::sync::RwLock::new(EdgeMode::CloudFallback);

    /// Natural language processing model
    pub static ref NLP_MODEL: std::sync::RwLock<NlpModel> = std::sync::RwLock::new(NlpModel::Bert);

    /// Data governance compliance level
    pub static ref COMPLIANCE_LEVEL: std::sync::RwLock<ComplianceLevel> = std::sync::RwLock::new(ComplianceLevel::Basic);

    /// Forecasting algorithm for predictive analytics
    pub static ref FORECASTING_ALGORITHM: std::sync::RwLock<ForecastingAlgorithm> = std::sync::RwLock::new(ForecastingAlgorithm::Arima);

    /// Conflict resolution strategy for collaboration
    pub static ref CONFLICT_STRATEGY: std::sync::RwLock<ConflictStrategy> = std::sync::RwLock::new(ConflictStrategy::LastWriteWins);

    /// Spatial-temporal indexing method
    pub static ref SPATIAL_TEMPORAL_INDEX: std::sync::RwLock<SpatialTemporalIndex> = std::sync::RwLock::new(SpatialTemporalIndex::RTree);

    /// Federated learning aggregation method
    pub static ref AGGREGATION_METHOD: std::sync::RwLock<AggregationMethod> = std::sync::RwLock::new(AggregationMethod::FedAvg);

    /// Auto-scaling strategy
    pub static ref SCALING_STRATEGY: std::sync::RwLock<ScalingStrategy> = std::sync::RwLock::new(ScalingStrategy::Reactive);

    /// Holographic data dimension
    pub static ref HOLOGRAPHIC_DIMENSION: std::sync::RwLock<HolographicDimension> = std::sync::RwLock::new(HolographicDimension::ThreeD);

    /// Neural network architecture for database operations
    pub static ref NEURAL_ARCHITECTURE: std::sync::RwLock<NeuralArchitecture> = std::sync::RwLock::new(NeuralArchitecture::Transformer);

    /// Quantum computing qubit state
    pub static ref QUBIT_STATE: std::sync::RwLock<QubitState> = std::sync::RwLock::new(QubitState::Superposition);

    /// Bio-inspired optimization algorithm
    pub static ref BIO_ALGORITHM: std::sync::RwLock<BioAlgorithm> = std::sync::RwLock::new(BioAlgorithm::Genetic);

    /// System consciousness level
    pub static ref CONSCIOUSNESS_LEVEL: std::sync::RwLock<ConsciousnessLevel> = std::sync::RwLock::new(ConsciousnessLevel::Adaptive);

    /// Multi-universe data realm
    pub static ref UNIVERSE_REALM: std::sync::RwLock<UniverseRealm> = std::sync::RwLock::new(UniverseRealm::Primary);

    /// Time crystal computation phase
    pub static ref TIME_CRYSTAL_PHASE: std::sync::RwLock<TimeCrystalPhase> = std::sync::RwLock::new(TimeCrystalPhase::Periodic);
}

/// Data type enumeration for schema definitions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    Boolean,
    Integer,
    BigInt,
    Float,
    Double,
    Decimal(u8, u8), // precision, scale
    String(usize),   // max length
    Text,
    Binary(usize), // max size
    Date,
    Time,
    DateTime,
    Timestamp,
    Uuid,
    Json,
    Array(Box<DataType>),
}

/// Query operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryType {
    Select,
    Insert,
    Update,
    Delete,
    Create,
    Drop,
    Alter,
    Grant,
    Revoke,
    Begin,
    Commit,
    Rollback,
}

/// Transaction isolation levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

/// Storage engine types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageEngine {
    Sled,
    Memory,
    RocksDb,
}

/// Index types supported by the database
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexType {
    BTree,
    Hash,
    FullText,
    Spatial,
    Temporal,
}

/// Runtime execution mode for the database engine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineMode {
    Cli,
    Gui,
    Web,
}

/// Distributed node roles in a cluster
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeRole {
    Primary,
    Replica,
}

/// Transaction states for concurrency control
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransactionState {
    Pending,
    Committed,
    RolledBack,
}

/// Isolation levels for transaction consistency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransactionIsolationLevel {
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

/// Lock modes for concurrency control
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LockMode {
    Shared,
    Exclusive,
}

/// Join strategies for query optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinStrategy {
    Hash,
    Merge,
    Nested,
}

/// CLI command types for user interface
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CliCommand {
    Connect,
    Query,
    Schema,
    Backup,
    Restore,
    Status,
    Help,
    Exit,
}

/// Output format options for CLI results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StdoutFormat {
    Table,
    Json,
}

/// Session storage backends for web interface
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStore {
    Memory,
    Redis,
}

/// Logging levels for system observability
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// System health status for monitoring
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    Healthy,
    Degraded,
}

/// Streaming window types for real-time operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamWindowType {
    Tumbling,
    Sliding,
    Hopping,
}

/// Temporal query operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalQueryMode {
    AsOf,
    Between,
    AllVersions,
    Latest,
}

/// Graph traversal algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphAlgorithm {
    BreadthFirst,
    DepthFirst,
    ShortestPath,
    AllPaths,
}

/// ML model types for AI operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    Classification,
    Regression,
    Clustering,
    AnomalyDetection,
}

/// Quantum-resistant cryptographic algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantumAlgorithm {
    CrystalsKyber,
    Dilithium,
    Falcon,
    Sphincs,
}

/// Blockchain consensus mechanisms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsensusType {
    ProofOfWork,
    ProofOfStake,
    ProofOfAuthority,
    ByzantineFault,
}

/// Edge computing execution modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeMode {
    LocalOnly,
    EdgeFirst,
    CloudFallback,
    Distributed,
}

/// Natural language processing models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NlpModel {
    Bert,
    Gpt,
    Transformer,
    Custom,
}

/// Data governance compliance levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComplianceLevel {
    Basic,
    Gdpr,
    Hipaa,
    Soc2,
    Custom,
}

/// Forecasting algorithms for predictive analytics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForecastingAlgorithm {
    Arima,
    Prophet,
    Lstm,
    ExponentialSmoothing,
}

/// Conflict resolution strategies for collaboration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictStrategy {
    LastWriteWins,
    MergePriority,
    ManualResolve,
    VersionVector,
}

/// Spatial-temporal indexing methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialTemporalIndex {
    RTree,
    QuadTree,
    HilbertCurve,
    ZOrder,
}

/// Federated learning aggregation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregationMethod {
    FedAvg,
    FedProx,
    Scaffold,
    Ditto,
}

/// Auto-scaling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalingStrategy {
    Horizontal,
    Vertical,
    Predictive,
    Reactive,
}

/// Holographic data dimensions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HolographicDimension {
    TwoD,
    ThreeD,
    FourD,
    MultiD,
}

/// Neural network architectures for database operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuralArchitecture {
    FeedForward,
    Convolutional,
    Recurrent,
    Transformer,
    GraphNeural,
}

/// Quantum computing qubit states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QubitState {
    Zero,
    One,
    Superposition,
    Entangled,
}

/// Bio-inspired optimization algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BioAlgorithm {
    Genetic,
    ParticleSwarm,
    AntColony,
    BeeColony,
    CuckooSearch,
}

/// System consciousness levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsciousnessLevel {
    Reactive,
    Adaptive,
    Predictive,
    SelfAware,
    Conscious,
}

/// Multi-universe data realms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UniverseRealm {
    Primary,
    Shadow,
    Parallel,
    Quantum,
    Temporal,
}

/// Time crystal computation phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeCrystalPhase {
    Discrete,
    Continuous,
    Periodic,
    Chaotic,
    Fractal,
}

/// UI adapter trait for frontend bridge
pub trait UiAdapter {
    fn render(&self, data: &serde_json::Value) -> String;
    fn handle_event(&mut self, event: &str) -> Result<(), Box<dyn std::error::Error>>;
}

/// Database configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub name: String,
    pub max_connections: usize,
    pub cache_size_mb: usize,
    pub storage_engine: StorageEngine,
    pub temporal_enabled: bool,
    pub temporal_retention_days: u32,
    pub query_timeout: Duration,
    pub transaction_timeout: Duration,
    pub data_directory: Option<String>,
    pub enable_authentication: bool,
    pub enable_ssl: bool,
    pub max_concurrent_transactions: usize,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            name: "auroradb".to_string(),
            max_connections: DEFAULT_MAX_CONNECTIONS,
            cache_size_mb: DEFAULT_CACHE_SIZE_MB,
            storage_engine: StorageEngine::Sled,
            temporal_enabled: true,
            temporal_retention_days: DEFAULT_TEMPORAL_RETENTION_DAYS,
            query_timeout: Duration::from_secs(DEFAULT_QUERY_TIMEOUT_SECS),
            transaction_timeout: Duration::from_secs(DEFAULT_TRANSACTION_TIMEOUT_SECS),
            data_directory: None,
            enable_authentication: true,
            enable_ssl: false,
            max_concurrent_transactions: 100,
        }
    }
}

/// Connection configuration for client connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub ssl_mode: SslMode,
    pub connection_timeout: Duration,
    pub keep_alive: bool,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 5432,
            database: "auroradb".to_string(),
            username: None,
            password: None,
            ssl_mode: SslMode::Prefer,
            connection_timeout: Duration::from_secs(30),
            keep_alive: true,
        }
    }
}

/// SSL connection modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SslMode {
    Disable,
    Allow,
    Prefer,
    Require,
    VerifyCa,
    VerifyFull,
}

/// Temporal query types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalQuery {
    Current,
    AsOf(Timestamp),
    Between(Timestamp, Timestamp),
    All,
}

/// Error types for database operations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatabaseError {
    ConnectionFailed,
    AuthenticationFailed,
    PermissionDenied,
    TableNotFound,
    ColumnNotFound,
    ConstraintViolation,
    TransactionConflict,
    QueryTimeout,
    InvalidSyntax,
    TypeMismatch,
    SerializationError,
    StorageError,
    TemporalError,
}

/// Result type alias for database operations
pub type DatabaseResult<T> = Result<T, DatabaseError>;

/// Common SQL keywords and operators
pub const SQL_KEYWORDS: &[&str] = &[
    "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "TABLE", "INDEX", "VIEW",
    "TRIGGER", "FUNCTION", "PROCEDURE", "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "PRIMARY", "FOREIGN", "KEY",
    "UNIQUE", "NOT", "NULL", "DEFAULT", "CHECK", "REFERENCES", "CASCADE", "RESTRICT", "JOIN", "INNER", "LEFT", "RIGHT",
    "FULL", "OUTER", "ON", "GROUP", "BY", "HAVING", "ORDER", "LIMIT", "OFFSET", "UNION", "ALL", "DISTINCT", "AS",
    "AND", "OR", "IN", "EXISTS", "LIKE", "BETWEEN", "IS", "TRUE", "FALSE", "UNKNOWN",
];

/// AuroraDB-specific SQL keywords for advanced features
pub const AURORADB_KEYWORDS: &[&str] = &[
    // Temporal/Time-travel keywords
    "AS_OF",
    "VALID_FROM",
    "VALID_TO",
    "TEMPORAL",
    "TIME_TRAVEL",
    "VERSION_HISTORY",
    // Document/JSON keywords
    "DOCUMENT",
    "COLLECTION",
    "JSON_EXTRACT",
    "JSON_SET",
    "JSON_ARRAY",
    "JSON_OBJECT",
    "UNNEST",
    "FLATTEN",
    "NEST",
    // Graph/Traversal keywords
    "GRAPH",
    "NODE",
    "EDGE",
    "PATH",
    "SHORTEST_PATH",
    "TRAVERSE",
    "CONNECTED_BY",
    // Real-time/Streaming keywords
    "STREAM",
    "WINDOW",
    "TUMBLING",
    "SLIDING",
    "HOPPING",
    "EMIT",
    "CHANGE_STREAM",
    // Machine Learning/AI keywords
    "PREDICT",
    "TRAIN",
    "MODEL",
    "FEATURE",
    "CLUSTER",
    "CLASSIFY",
    "ANOMALY_DETECT",
    // Security/Access Control keywords
    "GRANT_TENANT",
    "REVOKE_TENANT",
    "ROW_POLICY",
    "MASK",
    "ENCRYPT",
    "DECRYPT",
    // Multi-tenant keywords
    "TENANT",
    "ISOLATE",
    "SHARE_SCHEMA",
    "TENANT_CONTEXT",
    // Performance/Monitoring keywords
    "ANALYZE_QUERY",
    "EXPLAIN_PLAN",
    "PROFILE",
    "TRACE_QUERY",
    "METRICS",
    // Hybrid operations
    "RELATIONAL_TO_DOCUMENT",
    "DOCUMENT_TO_RELATIONAL",
    "HYBRID_JOIN",
    // Quantum-resistant cryptography
    "QUANTUM_SAFE",
    "POST_QUANTUM",
    "CRYSTALS_KYBER",
    "DILITHIUM",
    "FALCON",
    // Blockchain integration
    "BLOCKCHAIN",
    "LEDGER",
    "IMMUTABLE_CHAIN",
    "PROOF_OF_WORK",
    "SMART_CONTRACT",
    // Edge computing
    "EDGE_COMPUTE",
    "FOG_PROCESSING",
    "DISTRIBUTED_EXEC",
    "LOCAL_CACHE",
    "EDGE_SYNC",
    // Natural language processing
    "SEMANTIC_SEARCH",
    "NLP_QUERY",
    "TEXT_ANALYZE",
    "LANGUAGE_MODEL",
    "VECTOR_SEARCH",
    // Data governance and lineage
    "DATA_LINEAGE",
    "GOVERNANCE",
    "COMPLIANCE_CHECK",
    "AUDIT_TRAIL",
    "DATA_STEWARD",
    // Predictive analytics
    "FORECAST",
    "TREND_ANALYZE",
    "PREDICTIVE_MODEL",
    "TIME_SERIES",
    "ANOMALY_PREDICT",
    // Collaborative features
    "COLLABORATE",
    "CONFLICT_RESOLVE",
    "VERSION_MERGE",
    "COLLAB_EDIT",
    "SHARED_SCHEMA",
    // Spatial-temporal fusion
    "SPATIO_TEMPORAL",
    "GEOSPATIAL_TIME",
    "LOCATION_TRACK",
    "MOVEMENT_PATTERN",
    "GEO_FENCE",
    // Federated learning
    "FEDERATED_LEARN",
    "MODEL_AGGREGATE",
    "PRIVACY_PRESERVE",
    "DISTRIBUTED_TRAIN",
    "SECURE_AGG",
    // Auto-scaling and intelligence
    "AUTO_SCALE",
    "INTELLIGENT_ROUTE",
    "LOAD_BALANCE",
    "RESOURCE_PREDICT",
    "ADAPTIVE_CACHE",
    // Holographic data
    "HOLOGRAPHIC",
    "MULTI_DIMENSIONAL",
    "DATA_CUBE",
    "OLAP_ANALYZE",
    "DIMENSION_SLICE",
    // Neural database
    "NEURAL_INDEX",
    "AUTO_LEARN",
    "PATTERN_RECOGNIZE",
    "ADAPTIVE_QUERY",
    "LEARNING_OPTIMIZE",
    // Quantum computing integration
    "QUANTUM_COMPUTE",
    "QUBIT_PROCESS",
    "QUANTUM_SIMULATE",
    "SUPERPOSITION_QUERY",
    "ENTANGLED_DATA",
    // Bio-inspired algorithms
    "BIO_INSPIRED",
    "GENETIC_OPTIMIZE",
    "SWARM_INTELLIGENCE",
    "NEURAL_NETWORK_DB",
    "EVOLUTIONARY_QUERY",
    // Consciousness and awareness
    "SELF_AWARE",
    "ADAPTIVE_SYSTEM",
    "LEARNING_FEEDBACK",
    "SYSTEM_INTELLIGENCE",
    "AWARE_QUERY",
    // Multi-universe data
    "MULTIVERSE",
    "PARALLEL_UNIVERSE",
    "DIMENSIONAL_QUERY",
    "ALTERNATE_REALITY",
    "CROSS_REALM_JOIN",
    // Time crystal computing
    "TIME_CRYSTAL",
    "TEMPORAL_CRYSTAL",
    "PERIODIC_COMPUTE",
    "CRYSTAL_OPTIMIZE",
    "TIME_SYMMETRY",
];

/// Reserved table names that cannot be used by users
pub const RESERVED_TABLE_NAMES: &[&str] = &["information_schema", "pg_catalog", "aurora_system", "aurora_temp"];

/// Default system schemas
pub const SYSTEM_SCHEMAS: &[&str] = &["information_schema", "aurora_system"];

/// Authentication and security constants
pub const JWT_DEFAULT_EXPIRATION_HOURS: i64 = 24;
pub const PASSWORD_MIN_LENGTH: usize = 8;
pub const PASSWORD_MAX_LENGTH: usize = 128;
pub const SESSION_TIMEOUT_MINUTES: u64 = 60;

/// Performance monitoring constants
pub const METRICS_COLLECTION_INTERVAL_SECS: u64 = 60;
pub const SLOW_QUERY_THRESHOLD_MS: u64 = 1000;
pub const CONNECTION_POOL_SIZE_DEFAULT: usize = 10;

/// Temporal versioning constants
pub const TEMPORAL_VERSION_COLUMN: &str = "temporal_version";
pub const TEMPORAL_OPERATION_COLUMN: &str = "operation_type";

/// Operation types for temporal history tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalOperation {
    Insert,
    Update,
    Delete,
}

/// Document/collection constants for hybrid model
pub const MAX_COLLECTION_NAME_LENGTH: usize = 63;
pub const DOCUMENT_ID_FIELD: &str = "_id";
pub const DOCUMENT_VERSION_FIELD: &str = "_version";
pub const DOCUMENT_CREATED_AT_FIELD: &str = "_created_at";
pub const DOCUMENT_UPDATED_AT_FIELD: &str = "_updated_at";

/// Query optimization hints
pub const DEFAULT_JOIN_BUFFER_SIZE: usize = 8192;
pub const DEFAULT_SORT_BUFFER_SIZE: usize = 16384;
pub const MAX_JOIN_TABLES: usize = 64;

/// Backup and recovery constants
pub const BACKUP_CHUNK_SIZE_BYTES: usize = 64 * 1024 * 1024; // 64MB
pub const BACKUP_COMPRESSION_LEVEL: i32 = 6;
pub const WAL_SEGMENT_SIZE_BYTES: usize = 16 * 1024 * 1024; // 16MB

/// Network protocol constants
pub const PROTOCOL_VERSION: u16 = 1;
pub const MAX_MESSAGE_SIZE_BYTES: usize = 100 * 1024 * 1024; // 100MB
pub const HEARTBEAT_INTERVAL_SECS: u64 = 30;

/// Replication and clustering constants
pub const DEFAULT_REPLICATION_FACTOR: usize = 3;
pub const MAX_REPLICA_LAG_SECS: u64 = 300;
pub const ELECTION_TIMEOUT_MIN_MS: u64 = 150;
pub const ELECTION_TIMEOUT_MAX_MS: u64 = 300;

/// Memory management constants
pub const MEMORY_CLEANUP_INTERVAL_SECS: u64 = 300;
pub const CACHE_EVICTION_BATCH_SIZE: usize = 1000;
pub const MAX_MEMORY_USAGE_PERCENT: u8 = 80;

/// Document database constants
pub const MAX_DOCUMENT_DEPTH: usize = 10;

/// Key-value store constants
pub const MAX_KEY_LENGTH: usize = 255;
pub const MAX_NAMESPACE_LENGTH: usize = 63;

/// Logging and debugging constants
pub const LOG_MAX_FILE_SIZE_MB: usize = 100;
pub const LOG_MAX_FILES: usize = 10;
pub const QUERY_LOG_MAX_ENTRIES: usize = 10000;
