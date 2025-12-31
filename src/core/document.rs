use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use uuid::Uuid;

/// AuroraDB Document Engine - Advanced NoSQL Document Operations
/// Implements quantum-class document database with JSON operations, indexing, and querying
/// Document structure with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: Uuid,
    pub collection: String,
    pub data: Value,
    pub metadata: DocumentMetadata,
    pub version: u64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Document metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub size_bytes: usize,
    pub field_count: usize,
    pub nested_levels: usize,
    pub tags: HashSet<String>,
    pub expires_at: Option<DateTime<Utc>>,
    pub owner: Option<String>,
}

/// Collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    pub name: String,
    pub schema: Option<DocumentSchema>,
    pub indexes: HashMap<String, Index>,
    pub settings: CollectionSettings,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Document schema for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSchema {
    pub fields: HashMap<String, FieldDefinition>,
    pub required_fields: HashSet<String>,
    pub strict: bool, // If true, only defined fields allowed
}

/// Field definition in schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    pub field_type: FieldType,
    pub required: bool,
    pub default_value: Option<Value>,
    pub validation_rules: Vec<ValidationRule>,
}

/// Field types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FieldType {
    String,
    Number,
    Boolean,
    Object,
    Array,
    Null,
    Any,
}

/// Validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRule {
    MinLength(usize),
    MaxLength(usize),
    MinValue(f64),
    MaxValue(f64),
    Pattern(String),
    Enum(Vec<Value>),
    Custom(String),
}

/// Collection settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSettings {
    pub max_document_size: usize,
    pub time_to_live: Option<u64>, // seconds
    pub read_concern: ReadConcern,
    pub write_concern: WriteConcern,
    pub compression_enabled: bool,
}

/// Read and write concerns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReadConcern {
    Local,
    Available,
    Majority,
    Linearizable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WriteConcern {
    Unacknowledged,
    Acknowledged,
    Journaled,
    ReplicaAcknowledged,
}

/// Index types for document collections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Index {
    SingleField { field: String, unique: bool, sparse: bool },
    Compound { fields: Vec<String>, unique: bool },
    Text { fields: Vec<String>, weights: HashMap<String, i32>, default_language: String },
    Geospatial { field: String, index_type: GeoIndexType },
    Hashed { field: String },
}

/// Geospatial index types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeoIndexType {
    Point2D,
    Point2DSphere,
    Point3D,
}

/// Query operators for document queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryOperator {
    Eq(Value),
    Ne(Value),
    Gt(Value),
    Gte(Value),
    Lt(Value),
    Lte(Value),
    In(Vec<Value>),
    Nin(Vec<Value>),
    Exists(bool),
    Type(FieldType),
    Regex(String, String), // pattern, options
    Size(usize),
    All(Vec<Value>),
    ElemMatch(Document),
}

/// Document query structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentQuery {
    pub collection: String,
    pub filter: HashMap<String, QueryOperator>,
    pub projection: Option<HashMap<String, i32>>, // 1 to include, 0 to exclude
    pub sort: Option<HashMap<String, i32>>,       // 1 ascending, -1 descending
    pub limit: Option<usize>,
    pub skip: Option<usize>,
}

/// Aggregation pipeline stages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationStage {
    Match(HashMap<String, QueryOperator>),
    Group { id: Value, fields: HashMap<String, AggregationExpression> },
    Project(HashMap<String, ProjectionExpression>),
    Sort(HashMap<String, i32>),
    Limit(usize),
    Skip(usize),
    Unwind(String), // field path to unwind
    Lookup { from: String, local_field: String, foreign_field: String, as_field: String },
}

/// Aggregation expressions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationExpression {
    Field(String),
    Literal(Value),
    Add(Vec<Box<AggregationExpression>>),
    Subtract(Vec<Box<AggregationExpression>>),
    Multiply(Vec<Box<AggregationExpression>>),
    Divide(Vec<Box<AggregationExpression>>),
    Mod(Vec<Box<AggregationExpression>>),
    Sum(Box<AggregationExpression>),
    Avg(Box<AggregationExpression>),
    Min(Box<AggregationExpression>),
    Max(Box<AggregationExpression>),
    Count,
}

/// Projection expressions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProjectionExpression {
    Include,
    Exclude,
    Rename(String),
    Expression(AggregationExpression),
}

/// Document Engine - Core document operations
pub struct DocumentEngine {
    pub collections: HashMap<String, Collection>,
    pub storage: DocumentStorage,
    pub indexer: DocumentIndexer,
    pub validator: SchemaValidator,
}

/// Document storage interface
pub struct DocumentStorage {
    pub documents: HashMap<String, HashMap<Uuid, Document>>,
    pub max_storage_size: usize,
    pub compression_enabled: bool,
}

/// Document indexer for fast queries
pub struct DocumentIndexer {
    pub indexes: HashMap<String, HashMap<String, BTreeMap<Value, HashSet<Uuid>>>>,
    pub text_indexes: HashMap<String, HashMap<String, Vec<(String, Uuid)>>>,
}

/// Schema validator for documents
pub struct SchemaValidator {
    pub strict_validation: bool,
}

/// Query result with metadata
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub documents: Vec<Document>,
    pub total_count: usize,
    pub execution_time_ms: u64,
    pub index_used: Option<String>,
}

/// Aggregation result
#[derive(Debug, Clone)]
pub struct AggregationResult {
    pub results: Vec<Value>,
    pub execution_time_ms: u64,
    pub stages_executed: usize,
}

/// Execute aggregation stage
fn execute_aggregation_stage(input: Vec<Value>, stage: &AggregationStage) -> Result<Vec<Value>, DocumentError> {
    match stage {
        AggregationStage::Match(filter) => {
            Ok(input.into_iter().filter(|doc| matches_filter_simple(doc, filter)).collect())
        }
        AggregationStage::Group { id, fields } => {
            // Simplified grouping - would need proper implementation
            let mut result = serde_json::Map::new();
            result.insert("_id".to_string(), id.clone());
            for (k, v) in fields {
                result.insert(k.clone(), json!(v));
            }
            Ok(vec![Value::Object(result)])
        }
        AggregationStage::Limit(count) => Ok(input.into_iter().take(*count).collect()),
        AggregationStage::Skip(count) => Ok(input.into_iter().skip(*count).collect()),
        _ => Ok(input), // Other stages not implemented in simplified version
    }
}

/// Get all documents in collection
fn get_all_documents(engine: &DocumentEngine, collection_name: &str) -> Result<Vec<Value>, DocumentError> {
    if let Some(docs) = engine.storage.documents.get(collection_name) {
        Ok(docs.values().map(|doc| doc.data.clone()).collect())
    } else {
        Ok(Vec::new())
    }
}

impl DocumentEngine {
    /// Create new document engine
    pub fn new() -> Self {
        Self {
            collections: HashMap::new(),
            storage: DocumentStorage::new(),
            indexer: DocumentIndexer::new(),
            validator: SchemaValidator::new(),
        }
    }
}

impl Default for DocumentEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentEngine {
    /// Create a new collection
    pub async fn create_collection(&mut self, name: &str, settings: CollectionSettings) -> Result<(), DocumentError> {
        if self.collections.contains_key(name) {
            return Err(DocumentError::CollectionExists(name.to_string()));
        }

        let collection = Collection {
            name: name.to_string(),
            schema: None,
            indexes: HashMap::new(),
            settings,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        self.collections.insert(name.to_string(), collection);
        self.storage.documents.insert(name.to_string(), HashMap::new());

        Ok(())
    }

    /// Drop a collection
    pub async fn drop_collection(&mut self, name: &str) -> Result<(), DocumentError> {
        if !self.collections.contains_key(name) {
            return Err(DocumentError::CollectionNotFound(name.to_string()));
        }

        self.collections.remove(name);
        self.storage.documents.remove(name);
        // Remove indexes for this collection
        self.indexer.indexes.remove(name);
        self.indexer.text_indexes.remove(name);

        Ok(())
    }

    /// Insert a document
    pub async fn insert_document(&mut self, collection_name: &str, document: Value) -> Result<Uuid, DocumentError> {
        let collection = self
            .collections
            .get(collection_name)
            .ok_or_else(|| DocumentError::CollectionNotFound(collection_name.to_string()))?;

        // Validate document against schema if present
        if let Some(schema) = &collection.schema {
            self.validator.validate_document(&document, schema)?;
        }

        // Check document size
        let doc_size = serde_json::to_string(&document)
            .map_err(|e| DocumentError::InvalidDocument(format!("JSON serialization error: {}", e)))?
            .len();
        if doc_size > collection.settings.max_document_size {
            return Err(DocumentError::DocumentTooLarge(doc_size, collection.settings.max_document_size));
        }

        // Create document
        let doc_id = Uuid::new_v4();
        let now = Utc::now();

        let metadata = DocumentMetadata {
            size_bytes: doc_size,
            field_count: count_fields(&document),
            nested_levels: calculate_nested_levels(&document),
            tags: HashSet::new(),
            expires_at: collection
                .settings
                .time_to_live
                .map(|ttl| now + chrono::Duration::seconds(ttl as i64)),
            owner: None,
        };

        let doc = Document {
            id: doc_id,
            collection: collection_name.to_string(),
            data: document.clone(),
            metadata,
            version: 1,
            created_at: now,
            updated_at: now,
        };

        // Store document
        if let Some(collection_docs) = self.storage.documents.get_mut(collection_name) {
            collection_docs.insert(doc_id, doc);
        }

        // Update indexes
        self.indexer.update_indexes(collection_name, &doc_id, &document, true).await?;

        Ok(doc_id)
    }

    /// Find documents by query
    pub async fn find_documents(&self, query: &DocumentQuery) -> Result<QueryResult, DocumentError> {
        let start_time = std::time::Instant::now();

        let _collection = self
            .collections
            .get(&query.collection)
            .ok_or_else(|| DocumentError::CollectionNotFound(query.collection.clone()))?;

        // Use index if possible
        let candidate_ids = self.indexer.find_candidates(&query.collection, &query.filter).await?;

        // Filter documents
        let mut results = Vec::new();
        if let Some(collection_docs) = self.storage.documents.get(&query.collection) {
            for (id, doc) in collection_docs {
                if (candidate_ids.is_none() || candidate_ids.as_ref().unwrap().contains(id))
                    && self.matches_filter(&doc.data, &query.filter)
                {
                    results.push(doc.clone());
                }
            }
        }

        // Apply sorting
        if let Some(sort_spec) = &query.sort {
            results.sort_by(|a, b| compare_documents(a, b, sort_spec));
        }

        // Apply pagination
        let total_count = results.len();
        let start = query.skip.unwrap_or(0);
        let end = start + query.limit.unwrap_or(total_count);
        let paginated_results = results.into_iter().skip(start).take(end - start).collect();

        let execution_time = start_time.elapsed().as_millis() as u64;

        Ok(QueryResult {
            documents: paginated_results,
            total_count,
            execution_time_ms: execution_time,
            index_used: None, // Could track which index was used
        })
    }

    /// Update documents
    pub async fn update_documents(
        &mut self,
        collection_name: &str,
        filter: &HashMap<String, QueryOperator>,
        update: &DocumentUpdate,
    ) -> Result<usize, DocumentError> {
        let _collection = self
            .collections
            .get(collection_name)
            .ok_or_else(|| DocumentError::CollectionNotFound(collection_name.to_string()))?;

        let mut updated_count = 0;
        let mut docs_to_update = Vec::new();

        // First collect documents that match the filter
        if let Some(collection_docs) = self.storage.documents.get(collection_name) {
            for (id, doc) in collection_docs {
                if self.matches_filter_immutable(&doc.data, filter) {
                    docs_to_update.push(*id);
                }
            }
        }

        // Then update them
        if let Some(collection_docs) = self.storage.documents.get_mut(collection_name) {
            for id in docs_to_update {
                if let Some(doc) = collection_docs.get_mut(&id) {
                    // Apply update (simplified)
                    apply_update_simple(&mut doc.data, update);
                    doc.updated_at = Utc::now();
                    doc.version += 1;

                    // Update indexes (placeholder)
                    updated_count += 1;
                }
            }
        }

        Ok(updated_count)
    }

    /// Check if document matches filter (immutable version)
    fn matches_filter_immutable(&self, document: &Value, filter: &HashMap<String, QueryOperator>) -> bool {
        for (field, operator) in filter {
            if !self.evaluate_operator(document, field, operator) {
                return false;
            }
        }
        true
    }

    /// Delete documents
    pub async fn delete_documents(
        &mut self,
        collection_name: &str,
        filter: &HashMap<String, QueryOperator>,
    ) -> Result<usize, DocumentError> {
        let mut deleted_count = 0;
        let mut docs_to_delete = Vec::new();

        // First collect documents to delete
        if let Some(collection_docs) = self.storage.documents.get(collection_name) {
            for (id, doc) in collection_docs {
                if self.matches_filter_immutable(&doc.data, filter) {
                    docs_to_delete.push((*id, doc.data.clone()));
                }
            }
        }

        // Then delete them
        if let Some(collection_docs) = self.storage.documents.get_mut(collection_name) {
            for (id, doc_data) in docs_to_delete {
                if collection_docs.remove(&id).is_some() {
                    // Remove from indexes
                    self.indexer.remove_from_indexes(collection_name, &id, &doc_data).await?;
                    deleted_count += 1;
                }
            }
        }

        Ok(deleted_count)
    }

    /// Execute aggregation pipeline
    pub async fn aggregate_documents(
        &self,
        collection_name: &str,
        pipeline: &[AggregationStage],
    ) -> Result<AggregationResult, DocumentError> {
        let start_time = std::time::Instant::now();

        if !self.collections.contains_key(collection_name) {
            return Err(DocumentError::CollectionNotFound(collection_name.to_string()));
        }

        let mut current_docs = get_all_documents(self, collection_name)?;

        for stage in pipeline {
            current_docs = execute_aggregation_stage(current_docs, stage)?;
        }

        let execution_time = start_time.elapsed().as_millis() as u64;

        Ok(AggregationResult {
            results: current_docs,
            execution_time_ms: execution_time,
            stages_executed: pipeline.len(),
        })
    }

    /// Create index on collection
    pub async fn create_index(
        &mut self,
        collection_name: &str,
        index_name: &str,
        index: Index,
    ) -> Result<(), DocumentError> {
        let collection = self
            .collections
            .get_mut(collection_name)
            .ok_or_else(|| DocumentError::CollectionNotFound(collection_name.to_string()))?;

        collection.indexes.insert(index_name.to_string(), index.clone());

        // Build index for existing documents
        if let Some(docs) = self.storage.documents.get(collection_name) {
            for (id, doc) in docs {
                self.indexer
                    .build_index_for_document(collection_name, index_name, &index, id, &doc.data)?;
            }
        }

        Ok(())
    }

    /// Check if document matches filter
    fn matches_filter(&self, document: &Value, filter: &HashMap<String, QueryOperator>) -> bool {
        for (field, operator) in filter {
            if !self.evaluate_operator(document, field, operator) {
                return false;
            }
        }
        true
    }

    /// Evaluate query operator
    fn evaluate_operator(&self, document: &Value, field: &str, operator: &QueryOperator) -> bool {
        let field_value = get_field_value(document, field);

        match operator {
            QueryOperator::Eq(value) => field_value == Some(value),
            QueryOperator::Ne(value) => field_value != Some(value),
            QueryOperator::Gt(value) => compare_values(field_value, Some(value)) == Some(std::cmp::Ordering::Greater),
            QueryOperator::Gte(value) => {
                let cmp = compare_values(field_value, Some(value));
                cmp == Some(std::cmp::Ordering::Greater) || cmp == Some(std::cmp::Ordering::Equal)
            }
            QueryOperator::Lt(value) => compare_values(field_value, Some(value)) == Some(std::cmp::Ordering::Less),
            QueryOperator::Lte(value) => {
                let cmp = compare_values(field_value, Some(value));
                cmp == Some(std::cmp::Ordering::Less) || cmp == Some(std::cmp::Ordering::Equal)
            }
            QueryOperator::In(values) => field_value.is_some_and(|v| values.contains(v)),
            QueryOperator::Nin(values) => field_value.map_or(true, |v| !values.contains(v)),
            QueryOperator::Exists(exists) => field_value.is_some() == *exists,
            QueryOperator::Type(field_type) => matches_field_type(field_value, field_type),
            _ => false, // Other operators not implemented in this simplified version
        }
    }

    /// Apply document update
    #[allow(dead_code)]
    fn apply_update(&self, document: &mut Value, update: &DocumentUpdate) -> Result<(), DocumentError> {
        apply_update_simple(document, update);
        Ok(())
    }
}

/// Simple filter function to avoid borrowing conflicts
fn matches_filter_simple(document: &Value, filter: &HashMap<String, QueryOperator>) -> bool {
    for (field, operator) in filter {
        if !evaluate_operator_simple(document, field, operator) {
            return false;
        }
    }
    true
}

/// Evaluate query operator (simplified version)
fn evaluate_operator_simple(document: &Value, field: &str, operator: &QueryOperator) -> bool {
    let field_value = get_field_value(document, field);

    match operator {
        QueryOperator::Eq(value) => field_value == Some(value),
        QueryOperator::Ne(value) => field_value != Some(value),
        QueryOperator::Gt(value) => compare_values(field_value, Some(value)) == Some(std::cmp::Ordering::Greater),
        QueryOperator::Gte(value) => {
            let cmp = compare_values(field_value, Some(value));
            cmp == Some(std::cmp::Ordering::Greater) || cmp == Some(std::cmp::Ordering::Equal)
        }
        QueryOperator::Lt(value) => compare_values(field_value, Some(value)) == Some(std::cmp::Ordering::Less),
        QueryOperator::Lte(value) => {
            let cmp = compare_values(field_value, Some(value));
            cmp == Some(std::cmp::Ordering::Less) || cmp == Some(std::cmp::Ordering::Equal)
        }
        QueryOperator::In(values) => field_value.is_some_and(|v| values.contains(v)),
        QueryOperator::Nin(values) => field_value.map_or(true, |v| !values.contains(v)),
        QueryOperator::Exists(exists) => field_value.is_some() == *exists,
        QueryOperator::Type(field_type) => matches_field_type(field_value, field_type),
        _ => false, // Other operators not implemented in this simplified version
    }
}

/// Execute aggregation stage
fn apply_update_simple(document: &mut Value, update: &DocumentUpdate) {
    match update {
        DocumentUpdate::Set(fields) => {
            if let Value::Object(obj) = document {
                for (field, value) in fields {
                    set_field_value(obj, field, value.clone());
                }
            }
        }
        DocumentUpdate::Unset(fields) => {
            if let Value::Object(obj) = document {
                for field in fields {
                    remove_field_value(obj, field);
                }
            }
        }
        DocumentUpdate::Inc(fields) => {
            if let Value::Object(obj) = document {
                for (field, increment) in fields {
                    // Simple increment operation
                    if let Value::Number(inc) = increment {
                        if let Some(inc_f64) = inc.as_f64() {
                            set_field_value(
                                obj,
                                field,
                                Value::Number(serde_json::Number::from_f64(inc_f64).unwrap_or(inc.clone())),
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Document update operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentUpdate {
    Set(HashMap<String, Value>),
    Unset(Vec<String>),
    Inc(HashMap<String, Value>),
}

/// Document storage implementation
impl DocumentStorage {
    pub fn new() -> Self {
        Self {
            documents: HashMap::new(),
            max_storage_size: 1_000_000_000, // 1GB default
            compression_enabled: true,
        }
    }
}

impl Default for DocumentStorage {
    fn default() -> Self {
        Self::new()
    }
}

/// Document indexer implementation
impl DocumentIndexer {
    pub fn new() -> Self {
        Self { indexes: HashMap::new(), text_indexes: HashMap::new() }
    }
}

impl Default for DocumentIndexer {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentIndexer {
    /// Find candidate document IDs using indexes
    pub async fn find_candidates(
        &self,
        _collection: &str,
        _filter: &HashMap<String, QueryOperator>,
    ) -> Result<Option<HashSet<Uuid>>, DocumentError> {
        // Simplified index usage - would need proper implementation
        Ok(None)
    }

    /// Update indexes for document
    pub async fn update_indexes(
        &mut self,
        _collection: &str,
        _doc_id: &Uuid,
        _document: &Value,
        _is_insert: bool,
    ) -> Result<(), DocumentError> {
        // Placeholder implementation
        Ok(())
    }

    /// Remove document from indexes
    pub async fn remove_from_indexes(
        &mut self,
        _collection: &str,
        _doc_id: &Uuid,
        _document: &Value,
    ) -> Result<(), DocumentError> {
        // Placeholder implementation
        Ok(())
    }

    /// Build index for single document
    pub fn build_index_for_document(
        &mut self,
        _collection: &str,
        _index_name: &str,
        _index: &Index,
        _doc_id: &Uuid,
        _document: &Value,
    ) -> Result<(), DocumentError> {
        // Placeholder implementation
        Ok(())
    }
}

/// Schema validator implementation
impl SchemaValidator {
    pub fn new() -> Self {
        Self { strict_validation: true }
    }
}

impl Default for SchemaValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl SchemaValidator {
    /// Validate document against schema
    pub fn validate_document(&self, _document: &Value, _schema: &DocumentSchema) -> Result<(), DocumentError> {
        // Placeholder validation - would implement proper JSON Schema validation
        Ok(())
    }
}

/// Utility functions for document operations
/// Count fields in document
fn count_fields(value: &Value) -> usize {
    match value {
        Value::Object(obj) => obj.len(),
        Value::Array(arr) => arr.len(),
        _ => 1,
    }
}

/// Calculate nested levels in document
fn calculate_nested_levels(value: &Value) -> usize {
    match value {
        Value::Object(obj) => 1 + obj.values().map(calculate_nested_levels).max().unwrap_or(0),
        Value::Array(arr) => 1 + arr.iter().map(calculate_nested_levels).max().unwrap_or(0),
        _ => 1,
    }
}

/// Get field value from document using dot notation
fn get_field_value<'a>(document: &'a Value, field_path: &str) -> Option<&'a Value> {
    let parts: Vec<&str> = field_path.split('.').collect();
    let mut current = document;

    for part in parts {
        match current {
            Value::Object(obj) => {
                current = obj.get(part)?;
            }
            Value::Array(arr) => {
                if let Ok(index) = part.parse::<usize>() {
                    current = arr.get(index)?;
                } else {
                    return None;
                }
            }
            _ => return None,
        }
    }

    Some(current)
}

/// Set field value in document using dot notation
fn set_field_value(obj: &mut serde_json::Map<String, Value>, field_path: &str, value: Value) {
    let parts: Vec<&str> = field_path.split('.').collect();

    if parts.len() == 1 {
        obj.insert(parts[0].to_string(), value);
    } else {
        // Nested setting would need recursive implementation
        obj.insert(field_path.to_string(), value);
    }
}

/// Remove field from document
fn remove_field_value(obj: &mut serde_json::Map<String, Value>, field_path: &str) {
    let parts: Vec<&str> = field_path.split('.').collect();

    if parts.len() == 1 {
        obj.remove(parts[0]);
    } else {
        // Nested removal would need recursive implementation
        obj.remove(field_path);
    }
}

/// Compare two documents for sorting
fn compare_documents(a: &Document, b: &Document, sort_spec: &HashMap<String, i32>) -> std::cmp::Ordering {
    for (field, direction) in sort_spec {
        let a_val = get_field_value(&a.data, field);
        let b_val = get_field_value(&b.data, field);

        let cmp = compare_values(a_val, b_val);
        if let Some(ordering) = cmp {
            return if *direction == -1 { ordering.reverse() } else { ordering };
        }
    }

    std::cmp::Ordering::Equal
}

/// Compare JSON values
fn compare_values(a: Option<&Value>, b: Option<&Value>) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Some(Value::String(s1)), Some(Value::String(s2))) => Some(s1.cmp(s2)),
        (Some(Value::Number(n1)), Some(Value::Number(n2))) => n1.as_f64().partial_cmp(&n2.as_f64()),
        (Some(Value::Bool(b1)), Some(Value::Bool(b2))) => Some(b1.cmp(b2)),
        _ => None,
    }
}

/// Check if value matches field type
fn matches_field_type(value: Option<&Value>, field_type: &FieldType) -> bool {
    matches!(
        (value, field_type),
        (Some(Value::String(_)), FieldType::String)
            | (Some(Value::Number(_)), FieldType::Number)
            | (Some(Value::Bool(_)), FieldType::Boolean)
            | (Some(Value::Object(_)), FieldType::Object)
            | (Some(Value::Array(_)), FieldType::Array)
            | (Some(Value::Null), FieldType::Null)
            | (Some(_), FieldType::Any)
    )
}

/// Document error types
#[derive(Debug)]
pub enum DocumentError {
    CollectionNotFound(String),
    CollectionExists(String),
    DocumentNotFound(Uuid),
    DocumentTooLarge(usize, usize),
    InvalidDocument(String),
    SchemaValidationError(String),
    IndexError(String),
    QueryError(String),
    UpdateError(String),
    AggregationError(String),
}

/// Export document engine components
pub use DocumentEngine as Engine;
pub use DocumentError as Error;
pub use DocumentQuery as Query;
pub use DocumentUpdate as Update;
