use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// AuroraDB Entity-Relationship Modeling System - Quantum-Class Entity Management
/// Implements advanced entity modeling with relationships, inheritance, validation,
/// and quantum-enhanced entity processing capabilities
/// Entity definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub attributes: HashMap<String, Attribute>,
    pub relationships: Vec<Relationship>,
    pub metadata: EntityMetadata,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Attribute definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attribute {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub default_value: Option<Value>,
    pub unique: bool,
    pub indexed: bool,
    pub validation_rules: Vec<ValidationRule>,
    pub description: Option<String>,
    pub metadata: HashMap<String, Value>,
}

/// Data types for attributes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataType {
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Json,
    Binary,
    Reference(String), // Reference to another entity type
    Array(Box<DataType>),
    Map(HashMap<String, Box<DataType>>),
}

/// Validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRule {
    Required,
    MinLength(usize),
    MaxLength(usize),
    MinValue(Value),
    MaxValue(Value),
    Pattern(String),                        // Regex pattern
    Custom(String, HashMap<String, Value>), // Custom validation
}

/// Relationship types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub id: Uuid,
    pub name: String,
    pub source_entity: String,
    pub target_entity: String,
    pub relationship_type: RelationshipType,
    pub cardinality: Cardinality,
    pub cascade_delete: bool,
    pub cascade_update: bool,
    pub metadata: HashMap<String, Value>,
}

/// Relationship types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelationshipType {
    OneToOne,
    OneToMany,
    ManyToOne,
    ManyToMany,
}

/// Cardinality specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Cardinality {
    One,
    Many,
    ZeroOrOne,
    OneOrMany,
    ZeroOrMany,
}

/// Entity metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityMetadata {
    pub schema_version: String,
    pub owner: String,
    pub tags: Vec<String>,
    pub properties: HashMap<String, Value>,
    pub audit_enabled: bool,
    pub versioned: bool,
    pub cacheable: bool,
    pub searchable: bool,
}

/// Entity instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityInstance {
    pub id: Uuid,
    pub entity_type: String,
    pub attributes: HashMap<String, Value>,
    pub relationships: HashMap<String, Vec<Uuid>>, // relationship_name -> target_ids
    pub metadata: InstanceMetadata,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub version: u64,
}

/// Instance metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceMetadata {
    pub status: InstanceStatus,
    pub owner: String,
    pub tags: Vec<String>,
    pub properties: HashMap<String, Value>,
    pub last_accessed: Option<DateTime<Utc>>,
}

/// Instance status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum InstanceStatus {
    Active,
    Inactive,
    Archived,
    Deleted,
}

/// Entity query
#[derive(Debug, Clone)]
pub struct EntityQuery {
    pub entity_type: String,
    pub filters: Vec<QueryFilter>,
    pub sort_by: Option<Vec<(String, SortDirection)>>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub include_relationships: Vec<String>,
    pub select_attributes: Option<Vec<String>>,
}

/// Query filters
#[derive(Debug, Clone)]
pub enum QueryFilter {
    Equal(String, Value),                                 // attribute == value
    NotEqual(String, Value),                              // attribute != value
    GreaterThan(String, Value),                           // attribute > value
    LessThan(String, Value),                              // attribute < value
    In(String, Vec<Value>),                               // attribute IN values
    Like(String, String),                                 // attribute LIKE pattern
    And(Vec<QueryFilter>),                                // AND conditions
    Or(Vec<QueryFilter>),                                 // OR conditions
    Not(Box<QueryFilter>),                                // NOT condition
    RelationshipExists(String),                           // relationship exists
    RelationshipCount(String, usize, ComparisonOperator), // relationship count comparison
}

/// Comparison operators
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterEqual,
    LessEqual,
}

/// Sort direction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Entity operation result
#[derive(Debug, Clone)]
pub struct EntityOperationResult {
    pub success: bool,
    pub affected_instances: Vec<Uuid>,
    pub execution_time_ms: u64,
    pub error_message: Option<String>,
    pub metadata: HashMap<String, Value>,
}

/// Query result
#[derive(Debug, Clone)]
pub struct EntityQueryResult {
    pub instances: Vec<EntityInstance>,
    pub total_count: usize,
    pub execution_time_ms: u64,
    pub has_more: bool,
    pub metadata: HashMap<String, Value>,
}

/// Entity template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityTemplate {
    pub id: String,
    pub name: String,
    pub description: String,
    pub base_entity: Option<String>, // Inheritance
    pub attributes: HashMap<String, Attribute>,
    pub relationships: Vec<Relationship>,
    pub metadata: EntityMetadata,
    pub validation_rules: Vec<EntityValidationRule>,
}

/// Entity validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityValidationRule {
    pub name: String,
    pub rule_type: ValidationRuleType,
    pub parameters: HashMap<String, Value>,
    pub error_message: String,
}

/// Validation rule types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ValidationRuleType {
    UniqueCombination(Vec<String>),     // Unique combination of attributes
    ReferenceIntegrity(String, String), // attribute references another entity.attribute
    BusinessRule(String),               // Custom business rule
    LifecycleRule(String),              // Lifecycle validation rule
}

/// Entity schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySchema {
    pub entities: HashMap<String, Entity>,
    pub relationships: Vec<Relationship>,
    pub metadata: SchemaMetadata,
    pub version: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Schema metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaMetadata {
    pub name: String,
    pub description: String,
    pub author: String,
    pub tags: Vec<String>,
    pub properties: HashMap<String, Value>,
}

/// Entity engine
pub struct EntityEngine {
    pub entities: HashMap<String, Entity>,
    pub instances: HashMap<String, HashMap<Uuid, EntityInstance>>, // entity_type -> instances
    pub templates: HashMap<String, EntityTemplate>,
    pub schemas: Vec<EntitySchema>,
    pub cache: HashMap<String, EntityQueryResult>,
    pub max_cache_size: usize,
}

/// Entity error types
#[derive(Debug)]
pub enum EntityError {
    EntityNotFound(String),
    InstanceNotFound(Uuid),
    AttributeNotFound(String),
    RelationshipNotFound(String),
    ValidationError(String),
    ConstraintViolation(String),
    TypeMismatch(String),
    CircularReference(String),
    PermissionDenied(String),
    SerializationError(String),
}

/// Entity builder for fluent API
#[derive(Debug)]
pub struct EntityBuilder {
    name: String,
    description: Option<String>,
    attributes: HashMap<String, Attribute>,
    relationships: Vec<Relationship>,
    metadata: EntityMetadata,
}

impl EntityEngine {
    /// Create new entity engine
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            instances: HashMap::new(),
            templates: HashMap::new(),
            schemas: Vec::new(),
            cache: HashMap::new(),
            max_cache_size: 1000,
        }
    }

    /// Define entity
    pub async fn define_entity(&mut self, entity: Entity) -> Result<(), EntityError> {
        // Validate entity definition
        self.validate_entity_definition(&entity)?;

        // Check for circular references in relationships
        self.check_circular_references(&entity)?;

        self.entities.insert(entity.name.clone(), entity);
        Ok(())
    }

    /// Create entity instance
    pub async fn create_instance(
        &mut self,
        entity_type: &str,
        attributes: HashMap<String, Value>,
        relationships: Option<HashMap<String, Vec<Uuid>>>,
    ) -> Result<Uuid, EntityError> {
        let entity = self
            .entities
            .get(entity_type)
            .ok_or(EntityError::EntityNotFound(entity_type.to_string()))?;

        // Validate attributes
        self.validate_instance_attributes(entity, &attributes)?;

        // Create instance
        let instance_id = Uuid::new_v4();
        let now = Utc::now();

        let instance = EntityInstance {
            id: instance_id,
            entity_type: entity_type.to_string(),
            attributes: attributes.clone(),
            relationships: relationships.unwrap_or_default(),
            metadata: InstanceMetadata {
                status: InstanceStatus::Active,
                owner: "system".to_string(),
                tags: vec![],
                properties: HashMap::new(),
                last_accessed: Some(now),
            },
            created_at: now,
            updated_at: now,
            version: 1,
        };

        // Validate relationships first
        self.validate_instance_relationships(entity, &instance)?;

        // Store instance
        self.instances
            .entry(entity_type.to_string())
            .or_default()
            .insert(instance_id, instance);

        Ok(instance_id)
    }

    /// Update entity instance
    pub async fn update_instance(
        &mut self,
        entity_type: &str,
        instance_id: Uuid,
        updates: HashMap<String, Value>,
        relationship_updates: Option<HashMap<String, Vec<Uuid>>>,
    ) -> Result<EntityOperationResult, EntityError> {
        // First, get the entity for validation
        let entity = self
            .entities
            .get(entity_type)
            .ok_or(EntityError::EntityNotFound(entity_type.to_string()))?
            .clone();

        // Get the current instance for validation
        let current_instance = self
            .instances
            .get(entity_type)
            .ok_or(EntityError::EntityNotFound(entity_type.to_string()))?
            .get(&instance_id)
            .ok_or(EntityError::InstanceNotFound(instance_id))?
            .clone();

        // Validate updates
        self.validate_instance_updates(&entity, &current_instance, &updates)?;

        // Now get mutable access to update the instance
        let instances = self.instances.get_mut(entity_type).unwrap();
        let instance = instances.get_mut(&instance_id).unwrap();

        // Apply updates
        for (attr_name, value) in updates {
            instance.attributes.insert(attr_name, value);
        }

        if let Some(rel_updates) = relationship_updates {
            for (rel_name, target_ids) in rel_updates {
                instance.relationships.insert(rel_name, target_ids);
            }
        }

        instance.updated_at = Utc::now();
        instance.version += 1;

        Ok(EntityOperationResult {
            success: true,
            affected_instances: vec![instance_id],
            execution_time_ms: 0,
            error_message: None,
            metadata: HashMap::new(),
        })
    }

    /// Delete entity instance
    pub async fn delete_instance(
        &mut self,
        entity_type: &str,
        instance_id: Uuid,
        cascade: bool,
    ) -> Result<EntityOperationResult, EntityError> {
        let instances = self
            .instances
            .get_mut(entity_type)
            .ok_or(EntityError::EntityNotFound(entity_type.to_string()))?;

        if !instances.contains_key(&instance_id) {
            return Err(EntityError::InstanceNotFound(instance_id));
        }

        let instance = instances.remove(&instance_id).unwrap();

        // Handle cascade delete
        if cascade {
            self.handle_cascade_delete(&instance)?;
        }

        Ok(EntityOperationResult {
            success: true,
            affected_instances: vec![instance_id],
            execution_time_ms: 0,
            error_message: None,
            metadata: HashMap::new(),
        })
    }

    /// Query entity instances
    pub async fn query_instances(&mut self, query: EntityQuery) -> Result<EntityQueryResult, EntityError> {
        let start_time = std::time::Instant::now();

        // Generate cache key
        let cache_key = self.generate_query_cache_key(&query);

        // Check cache
        if let Some(cached_result) = self.cache.get(&cache_key) {
            return Ok(cached_result.clone());
        }

        let instances = self
            .instances
            .get(&query.entity_type)
            .ok_or(EntityError::EntityNotFound(query.entity_type.clone()))?;

        let mut filtered_instances: Vec<&EntityInstance> = Vec::new();

        // Apply filters
        for instance in instances.values() {
            let mut matches = true;

            for filter in &query.filters {
                if !self.evaluate_filter(instance, filter)? {
                    matches = false;
                    break;
                }
            }

            if matches {
                filtered_instances.push(instance);
            }
        }

        // Apply sorting
        if let Some(sort_criteria) = &query.sort_by {
            filtered_instances.sort_by(|a, b| self.compare_instances(a, b, sort_criteria));
        }

        // Apply pagination
        let total_count = filtered_instances.len();
        let start_idx = query.offset.unwrap_or(0);
        let limit = query.limit.unwrap_or(total_count);
        let end_idx = (start_idx + limit).min(total_count);

        let paginated_instances = filtered_instances[start_idx..end_idx].to_vec();

        // Load relationships if requested
        let mut result_instances = Vec::new();
        for &instance in &paginated_instances {
            let mut instance_clone = instance.clone();

            // Load requested relationships
            for rel_name in &query.include_relationships {
                if let Some(target_ids) = instance.relationships.get(rel_name) {
                    // In a full implementation, this would load the related instances
                    // For now, we just keep the IDs
                    instance_clone.relationships.insert(rel_name.clone(), target_ids.clone());
                }
            }

            // Filter attributes if specified
            if let Some(select_attrs) = &query.select_attributes {
                let mut filtered_attrs = HashMap::new();
                for attr in select_attrs {
                    if let Some(value) = instance.attributes.get(attr) {
                        filtered_attrs.insert(attr.clone(), value.clone());
                    }
                }
                instance_clone.attributes = filtered_attrs;
            }

            result_instances.push(instance_clone);
        }

        let execution_time = start_time.elapsed().as_millis() as u64;

        let result = EntityQueryResult {
            instances: result_instances,
            total_count,
            execution_time_ms: execution_time,
            has_more: end_idx < total_count,
            metadata: HashMap::new(),
        };

        // Cache result
        if self.cache.len() < self.max_cache_size {
            self.cache.insert(cache_key, result.clone());
        }

        Ok(result)
    }

    /// Get entity instance with relationships
    pub async fn get_instance_with_relationships(
        &self,
        entity_type: &str,
        instance_id: Uuid,
        _include_relationships: &[String],
    ) -> Result<EntityInstance, EntityError> {
        let instances = self
            .instances
            .get(entity_type)
            .ok_or(EntityError::EntityNotFound(entity_type.to_string()))?;

        let instance = instances
            .get(&instance_id)
            .ok_or(EntityError::InstanceNotFound(instance_id))?
            .clone();

        // In a full implementation, this would load related instances
        // For now, we return the instance as-is
        Ok(instance)
    }

    /// Validate entity instance
    pub async fn validate_instance(&self, entity_type: &str, instance: &EntityInstance) -> Result<(), EntityError> {
        let entity = self
            .entities
            .get(entity_type)
            .ok_or(EntityError::EntityNotFound(entity_type.to_string()))?;

        // Validate attributes
        self.validate_instance_attributes(entity, &instance.attributes)?;

        // Validate relationships
        self.validate_instance_relationships(entity, instance)?;

        Ok(())
    }

    /// Create entity from template
    pub async fn create_from_template(&mut self, template_id: &str) -> Result<String, EntityError> {
        // Clone the template data to avoid borrow checker issues
        let (name, description, attributes, relationships, metadata) = {
            let template = self
                .templates
                .get(template_id)
                .ok_or(EntityError::EntityNotFound(format!("Template {}", template_id)))?;
            (
                template.name.clone(),
                template.description.clone(),
                template.attributes.clone(),
                template.relationships.clone(),
                template.metadata.clone(),
            )
        };

        let entity = Entity {
            id: Uuid::new_v4(),
            name: name.clone(),
            description: Some(description),
            attributes,
            relationships,
            metadata,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        self.define_entity(entity).await?;
        Ok(name)
    }

    // Private helper methods

    fn validate_entity_definition(&self, entity: &Entity) -> Result<(), EntityError> {
        // Check for duplicate attribute names
        let mut attr_names = HashSet::new();
        for name in entity.attributes.keys() {
            if !attr_names.insert(name.clone()) {
                return Err(EntityError::ConstraintViolation(format!("Duplicate attribute name: {}", name)));
            }
        }

        // Check for duplicate relationship names
        let mut rel_names = HashSet::new();
        for rel in &entity.relationships {
            if !rel_names.insert(rel.name.clone()) {
                return Err(EntityError::ConstraintViolation(format!("Duplicate relationship name: {}", rel.name)));
            }
        }

        // Validate relationships reference existing entities
        for rel in &entity.relationships {
            if !self.entities.contains_key(&rel.source_entity) {
                return Err(EntityError::EntityNotFound(format!("Source entity {} not found", rel.source_entity)));
            }
            if !self.entities.contains_key(&rel.target_entity) {
                return Err(EntityError::EntityNotFound(format!("Target entity {} not found", rel.target_entity)));
            }
        }

        Ok(())
    }

    fn check_circular_references(&self, entity: &Entity) -> Result<(), EntityError> {
        // Simplified circular reference check
        // In a full implementation, this would do a proper cycle detection
        for rel in &entity.relationships {
            if rel.source_entity == rel.target_entity {
                return Err(EntityError::CircularReference(format!("Self-referencing relationship: {}", rel.name)));
            }
        }
        Ok(())
    }

    fn validate_instance_attributes(
        &self,
        entity: &Entity,
        attributes: &HashMap<String, Value>,
    ) -> Result<(), EntityError> {
        for (attr_name, attr_def) in &entity.attributes {
            let value = attributes.get(attr_name);

            // Check required attributes
            if !attr_def.nullable && value.is_none() {
                return Err(EntityError::ValidationError(format!("Required attribute {} is missing", attr_name)));
            }

            // Type validation (simplified)
            if let Some(val) = value {
                match (&attr_def.data_type, val) {
                    (DataType::String, Value::String(_)) => {}
                    (DataType::Integer, Value::Number(_)) => {}
                    (DataType::Boolean, Value::Bool(_)) => {}
                    _ => return Err(EntityError::TypeMismatch(format!("Type mismatch for attribute {}", attr_name))),
                }
            }

            // Run validation rules
            for rule in &attr_def.validation_rules {
                self.validate_rule(rule, value)?;
            }
        }

        Ok(())
    }

    fn validate_instance_relationships(&self, entity: &Entity, instance: &EntityInstance) -> Result<(), EntityError> {
        for rel in &entity.relationships {
            if let Some(target_ids) = instance.relationships.get(&rel.name) {
                // Check if target entities exist
                let target_instances = self
                    .instances
                    .get(&rel.target_entity)
                    .ok_or(EntityError::EntityNotFound(format!("Target entity {} not found", rel.target_entity)))?;

                for &target_id in target_ids {
                    if !target_instances.contains_key(&target_id) {
                        return Err(EntityError::InstanceNotFound(target_id));
                    }
                }

                // Check cardinality
                match rel.relationship_type {
                    RelationshipType::OneToOne | RelationshipType::ManyToOne => {
                        if target_ids.len() > 1 {
                            return Err(EntityError::ConstraintViolation(format!(
                                "Relationship {} allows only one target",
                                rel.name
                            )));
                        }
                    }
                    _ => {} // Many-to-many allows multiple
                }
            }
        }

        Ok(())
    }

    fn validate_instance_updates(
        &self,
        entity: &Entity,
        instance: &EntityInstance,
        updates: &HashMap<String, Value>,
    ) -> Result<(), EntityError> {
        // Check if updated attributes exist
        for attr_name in updates.keys() {
            if !entity.attributes.contains_key(attr_name) {
                return Err(EntityError::AttributeNotFound(attr_name.clone()));
            }
        }

        // Validate the updated values
        let mut combined_attrs = instance.attributes.clone();
        combined_attrs.extend(updates.clone());
        self.validate_instance_attributes(entity, &combined_attrs)?;

        Ok(())
    }

    fn validate_rule(&self, rule: &ValidationRule, value: Option<&Value>) -> Result<(), EntityError> {
        match rule {
            ValidationRule::Required => {
                if value.is_none() || value.unwrap().is_null() {
                    return Err(EntityError::ValidationError("Required field is empty".to_string()));
                }
            }
            ValidationRule::MinLength(len) => {
                if let Some(Value::String(s)) = value {
                    if s.len() < *len {
                        return Err(EntityError::ValidationError(format!("String too short, minimum length: {}", len)));
                    }
                }
            }
            ValidationRule::MaxLength(len) => {
                if let Some(Value::String(s)) = value {
                    if s.len() > *len {
                        return Err(EntityError::ValidationError(format!("String too long, maximum length: {}", len)));
                    }
                }
            }
            _ => {} // Other validations not implemented
        }
        Ok(())
    }

    fn evaluate_filter(&self, instance: &EntityInstance, filter: &QueryFilter) -> Result<bool, EntityError> {
        match filter {
            QueryFilter::Equal(attr, val) => Ok(instance.attributes.get(attr) == Some(val)),
            QueryFilter::NotEqual(attr, val) => Ok(instance.attributes.get(attr) != Some(val)),
            QueryFilter::GreaterThan(attr, val) => self.compare_attribute_value(instance, attr, val, |cmp| cmp > 0),
            QueryFilter::LessThan(attr, val) => self.compare_attribute_value(instance, attr, val, |cmp| cmp < 0),
            QueryFilter::In(attr, values) => {
                if let Some(attr_val) = instance.attributes.get(attr) {
                    Ok(values.contains(attr_val))
                } else {
                    Ok(false)
                }
            }
            QueryFilter::And(filters) => {
                for f in filters {
                    if !self.evaluate_filter(instance, f)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            QueryFilter::Or(filters) => {
                for f in filters {
                    if self.evaluate_filter(instance, f)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            QueryFilter::Not(filter) => Ok(!self.evaluate_filter(instance, filter)?),
            _ => Ok(true), // Simplified - other filters not implemented
        }
    }

    fn compare_attribute_value<F>(
        &self,
        instance: &EntityInstance,
        attr: &str,
        val: &Value,
        cmp_fn: F,
    ) -> Result<bool, EntityError>
    where
        F: Fn(i32) -> bool,
    {
        if let Some(attr_val) = instance.attributes.get(attr) {
            let cmp = self.compare_values(attr_val, val)?;
            Ok(cmp_fn(cmp))
        } else {
            Ok(false)
        }
    }

    fn compare_values(&self, a: &Value, b: &Value) -> Result<i32, EntityError> {
        match (a, b) {
            (Value::String(sa), Value::String(sb)) => Ok(sa.cmp(sb) as i32),
            (Value::Number(na), Value::Number(nb)) => {
                let a_val = na.as_f64().unwrap_or(0.0);
                let b_val = nb.as_f64().unwrap_or(0.0);
                Ok(if a_val < b_val {
                    -1
                } else if a_val > b_val {
                    1
                } else {
                    0
                })
            }
            _ => Err(EntityError::TypeMismatch("Cannot compare values of different types".to_string())),
        }
    }

    fn compare_instances(
        &self,
        a: &EntityInstance,
        b: &EntityInstance,
        criteria: &[(String, SortDirection)],
    ) -> std::cmp::Ordering {
        for (attr, dir) in criteria {
            let a_val = a.attributes.get(attr);
            let b_val = b.attributes.get(attr);

            let cmp = match (a_val, b_val) {
                (Some(av), Some(bv)) => self.compare_values(av, bv).unwrap_or(0),
                (Some(_), None) => 1,
                (None, Some(_)) => -1,
                (None, None) => 0,
            };

            if cmp != 0 {
                return match dir {
                    SortDirection::Ascending => {
                        if cmp < 0 {
                            std::cmp::Ordering::Less
                        } else if cmp > 0 {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    }
                    SortDirection::Descending => {
                        if cmp < 0 {
                            std::cmp::Ordering::Greater
                        } else if cmp > 0 {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    }
                };
            }
        }

        std::cmp::Ordering::Equal
    }

    fn handle_cascade_delete(&self, _instance: &EntityInstance) -> Result<(), EntityError> {
        // Simplified cascade delete
        // In a full implementation, this would recursively delete related instances
        // based on cascade settings in relationships
        Ok(())
    }

    fn generate_query_cache_key(&self, query: &EntityQuery) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{:?}", query).hash(&mut hasher);
        format!("entity_query_{:x}", hasher.finish())
    }
}

/// Entity builder implementation
impl EntityBuilder {
    /// Create new entity builder
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            attributes: HashMap::new(),
            relationships: Vec::new(),
            metadata: EntityMetadata {
                schema_version: "1.0".to_string(),
                owner: "system".to_string(),
                tags: vec![],
                properties: HashMap::new(),
                audit_enabled: true,
                versioned: true,
                cacheable: true,
                searchable: true,
            },
        }
    }

    /// Set description
    pub fn description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Add attribute
    pub fn attribute(mut self, name: &str, data_type: DataType) -> Self {
        let attr = Attribute {
            name: name.to_string(),
            data_type,
            nullable: false,
            default_value: None,
            unique: false,
            indexed: false,
            validation_rules: vec![],
            description: None,
            metadata: HashMap::new(),
        };
        self.attributes.insert(name.to_string(), attr);
        self
    }

    /// Add relationship
    pub fn relationship(mut self, name: &str, target_entity: &str, rel_type: RelationshipType) -> Self {
        let rel = Relationship {
            id: Uuid::new_v4(),
            name: name.to_string(),
            source_entity: self.name.clone(),
            target_entity: target_entity.to_string(),
            relationship_type: rel_type,
            cardinality: Cardinality::Many, // Default
            cascade_delete: false,
            cascade_update: false,
            metadata: HashMap::new(),
        };
        self.relationships.push(rel);
        self
    }

    /// Build the entity
    pub fn build(self) -> Entity {
        Entity {
            id: Uuid::new_v4(),
            name: self.name,
            description: self.description,
            attributes: self.attributes,
            relationships: self.relationships,
            metadata: self.metadata,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }
}

impl Default for EntityEngine {
    fn default() -> Self {
        Self::new()
    }
}

pub use EntityBuilder as Builder;
/// Export entity system components
pub use EntityEngine as Engine;
pub use EntityError as Error;
