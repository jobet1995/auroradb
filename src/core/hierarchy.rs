use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// AuroraDB Hierarchical Data Management System - Tree Structures and Navigation
/// Implements basic hierarchical data operations with tree traversal and path finding
/// Tree structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree<T> {
    pub id: Uuid,
    pub name: String,
    pub root_id: Option<Uuid>,
    pub nodes: HashMap<Uuid, TreeNode<T>>,
}

/// Node structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode<T> {
    pub id: Uuid,
    pub parent_id: Option<Uuid>,
    pub data: T,
    pub children: Vec<Uuid>,
    pub level: u32,
}

/// Tree operation result
#[derive(Debug, Clone)]
pub struct TreeOperationResult {
    pub success: bool,
    pub affected_nodes: Vec<Uuid>,
    pub execution_time_ms: u64,
    pub error_message: Option<String>,
}

/// Path finder result
#[derive(Debug, Clone)]
pub struct PathResult {
    pub path: Vec<Uuid>,
    pub distance: u32,
    pub found: bool,
    pub visited_nodes: usize,
}

/// Hierarchy engine
pub struct HierarchyEngine<T> {
    pub trees: HashMap<Uuid, Tree<T>>,
}

/// Hierarchy error types
#[derive(Debug)]
pub enum HierarchyError {
    TreeNotFound(Uuid),
    NodeNotFound(Uuid),
    InvalidOperation(String),
    CycleDetected,
}

impl<T> HierarchyEngine<T>
where
    T: Clone,
{
    /// Create new hierarchy engine
    pub fn new() -> Self {
        Self {
            trees: HashMap::new(),
        }
    }

    /// Create tree
    pub fn create_tree(&mut self, tree: Tree<T>) -> Uuid {
        let tree_id = tree.id;
        self.trees.insert(tree_id, tree);
        tree_id
    }

    /// Insert node
    pub fn insert_node(&mut self, tree_id: Uuid, parent_id: Option<Uuid>, data: T) -> Result<Uuid, HierarchyError> {
        let tree = self.trees.get_mut(&tree_id)
            .ok_or(HierarchyError::TreeNotFound(tree_id))?;

        let node_id = Uuid::new_v4();

        // Validate parent exists if specified
        if let Some(pid) = parent_id {
            if !tree.nodes.contains_key(&pid) {
                return Err(HierarchyError::NodeNotFound(pid));
            }
        }

        // Determine level
        let level = if let Some(pid) = parent_id {
            let parent = tree.nodes.get(&pid).unwrap();
            let new_level = parent.level + 1;

            // Update parent's children
            let parent_node = tree.nodes.get_mut(&pid).unwrap();
            parent_node.children.push(node_id);

            new_level
        } else {
            // Root node
            if tree.root_id.is_some() {
                return Err(HierarchyError::InvalidOperation("Tree already has a root".to_string()));
            }
            tree.root_id = Some(node_id);
            0
        };

        // Create node
        let node = TreeNode {
            id: node_id,
            parent_id,
            data,
            children: vec![],
            level,
        };

        tree.nodes.insert(node_id, node);

        Ok(node_id)
    }

    /// Delete node
    pub fn delete_node(&mut self, tree_id: Uuid, node_id: Uuid) -> Result<TreeOperationResult, HierarchyError> {
        let tree = self.trees.get_mut(&tree_id)
            .ok_or(HierarchyError::TreeNotFound(tree_id))?;

        if !tree.nodes.contains_key(&node_id) {
            return Err(HierarchyError::NodeNotFound(node_id));
        }

        // Remove from parent's children
        if let Some(parent_id) = tree.nodes[&node_id].parent_id {
            if let Some(parent) = tree.nodes.get_mut(&parent_id) {
                parent.children.retain(|&id| id != node_id);
            }
        }

        // Remove node
        tree.nodes.remove(&node_id);

        // Update root if necessary
        if tree.root_id == Some(node_id) {
            tree.root_id = None;
        }

        Ok(TreeOperationResult {
            success: true,
            affected_nodes: vec![node_id],
            execution_time_ms: 0,
            error_message: None,
        })
    }

    /// Get node
    pub fn get_node(&self, tree_id: Uuid, node_id: Uuid) -> Result<&TreeNode<T>, HierarchyError> {
        let tree = self.trees.get(&tree_id)
            .ok_or(HierarchyError::TreeNotFound(tree_id))?;

        tree.nodes.get(&node_id)
            .ok_or(HierarchyError::NodeNotFound(node_id))
    }

    /// Get children
    pub fn get_children(&self, tree_id: Uuid, node_id: Uuid) -> Result<Vec<&TreeNode<T>>, HierarchyError> {
        let tree = self.trees.get(&tree_id)
            .ok_or(HierarchyError::TreeNotFound(tree_id))?;

        let node = tree.nodes.get(&node_id)
            .ok_or(HierarchyError::NodeNotFound(node_id))?;

        let mut children = Vec::new();
        for &child_id in &node.children {
            if let Some(child) = tree.nodes.get(&child_id) {
                children.push(child);
            }
        }

        Ok(children)
    }

    /// Get parent
    pub fn get_parent(&self, tree_id: Uuid, node_id: Uuid) -> Result<Option<&TreeNode<T>>, HierarchyError> {
        let tree = self.trees.get(&tree_id)
            .ok_or(HierarchyError::TreeNotFound(tree_id))?;

        let node = tree.nodes.get(&node_id)
            .ok_or(HierarchyError::NodeNotFound(node_id))?;

        if let Some(parent_id) = node.parent_id {
            Ok(tree.nodes.get(&parent_id))
        } else {
            Ok(None)
        }
    }

    /// Find path between nodes
    pub fn find_path(&self, tree_id: Uuid, from_node: Uuid, to_node: Uuid) -> Result<PathResult, HierarchyError> {
        let tree = self.trees.get(&tree_id)
            .ok_or(HierarchyError::TreeNotFound(tree_id))?;

        if !tree.nodes.contains_key(&from_node) || !tree.nodes.contains_key(&to_node) {
            return Ok(PathResult {
                path: vec![],
                distance: 0,
                found: false,
                visited_nodes: 0,
            });
        }

        // Simple path finding - get path to root for both and find common ancestor
        let path1 = self.get_path_to_root(tree, from_node)?;
        let path2 = self.get_path_to_root(tree, to_node)?;

        // Find the common ancestor
        let mut common_ancestor_index = 0;
        for (i, (&n1, &n2)) in path1.iter().rev().zip(path2.iter().rev()).enumerate() {
            if n1 != n2 {
                break;
            }
            common_ancestor_index = i;
        }

        // Build path from from_node up to common ancestor, then down to to_node
        let ancestor = path1[path1.len() - 1 - common_ancestor_index];
        let mut path = Vec::new();

        // From from_node to ancestor
        let mut current = from_node;
        while current != ancestor {
            path.push(current);
            if let Some(parent) = tree.nodes.get(&current).and_then(|n| n.parent_id) {
                current = parent;
            } else {
                break;
            }
        }
        path.push(ancestor);

        // From ancestor to to_node (excluding ancestor)
        if ancestor != to_node {
            let mut temp_path = Vec::new();
            current = to_node;
            while current != ancestor {
                temp_path.push(current);
                if let Some(parent) = tree.nodes.get(&current).and_then(|n| n.parent_id) {
                    current = parent;
                } else {
                    break;
                }
            }
            temp_path.reverse();
            path.extend(temp_path);
        }

        let path_len = path.len();
        let found = *path.last().unwrap_or(&Uuid::nil()) == to_node;

        Ok(PathResult {
            path,
            distance: path_len.saturating_sub(1) as u32,
            found,
            visited_nodes: path1.len() + path2.len(),
        })
    }

    // Private helper methods
    fn get_path_to_root(&self, tree: &Tree<T>, node_id: Uuid) -> Result<Vec<Uuid>, HierarchyError> {
        let mut path = vec![];
        let mut current = node_id;

        while let Some(node) = tree.nodes.get(&current) {
            path.push(current);
            if let Some(parent) = node.parent_id {
                current = parent;
            } else {
                break;
            }
        }

        Ok(path)
    }
}

/// Tree builder for fluent API
#[derive(Debug)]
pub struct TreeBuilder<T> {
    name: String,
    description: Option<String>,
    root_data: Option<T>,
}

impl<T> TreeBuilder<T> {
    /// Create new tree builder
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            root_data: None,
        }
    }

    /// Set description
    pub fn description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Set root data
    pub fn root_data(mut self, data: T) -> Self {
        self.root_data = Some(data);
        self
    }

    /// Build the tree
    pub fn build(self) -> Tree<T> {
        let mut tree = Tree {
            id: Uuid::new_v4(),
            name: self.name,
            root_id: None,
            nodes: HashMap::new(),
        };

        // Create root node if data provided
        if let Some(root_data) = self.root_data {
            let root_node = TreeNode {
                id: Uuid::new_v4(),
                parent_id: None,
                data: root_data,
                children: vec![],
                level: 0,
            };

            tree.root_id = Some(root_node.id);
            tree.nodes.insert(root_node.id, root_node);
        }

        tree
    }
}

impl Default for HierarchyEngine<()> {
    fn default() -> Self {
        Self::new()
    }
}

/// Export hierarchy system components
pub use HierarchyEngine as Engine;
pub use HierarchyError as Error;
pub use Tree as Hierarchy;
