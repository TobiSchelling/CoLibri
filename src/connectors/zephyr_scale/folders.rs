//! Folder tree builder for Zephyr Scale folder hierarchies.
//!
//! Builds a lookup from flat `(id, name, parent_id)` tuples into
//! resolved full paths like `/Regression/API/Auth`.

use std::collections::{HashMap, HashSet};

/// A raw folder record from the Zephyr Scale API.
#[derive(Debug, Clone)]
pub struct RawFolder {
    pub id: i64,
    pub name: String,
    pub parent_id: Option<i64>,
}

/// A resolved folder node with its full path.
#[derive(Debug)]
struct FolderNode {
    name: String,
    parent_id: Option<i64>,
    path: String,
    children: Vec<i64>,
}

/// A tree of folders with resolved paths.
pub struct FolderTree {
    nodes: HashMap<i64, FolderNode>,
}

impl FolderTree {
    /// Build a folder tree from a flat list of raw folders.
    ///
    /// Resolves parent chains to compute full paths. Orphan folders
    /// (whose parent_id references a non-existent folder) become roots.
    pub fn build(folders: Vec<RawFolder>) -> Self {
        let mut nodes: HashMap<i64, FolderNode> = HashMap::new();

        // First pass: insert all nodes
        for f in &folders {
            nodes.insert(
                f.id,
                FolderNode {
                    name: f.name.clone(),
                    parent_id: f.parent_id,
                    path: String::new(),
                    children: Vec::new(),
                },
            );
        }

        // Second pass: link children to parents
        let ids_with_parents: Vec<(i64, i64)> = folders
            .iter()
            .filter_map(|f| f.parent_id.map(|pid| (f.id, pid)))
            .collect();

        for (child_id, parent_id) in ids_with_parents {
            if nodes.contains_key(&parent_id) {
                if let Some(parent) = nodes.get_mut(&parent_id) {
                    parent.children.push(child_id);
                }
            }
        }

        // Third pass: resolve paths via iterative traversal
        let all_ids: Vec<i64> = nodes.keys().copied().collect();
        for id in all_ids {
            let path = Self::resolve_path(id, &nodes);
            if let Some(node) = nodes.get_mut(&id) {
                node.path = path;
            }
        }

        Self { nodes }
    }

    /// Resolve the full path for a folder by walking parent chains.
    fn resolve_path(id: i64, nodes: &HashMap<i64, FolderNode>) -> String {
        let mut parts = Vec::new();
        let mut current = Some(id);
        let mut visited = HashSet::new();

        while let Some(cid) = current {
            if !visited.insert(cid) {
                break; // cycle protection
            }
            if let Some(node) = nodes.get(&cid) {
                parts.push(node.name.clone());
                current = node.parent_id.filter(|pid| nodes.contains_key(pid));
            } else {
                break;
            }
        }

        parts.reverse();
        format!("/{}", parts.join("/"))
    }

    /// Get the resolved path for a folder ID.
    pub fn get_path(&self, folder_id: i64) -> Option<&str> {
        self.nodes.get(&folder_id).map(|n| n.path.as_str())
    }

    /// Find a folder by its resolved path (case-insensitive).
    pub fn find_by_path(&self, path: &str) -> Option<i64> {
        let path_lower = path.to_lowercase();
        self.nodes
            .iter()
            .find(|(_, n)| n.path.to_lowercase() == path_lower)
            .map(|(id, _)| *id)
    }

    /// Get all folder IDs in a subtree (the folder itself + all descendants).
    pub fn get_subtree_ids(&self, folder_id: i64) -> HashSet<i64> {
        let mut result = HashSet::new();
        let mut stack = vec![folder_id];

        while let Some(id) = stack.pop() {
            if result.insert(id) {
                if let Some(node) = self.nodes.get(&id) {
                    stack.extend(&node.children);
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_folders() -> Vec<RawFolder> {
        vec![
            RawFolder {
                id: 1,
                name: "Regression".into(),
                parent_id: None,
            },
            RawFolder {
                id: 2,
                name: "API".into(),
                parent_id: Some(1),
            },
            RawFolder {
                id: 3,
                name: "Auth".into(),
                parent_id: Some(2),
            },
            RawFolder {
                id: 4,
                name: "Smoke".into(),
                parent_id: None,
            },
        ]
    }

    #[test]
    fn build_resolves_paths() {
        let tree = FolderTree::build(sample_folders());
        assert_eq!(tree.get_path(1), Some("/Regression"));
        assert_eq!(tree.get_path(2), Some("/Regression/API"));
        assert_eq!(tree.get_path(3), Some("/Regression/API/Auth"));
        assert_eq!(tree.get_path(4), Some("/Smoke"));
    }

    #[test]
    fn build_from_empty_input() {
        let tree = FolderTree::build(vec![]);
        assert_eq!(tree.get_path(1), None);
    }

    #[test]
    fn orphan_folders_become_roots() {
        let folders = vec![
            RawFolder {
                id: 10,
                name: "Child".into(),
                parent_id: Some(999), // parent doesn't exist
            },
            RawFolder {
                id: 11,
                name: "Normal".into(),
                parent_id: None,
            },
        ];
        let tree = FolderTree::build(folders);
        // Orphan becomes root since parent 999 doesn't exist
        assert_eq!(tree.get_path(10), Some("/Child"));
        assert_eq!(tree.get_path(11), Some("/Normal"));
    }

    #[test]
    fn get_subtree_ids_returns_all_descendants() {
        let tree = FolderTree::build(sample_folders());
        let subtree = tree.get_subtree_ids(1);
        assert!(subtree.contains(&1));
        assert!(subtree.contains(&2));
        assert!(subtree.contains(&3));
        assert!(!subtree.contains(&4));
    }

    #[test]
    fn get_subtree_ids_leaf_returns_self() {
        let tree = FolderTree::build(sample_folders());
        let subtree = tree.get_subtree_ids(3);
        assert_eq!(subtree.len(), 1);
        assert!(subtree.contains(&3));
    }

    #[test]
    fn get_subtree_ids_nonexistent_returns_empty() {
        let tree = FolderTree::build(sample_folders());
        let subtree = tree.get_subtree_ids(999);
        // Contains the ID itself since insert succeeds, but no children
        assert_eq!(subtree.len(), 1);
    }

    #[test]
    fn find_by_path_case_insensitive() {
        let tree = FolderTree::build(sample_folders());
        assert_eq!(tree.find_by_path("/regression/api"), Some(2));
        assert_eq!(tree.find_by_path("/REGRESSION/API"), Some(2));
    }

    #[test]
    fn find_by_path_not_found() {
        let tree = FolderTree::build(sample_folders());
        assert_eq!(tree.find_by_path("/NonExistent"), None);
    }

    #[test]
    fn deep_nesting() {
        let folders = vec![
            RawFolder {
                id: 1,
                name: "A".into(),
                parent_id: None,
            },
            RawFolder {
                id: 2,
                name: "B".into(),
                parent_id: Some(1),
            },
            RawFolder {
                id: 3,
                name: "C".into(),
                parent_id: Some(2),
            },
            RawFolder {
                id: 4,
                name: "D".into(),
                parent_id: Some(3),
            },
            RawFolder {
                id: 5,
                name: "E".into(),
                parent_id: Some(4),
            },
        ];
        let tree = FolderTree::build(folders);
        assert_eq!(tree.get_path(5), Some("/A/B/C/D/E"));
    }
}
