//! Zephyr Scale test case connector.
//!
//! Syncs test cases from the Zephyr Scale Cloud API into CoLibri's
//! canonical store as searchable markdown with YAML frontmatter.

pub mod api;
pub mod folders;
pub mod html_to_md;
pub mod render;
