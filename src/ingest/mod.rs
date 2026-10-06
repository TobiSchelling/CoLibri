//! Files-first ingestion: walking folders, converting files and reconciling
//! them with the metadata DB.

pub mod calibre;
pub mod convert;
pub mod frontmatter;
pub mod library;
pub mod mirror;
pub mod plantuml;
pub mod update;
pub mod walk;
