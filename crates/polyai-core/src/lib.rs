//! Observation and action encoding, game setups and game records for the polytopia AI (see `docs/encoding.md`).

pub mod action;
pub mod encoding;
pub mod game;
pub mod record;
pub mod setup;
pub mod slots;

pub use action::*;
pub use encoding::*;
pub use game::*;
pub use record::*;
pub use setup::*;
pub use slots::*;
