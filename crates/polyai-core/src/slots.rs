//! Player slots relative to the player to move (encoding.md section 1).

use polytopia_sim::engine::{Engine, NATURE};

pub const NATURE_SLOT: usize = 8;
pub const N_SLOTS: usize = 9;
pub const N_OTHER_SLOTS: usize = 7;

/// Player id per slot for the mover `me`, given the non-Nature player ids in turn order. Slot 0 is `me`, slots 1..7
/// follow in turn order, `0` marks a slot without a player; the Nature slot is not set here.
pub fn slot_ids(order: &[u8], me: u8) -> [u8; N_SLOTS] {
    let at = order.iter().position(|&id| id == me).unwrap_or(0);
    let mut ids = [0u8; N_SLOTS];
    for k in 0..order.len().min(N_SLOTS - 1) {
        ids[k] = order[(at + k) % order.len()];
    }
    ids
}

/// Maps engine player ids to slots and back. Slot 0 is the mover, 1..7 follow in turn order, 8 is Nature.
#[derive(Debug, Clone)]
pub struct SlotMap {
    pub me: u8,
    /// Player id per slot, `0` for a slot without a player.
    ids: [u8; N_SLOTS],
}

impl SlotMap {
    pub fn new(e: &Engine, me: u8) -> Self {
        let order: Vec<u8> = e.s.players.iter().map(|p| p.id).filter(|&id| id != NATURE).collect();
        let mut ids = slot_ids(&order, me);
        if e.s.players.iter().any(|p| p.id == NATURE) {
            ids[NATURE_SLOT] = NATURE;
        }
        Self { me, ids }
    }

    pub fn slot_of(&self, id: u8) -> Option<usize> {
        if id == 0 {
            return None;
        }
        self.ids.iter().position(|&x| x == id)
    }

    pub fn id_of(&self, slot: usize) -> Option<u8> {
        self.ids.get(slot).copied().filter(|&id| id != 0)
    }

    /// `0` for no player, else `1 + slot`: the value of a slot-valued tile column.
    pub fn tile_value(&self, id: u8) -> i16 {
        self.slot_of(id).map_or(0, |s| s as i16 + 1)
    }
}
