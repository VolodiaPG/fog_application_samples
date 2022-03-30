use std::collections::HashMap;

use crate::models::{BidId, BidRecord, ProvisionedRecord};

pub struct BidDataBase {
    database: HashMap<BidId, BidRecord>,
}

pub struct ProvisionedDataBase {
    database: HashMap<BidId, ProvisionedRecord>,
}

impl BidDataBase {
    pub fn new() -> Self {
        BidDataBase {
            database: HashMap::new(),
        }
    }

    pub fn insert(&mut self, bid: BidRecord) -> BidId {
        let uuid = BidId::new_v4();
        self.database.insert(uuid, bid);
        uuid
    }

    pub fn get(&self, id: &BidId) -> Option<&BidRecord> {
        self.database.get(id)
    }

    pub fn remove(&mut self, id: &BidId) {
        self.database.remove(id);
    }
}

impl ProvisionedDataBase {
    pub fn new() -> Self {
        ProvisionedDataBase {
            database: HashMap::new(),
        }
    }

    pub fn insert(&mut self, bid_id: BidId, bid: ProvisionedRecord) {
        self.database.insert(bid_id, bid);
    }

    pub fn get(&self, id: &BidId) -> Option<&ProvisionedRecord> {
        self.database.get(id)
    }
}
