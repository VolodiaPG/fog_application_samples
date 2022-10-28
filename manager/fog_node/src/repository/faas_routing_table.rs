use std::fmt::Debug;

use async_trait::async_trait;

use model::dto::routing::Direction;
use model::BidId;

#[async_trait]
pub trait FaaSRoutingTable: Debug + Sync + Send {
    /// Update the breadcrumb route to the [BidId] passing by the next
    /// [NodeId].
    async fn update(&self, source: BidId, target: Direction);

    async fn get(&self, bid_id: &BidId) -> Option<Direction>;
}

#[derive(Debug)]
pub struct FaaSRoutingTableHashMap {
    table: flurry::HashMap<BidId, Direction>,
}

impl FaaSRoutingTableHashMap {
    pub fn new() -> Self { Self { table: flurry::HashMap::new() } }
}

#[async_trait]
impl FaaSRoutingTable for FaaSRoutingTableHashMap {
    async fn update(&self, source: BidId, target: Direction) {
        trace!("Updating routing table {} -> {:?}", source, target);
        self.table.pin().insert(source, target);
    }

    async fn get(&self, bid_id: &BidId) -> Option<Direction> {
        self.table.pin().get(bid_id).map(|x| x.clone())
    }
}
