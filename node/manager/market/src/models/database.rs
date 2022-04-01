use std::{cmp::Ordering, collections::HashMap};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use sla::Sla;
use uom::si::f64::Time;

use super::{BidId, NodeId, NodeRecordDisk};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BidRecord {
    pub bids: Vec<BidProposal>,
    pub sla: Sla,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BidProposal {
    pub node_id: NodeId,
    pub id: BidId,
    pub bid: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct NodeRecord {
    pub ip: String,
    pub latency: RollingAvg,
    pub accepted_bids: HashMap<BidId, AcceptedBid>, // TODO change name
}

impl From<NodeRecordDisk> for NodeRecord {
    fn from(disk: NodeRecordDisk) -> Self {
        NodeRecord {
            ip: disk.ip,
            ..Default::default()
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AcceptedBid {
    pub sla: Sla,
    pub bid: BidProposal,
}

#[serde_with::serde_as]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RollingAvg {
    #[serde_as(as = "super::DateTimeHelper")]
    last_update: DateTime<Utc>,
    #[serde_as(as = "sla::uom_time::Helper")]
    avg: Time,
    count: u32,
}

impl RollingAvg {
    pub fn update(&mut self, now: DateTime<Utc>, emission: DateTime<Utc>) {
        let latency =
            Time::new::<uom::si::time::millisecond>((now - emission).num_milliseconds() as f64);

        self.count += 1;
        self.avg = (latency + self.avg * ((self.count - 1) as f64)) / self.count as f64;

        self.last_update = now;
    }

    pub fn get_avg(&self) -> Time {
        self.avg
    }

    pub fn get_last_update(&self) -> DateTime<Utc> {
        self.last_update
    }
}

impl Default for RollingAvg {
    fn default() -> Self {
        RollingAvg {
            last_update: Utc::now(),
            avg: Time::new::<uom::si::time::millisecond>(0.0),
            count: 0,
        }
    }
}

impl PartialOrd for BidProposal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.bid.partial_cmp(&other.bid)
    }
}

impl PartialEq for BidProposal {
    fn eq(&self, other: &Self) -> bool {
        self.bid == other.bid
    }
}

impl Eq for BidProposal {}
