use chrono::{DateTime, Utc};
use helper::uom_helper::information_rate;
#[cfg(feature = "offline")]
use helper::uom_helper::time;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::collections::HashMap;
use std::net::IpAddr;
#[cfg(feature = "offline")]
use uom::si::f64::Time;
use uom::si::rational64::InformationRate;

use crate::dto::node::NodeRecord;
use crate::view::auction::AcceptedBid;
use crate::{BidId, FogNodeFaaSPortExternal, FogNodeHTTPPort};
use helper::chrono as chrono_helper;

use super::super::NodeId;

/// Update information in the node node
/// - Must indicate the time the request was createdAt in order to update the
///   rolling average
/// - Subsequent requests will also include the last_answered_at time, returned
///   in [PostNodeResponse]
/// - Same for last_answered_at
#[serde_with::serde_as]
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PostNode {
    #[serde_as(as = "chrono_helper::DateTimeHelper")]
    pub created_at: DateTime<Utc>,

    #[serde_as(as = "Option<chrono_helper::DateTimeHelper>")]
    #[serde(default)]
    pub last_answered_at: Option<DateTime<Utc>>,

    #[serde_as(as = "Option<chrono_helper::DateTimeHelper>")]
    #[serde(default)]
    pub last_answer_received_at: Option<DateTime<Utc>>,

    pub from: NodeId,
}

/// The answer to [PostNode]
#[serde_with::serde_as]
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PostNodeResponse {
    #[serde_as(as = "chrono_helper::DateTimeHelper")]
    pub answered_at: DateTime<Utc>,
}

#[serde_with::serde_as]
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub enum RegisterNode {
    MarketNode {
        node_id:              NodeId,
        ip:                   IpAddr,
        port_http:            FogNodeHTTPPort,
        port_faas:            FogNodeFaaSPortExternal,
        tags:                 Vec<String>,
        #[serde_as(as = "information_rate::Helper")]
        advertised_bandwidth: InformationRate,
    },
    Node {
        parent:               NodeId,
        node_id:              NodeId,
        ip:                   IpAddr,
        port_http:            FogNodeHTTPPort,
        port_faas:            FogNodeFaaSPortExternal,
        tags:                 Vec<String>,
        #[serde_as(as = "information_rate::Helper")]
        advertised_bandwidth: InformationRate,
        #[cfg(feature = "offline")]
        #[serde_as(as = "time::Helper")]
        offline_latency:      Time,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GetFogNodes {
    pub id:            NodeId,
    pub ip:            IpAddr,
    pub tags:          Vec<String>,
    pub accepted_bids: HashMap<BidId, AcceptedBid>,
}

impl From<(NodeId, NodeRecord)> for GetFogNodes {
    fn from((id, record): (NodeId, NodeRecord)) -> Self {
        GetFogNodes {
            id,
            ip: record.ip,
            tags: record.tags,
            accepted_bids: record.accepted_bids,
        }
    }
}
