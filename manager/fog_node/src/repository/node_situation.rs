use std::fmt::Debug;
use std::net::IpAddr;

use model::dto::node::NodeCategory::{MarketConnected, NodeConnected};
use model::dto::node::{NodeDescription, NodeSituationData};
use model::{FogNodeHTTPPort, FogNodeRPCPort, MarketHTTPPort, NodeId};

pub trait NodeSituation: Debug + Sync + Send {
    fn register(&self, id: NodeId, description: NodeDescription);
    /// Get a node: children, parent
    fn get_fog_node_neighbor(&self, id: &NodeId) -> Option<NodeDescription>;
    fn get_my_id(&self) -> NodeId;
    fn get_parent_id(&self) -> Option<NodeId>;
    fn get_my_tags(&self) -> Vec<String>;
    /// Whether the node is connected to the market (i.e., doesn't have any
    /// parent = root of the network)
    fn is_market(&self) -> bool;
    fn get_parent_node_address(
        &self,
    ) -> Option<(IpAddr, FogNodeHTTPPort, FogNodeRPCPort)>;
    fn get_market_node_address(&self) -> Option<(IpAddr, MarketHTTPPort)>;
    /// Return iter over both the parent and the children node...
    /// Aka all the nodes interesting that can accommodate a function
    fn get_neighbors(&self) -> Vec<NodeId>;
    /// Get the public ip associated with this server
    fn get_my_public_ip(&self) -> IpAddr;
    /// Get the public port associated with this server (HTTP)
    fn get_my_public_port_http(&self) -> FogNodeHTTPPort;
    /// Get the public port associated with this server (RPC)
    fn get_my_public_port_rpc(&self) -> FogNodeRPCPort;
}

#[derive(Debug)]
pub struct NodeSituationHashSetImpl {
    database: NodeSituationData,
}

impl NodeSituationHashSetImpl {
    pub fn new(situation: NodeSituationData) -> Self {
        Self { database: situation }
    }
}

impl NodeSituation for NodeSituationHashSetImpl {
    fn register(&self, id: NodeId, description: NodeDescription) {
        self.database.children.insert(id, description);
    }

    fn get_fog_node_neighbor(&self, id: &NodeId) -> Option<NodeDescription> {
        let ret = self.database.children.get(id);
        if ret.is_none() {
            match &self.database.situation {
                NodeConnected {
                    parent_node_ip,
                    parent_node_port_http,
                    parent_node_port_rpc,
                    parent_id,
                    ..
                } => {
                    if parent_id == id {
                        return Some(NodeDescription {
                            ip:        *parent_node_ip,
                            port_http: parent_node_port_http.clone(),
                            port_rpc:  parent_node_port_rpc.clone(),
                        });
                    }
                }
                MarketConnected { .. } => (),
            }
        }
        ret.map(|x| x.clone())
    }

    fn get_my_id(&self) -> NodeId { self.database.my_id.clone() }

    fn get_parent_id(&self) -> Option<NodeId> {
        match &self.database.situation {
            NodeConnected { parent_id, .. } => Some(parent_id.clone()),
            _ => None,
        }
    }

    fn get_my_tags(&self) -> Vec<String> { self.database.tags.clone() }

    fn is_market(&self) -> bool {
        matches!(self.database.situation, MarketConnected { .. })
    }

    fn get_parent_node_address(
        &self,
    ) -> Option<(IpAddr, FogNodeHTTPPort, FogNodeRPCPort)> {
        match &self.database.situation {
            NodeConnected {
                parent_node_ip,
                parent_node_port_http,
                parent_node_port_rpc,
                ..
            } => Some((
                *parent_node_ip,
                parent_node_port_http.clone(),
                parent_node_port_rpc.clone(),
            )),
            _ => None,
        }
    }

    fn get_market_node_address(&self) -> Option<(IpAddr, MarketHTTPPort)> {
        match &self.database.situation {
            MarketConnected { market_ip, market_port, .. } => {
                Some((*market_ip, market_port.clone()))
            }
            _ => None,
        }
    }

    fn get_neighbors(&self) -> Vec<NodeId> {
        let mut ret: Vec<NodeId> =
            self.database.children.iter().map(|x| x.key().clone()).collect();

        match &self.database.situation {
            NodeConnected { parent_id, .. } => {
                ret.push(parent_id.clone());
            }
            MarketConnected { .. } => (),
        }

        ret
    }

    fn get_my_public_ip(&self) -> IpAddr { self.database.my_public_ip }

    fn get_my_public_port_http(&self) -> FogNodeHTTPPort {
        self.database.my_public_port_http.clone()
    }

    fn get_my_public_port_rpc(&self) -> FogNodeRPCPort {
        self.database.my_public_port_rpc.clone()
    }
}
