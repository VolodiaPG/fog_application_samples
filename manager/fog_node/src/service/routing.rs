use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::try_join_all;

use model::domain::routing::Packet;
use model::dto::routing::Direction;
use model::view::routing::{Route, RouteDirection, RouteLinking};
use model::{BidId, NodeId};
use openfaas::DefaultApi;

use crate::repository::faas_routing_table::FaaSRoutingTable;
use crate::repository::routing::Routing as RoutingRepository;
use crate::service::faas::FaaSBackend;
use crate::NodeSituation;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Routing(#[from] crate::repository::routing::Error),
    #[error("The next node doesn't exist: {0}")]
    NextNodeDoesntExist(NodeId),
    #[error("The next node isn't defined in the routing solution")]
    NextNodeIsNotDefined,
    #[error("The routing stack was malformed and could not be utilized")]
    MalformedRoutingStack,
    #[error("The bid id / function id is not known: {0}")]
    UnknownBidId(BidId),
    #[error("Failed to serialize: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error(transparent)]
    OpenFaas(#[from] openfaas::Error<String>),
}

/// Service to manage the behaviour of the routing
#[async_trait]
pub trait Router: Debug + Send + Sync {
    /// Start the process of registering a new route
    async fn register_function_route(&self, route: Route)
        -> Result<(), Error>;

    /// Link associations and do the follow-ups
    async fn route_linking(
        &self,
        linking: RouteLinking,
        read_only: bool,
    ) -> Result<(), Error>;

    /// Forward payloads to a neighbour node
    async fn forward(&self, packet: &Packet) -> Result<Bytes, Error>;
}

#[derive(Debug)]
pub struct RouterImpl<R>
where
    R: RoutingRepository,
{
    faas_routing_table: Arc<dyn FaaSRoutingTable>,
    node_situation:     Arc<dyn NodeSituation>,
    routing:            Arc<R>,
    faas:               Arc<dyn FaaSBackend>,
    faas_api:           Arc<dyn DefaultApi>,
}

impl<R> RouterImpl<R>
where
    R: RoutingRepository,
{
    pub fn new(
        faas_routing_table: Arc<dyn FaaSRoutingTable>,
        node_situation: Arc<dyn NodeSituation>,
        routing: Arc<R>,
        faas: Arc<dyn FaaSBackend>,
        faas_api: Arc<dyn DefaultApi>,
    ) -> Self {
        Self { faas_routing_table, node_situation, routing, faas, faas_api }
    }
}

#[async_trait]
impl<R> Router for RouterImpl<R>
where
    R: RoutingRepository,
{
    async fn register_function_route(
        &self,
        route: Route,
    ) -> Result<(), Error> {
        trace!("Linking upon requested route: {:?}", route);

        let mut parts = vec![];

        // Order is important, as stack_asc is the destination, so the last
        // redirection to be written if the case of a “^”-shaped branch comes
        // into consideration
        if !route.stack_rev.is_empty() {
            let stack = VecDeque::from(route.stack_rev);
            let prev_last_node = if let Some(prev) = stack.get(1) {
                prev.clone()
            } else {
                self.node_situation.get_my_id().await
            };

            let link = RouteLinking {
                direction: RouteDirection::FinishToStart { prev_last_node },
                stack,
                function: route.function.clone(),
            };

            parts.push(self.route_linking(link, true));
        }

        if !route.stack_asc.is_empty() {
            let link = RouteLinking {
                stack:     VecDeque::from(route.stack_asc),
                direction: RouteDirection::StartToFinish,
                function:  route.function.clone(),
            };

            parts.push(self.route_linking(link, false));
        } else {
            // We are the first node and are routing to ourselves
            self.faas_routing_table
                .update(route.function, Direction::CurrentNode)
                .await;
        }

        try_join_all(parts.into_iter()).await?;
        Ok(())
    }

    async fn route_linking(
        &self,
        mut linking: RouteLinking,
        readonly: bool,
    ) -> Result<(), Error> {
        trace!(
            "Linking route subpart (readonly: {:?}): {:?}",
            readonly,
            linking
        );

        let linking_empty_at_start = linking.stack.is_empty();
        let target = if linking_empty_at_start {
            match linking.direction {
                RouteDirection::StartToFinish => Direction::CurrentNode,
                RouteDirection::FinishToStart { ref prev_last_node } => {
                    Direction::NextNode(prev_last_node.clone())
                }
            }
        } else {
            let next = match linking.direction {
                RouteDirection::StartToFinish => linking.stack.pop_front(),
                RouteDirection::FinishToStart { .. } => {
                    linking.stack.pop_back()
                }
            }
            .ok_or(Error::NextNodeIsNotDefined)?;
            Direction::NextNode(next)
        };

        if !readonly {
            let target = match linking.direction {
                RouteDirection::StartToFinish => target.clone(),
                RouteDirection::FinishToStart { .. } => Direction::NextNode(
                    self.node_situation
                        .get_parent_id()
                        .await
                        .ok_or(Error::NextNodeIsNotDefined)?,
                ),
            };
            self.faas_routing_table
                .update(linking.function.clone(), target)
                .await;
        }

        if linking_empty_at_start {
            return Ok(());
        }

        if let Direction::NextNode(next) = target {
            trace!("Sending next step to node {:?}", next);

            self.forward(&Packet::FogNode {
                route_to_stack: vec![
                    next,
                    self.node_situation.get_my_id().await,
                ],
                resource_uri:   "route_linking".to_string(),
                data:           &serde_json::value::to_raw_value(&linking)?,
            })
            .await?;
        }

        Ok(())
    }

    async fn forward(&self, packet: &Packet) -> Result<Bytes, Error> {
        match packet {
            Packet::FaaSFunction { to, data: payload } => {
                let node_to = self
                    .faas_routing_table
                    .get(to)
                    .await
                    .ok_or_else(|| Error::UnknownBidId(to.to_owned()))?;

                match node_to {
                    // TODO: optimization: is it possible to send the packet
                    // directly to the node? w/o redoing the
                    // same structure, what impact?
                    Direction::NextNode(next) => {
                        let next = self
                            .node_situation
                            .get_fog_node_neighbor(&next)
                            .await
                            .ok_or_else(|| {
                                Error::NextNodeDoesntExist(next.to_owned())
                            })?;
                        Ok(self
                            .routing
                            .forward_to_routing(
                                &next.ip,
                                &next.port,
                                &Packet::FaaSFunction {
                                    to:   to.to_owned(),
                                    data: payload,
                                },
                            )
                            .await?)
                    }
                    Direction::CurrentNode => {
                        let record = self
                            .faas
                            .get_provisioned_function(to)
                            .await
                            .ok_or_else(|| {
                                Error::UnknownBidId(to.to_owned())
                            })?;
                        self.faas_api
                            .async_function_name_post(
                                &record.function_name,
                                serde_json::to_string(payload)?,
                            )
                            .await
                            .map_err(Error::from)?;
                        // TODO check if that doesn't cause any harm
                        Ok(Bytes::new())
                    }
                }
            }
            Packet::FogNode {
                route_to_stack: route_to,
                resource_uri,
                data,
            } => {
                let mut route_to = route_to.clone();
                let current_node =
                    route_to.pop().ok_or(Error::MalformedRoutingStack)?;

                if current_node != self.node_situation.get_my_id().await {
                    return Err(Error::MalformedRoutingStack);
                }

                if route_to.is_empty() {
                    let my_ip = self.node_situation.get_my_public_ip().await;
                    let my_port =
                        self.node_situation.get_my_public_port().await;

                    Ok(self
                        .routing
                        .forward_to_url(&my_ip, &my_port, resource_uri, data)
                        .await?)
                } else {
                    let next = route_to.last().unwrap();
                    let next = self
                        .node_situation
                        .get_fog_node_neighbor(next)
                        .await
                        .ok_or_else(|| {
                            Error::NextNodeDoesntExist(next.to_owned())
                        })?;
                    Ok(self
                        .routing
                        .forward_to_routing(
                            &next.ip,
                            &next.port,
                            &Packet::FogNode {
                                route_to_stack: route_to,
                                resource_uri: resource_uri.to_owned(),
                                data,
                            },
                        )
                        .await?)
                }
            }
            Packet::Market { resource_uri, data } => {
                if self.node_situation.is_market().await {
                    trace!(
                        "Transmitting market packet to market: {:?}",
                        packet
                    );
                    let (ip, port) = self
                        .node_situation
                        .get_market_node_address()
                        .await
                        .unwrap();
                    Ok(self
                        .routing
                        .forward_to_url(&ip, &port, resource_uri, data)
                        .await?)
                } else {
                    trace!(
                        "Transmitting market packet to other node: {:?}",
                        packet
                    );
                    let (ip, port) = self
                        .node_situation
                        .get_parent_node_address()
                        .await
                        .unwrap();
                    Ok(self
                        .routing
                        .forward_to_routing(&ip, &port, packet)
                        .await?)
                }
            }
        }
    }
}
