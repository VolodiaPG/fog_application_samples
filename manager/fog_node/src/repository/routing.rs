use std::fmt::Debug;
use std::net::IpAddr;
use std::sync::Arc;

use async_trait::async_trait;

use model::domain::routing::Packet;
use model::{FogNodeHTTPPort, MarketHTTPPort};
use reqwest::StatusCode;
use serde::Serialize;

use serde_json::Value;

#[cfg(feature = "jaeger")]
type HttpClient = reqwest_middleware::ClientWithMiddleware;
#[cfg(not(feature = "jaeger"))]
type HttpClient = reqwest::Client;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Error forwarding the payload: {0}")]
    Forwarding(#[from] reqwest::Error),
    #[cfg(feature = "jaeger")]
    #[error(transparent)]
    ReqwestMiddleware(#[from] reqwest_middleware::Error),
    #[error("Next node {0} answered with code {1}: {2}")]
    ForwardingResponse(String, StatusCode, String),
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
}

/// Behaviour of the routing
#[async_trait]
pub trait Routing: Debug + Sync + Send {
    /// Forward to the url to be handled by the routing service of the node
    // async fn forward_to_routing(
    //     &self,
    //     ip: &IpAddr,
    //     port: &FogNodeRPCPort,
    //     packet: &Packet,
    // ) -> Result<Value, Error>;

    /// Forward to the url to be handled by the routing service of the node
    async fn forward_to_routing(
        &self,
        ip: &IpAddr,
        port: &FogNodeHTTPPort,
        packet: &Packet,
    ) -> Result<Option<Value>, Error>;

    /// Forward to the url to be handled by arbitrary route, on another node
    async fn forward_to_fog_node_url<'a, 'b, T>(
        &self,
        node_ip: &IpAddr,
        node_port: &FogNodeHTTPPort,
        resource_uri: &'b str,
        data: &'a T,
    ) -> Result<Option<Value>, Error>
    where
        T: Serialize + Send + Sync;

    /// Forward to the url to be handled by arbitrary route, on the marke node
    async fn forward_to_market_url<'a, 'b, T>(
        &self,
        node_ip: &IpAddr,
        node_port: &MarketHTTPPort,
        resource_uri: &'b str,
        data: &'a T,
    ) -> Result<Option<Value>, Error>
    where
        T: Serialize + Send + Sync;
}

#[derive(Debug)]
pub struct RoutingImpl {
    client: Arc<HttpClient>,
}

impl RoutingImpl {
    pub fn new(client: Arc<HttpClient>) -> Self { Self { client } }

    async fn forward_to<'a, T>(
        &self,
        data: &'a T,
        full_url: &'a str,
    ) -> Result<Option<Value>, Error>
    where
        T: Serialize + Send + Sync,
    {
        let res = self.client.post(full_url).json(data).send().await?;

        if res.status().is_success() {
            Ok(res.json().await.ok())
        } else {
            Err(Error::ForwardingResponse(
                full_url.to_string(),
                res.status(),
                res.text().await.unwrap(),
            ))
        }
    }
}

#[async_trait]
impl Routing for RoutingImpl {
    // #[instrument(level = "trace", skip(self, packet))]
    // async fn forward_to_routing(
    //     &self,
    //     ip: &IpAddr,
    //     port: &FogNodeRPCPort,
    //     packet: &Packet,
    // ) -> Result<Value, Error> {
    //     let key = format!("{}:{}", ip, port);
    //     trace!("RPC-ing to routing on {}...", key);
    //     // let mut client = self.dialed_up.get(&key);

    //     // if client.is_none() {
    //     // let c = Client::<JsonCodec>::dial(&key).await.unwrap();
    //     // self.dialed_up.insert(key.clone(), c);
    //     // client = self.dialed_up.get(&key);
    //     // }
    //     let mut client = Client::<JsonCodec>::dial(&key).await.unwrap();

    //     let res = client.call("routing", packet).await.map_err(Error::from);
    //     client.shutdown().await;
    //     res
    // }

    #[instrument(level = "trace", skip(self, packet))]
    async fn forward_to_routing(
        &self,
        ip: &IpAddr,
        port: &FogNodeHTTPPort,
        packet: &Packet,
    ) -> Result<Option<Value>, Error> {
        let url = match packet {
            Packet::FaaSFunction { .. } => {
                format!("http://{}:{}/api/routing", ip, port)
            }
            Packet::FogNode { .. } | Packet::Market { .. } => {
                format!("http://{}:{}/api/sync-routing", ip, port)
            }
        };
        self.forward_to(packet, &url).await
    }

    #[instrument(level = "trace", skip(self, data))]
    async fn forward_to_fog_node_url<'a, 'b, T>(
        &self,
        node_ip: &IpAddr,
        node_port: &FogNodeHTTPPort,
        resource_uri: &'b str,
        data: &'a T,
    ) -> Result<Option<Value>, Error>
    where
        T: Serialize + Send + Sync,
    {
        let url =
            format!("http://{}:{}/api/{}", node_ip, node_port, resource_uri);
        trace!("Posting (forward) to {}", &url);
        self.forward_to(data, &url).await
    }

    #[instrument(level = "trace", skip(self, data))]
    async fn forward_to_market_url<'a, 'b, T>(
        &self,
        node_ip: &IpAddr,
        node_port: &MarketHTTPPort,
        resource_uri: &'b str,
        data: &'a T,
    ) -> Result<Option<Value>, Error>
    where
        T: Serialize + Send + Sync,
    {
        let url =
            format!("http://{}:{}/api/{}", node_ip, node_port, resource_uri);
        trace!("Posting (forward) to {}", &url);
        self.forward_to(data, &url).await
    }
}
