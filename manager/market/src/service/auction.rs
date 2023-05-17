use super::faas::FogNodeFaaS;
use super::fog_node_network::FogNodeNetwork;
use crate::repository::auction::Auction as AuctionRepository;
use crate::repository::node_communication::NodeCommunication;
use anyhow::{anyhow, bail, Context, Result};
use model::domain::auction::AuctionResult;
use model::domain::sla::Sla;
use model::dto::function::ChosenBid;
use model::dto::node::NodeRecord;
use model::view::auction::{AcceptedBid, BidProposals, InstanciatedBid};
use model::NodeId;
use std::sync::Arc;
use tokio::time::Instant;
use tracing::{info, trace};

pub struct Auction {
    auction_process:    Arc<AuctionRepository>,
    node_communication: Arc<NodeCommunication>,
    fog_node_network:   Arc<FogNodeNetwork>,
    faas:               Arc<FogNodeFaaS>,
}

impl Auction {
    pub fn new(
        auction_process: Arc<AuctionRepository>,
        node_communication: Arc<NodeCommunication>,
        fog_node_network: Arc<FogNodeNetwork>,
        faas: Arc<FogNodeFaaS>,
    ) -> Self {
        Self { auction_process, node_communication, fog_node_network, faas }
    }

    async fn call_for_bids(
        &self,
        to: NodeId,
        sla: &'_ Sla,
    ) -> Result<BidProposals> {
        trace!("call for bids: {:?}", sla);

        self.node_communication
            .request_bids_from_node(to.clone(), sla)
            .await
            .with_context(|| format!("Failed to get bids from {}", to))
    }

    async fn do_auction(
        &self,
        proposals: &BidProposals,
    ) -> Result<AuctionResult> {
        trace!("do auction: {:?}", proposals);
        let auction_result =
            self.auction_process.auction(&proposals.bids).ok_or_else(
                || anyhow!("Auction failed, no winners were selected"),
            )?;
        Ok(AuctionResult { chosen_bid: auction_result })
    }

    async fn process_provisioning_details(
        &self,
        proposals: BidProposals,
        chosen_bid: ChosenBid,
        sla: Sla,
    ) -> Result<AcceptedBid> {
        let NodeRecord { ip, port_faas, .. } = self
            .fog_node_network
            .get_node(&chosen_bid.bid.node_id)
            .await
            .ok_or_else(|| {
                anyhow!(
                    "Node record of {} is not present in my database",
                    chosen_bid.bid.node_id
                )
            })?;
        let accepted = AcceptedBid {
            chosen: InstanciatedBid {
                bid: chosen_bid.bid,
                price: chosen_bid.price,
                ip,
                port: port_faas,
            },
            proposals,
            sla,
        };

        self.faas
            .provision_function(accepted.clone())
            .await
            .context("Failed to provision function")?;

        Ok(accepted)
    }

    pub async fn start_auction(
        &self,
        target_node: NodeId,
        sla: Sla,
    ) -> Result<AcceptedBid> {
        let started = Instant::now();

        let proposals = self
            .call_for_bids(target_node.clone(), &sla)
            .await
            .with_context(|| {
                format!(
                    "Failed to call the network for bids with node {} as the \
                     starting point",
                    target_node,
                )
            })?;

        let AuctionResult { chosen_bid } =
            self.do_auction(&proposals).await.context("Auction failed")?;

        let res = self
            .process_provisioning_details(
                proposals,
                chosen_bid.clone(),
                sla.clone(),
            )
            .await;

        if let Ok(accepted) = res {
            let finished = Instant::now();

            let duration = finished - started;

            crate::prom_metrics::FUNCTION_DEPLOYMENT_TIME_GAUGE
                .with_label_values(&[
                    &sla.function_live_name,
                    &chosen_bid.bid.id.to_string(),
                    &sla.id.to_string(),
                ])
                .set(duration.as_millis() as f64 / 1000.0);

            return Ok(accepted);
        }

        info!("Provisioning a function failed, retrying and bidding again...");
        bail!("Failed to provision function after several retries.")
    }
}
