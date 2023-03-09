use crate::service::function_life::FunctionLife;
use anyhow::{Context, Result};
use model::view::auction::{BidProposals, BidRequestOwned};
use model::BidId;
use std::sync::Arc;

/// Return a bid for the SLA. And makes the follow up to ask other nodes for
/// their bids.
pub async fn bid_on(
    bid_request: BidRequestOwned,
    function: &Arc<FunctionLife>,
) -> Result<BidProposals> {
    trace!("bidding on... {:?}", bid_request);
    function
        .bid_on_new_function_and_transmit(
            &bid_request.sla,
            bid_request.node_origin,
            bid_request.accumulated_latency,
        )
        .await
        .context("Failed to bid on function and transmit it to neighbors")
}

/// Returns a bid for the SLA.
/// Creates the function on OpenFaaS and use the SLA to enable the limits
pub async fn provision_from_bid(
    id: BidId,
    function: &Arc<FunctionLife>,
) -> Result<()> {
    trace!("Transforming bid into provisioned resource {:?}", id);
    function.provision_function(id.clone()).await.with_context(|| {
        format!("Failed to provision function from bid {}", id)
    })?;
    Ok(())
}
