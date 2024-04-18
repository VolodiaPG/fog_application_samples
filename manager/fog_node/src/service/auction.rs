use crate::monitoring::BidGauge;
use crate::repository::function_tracking::FunctionTracking;
use crate::repository::resource_tracking::ResourceTracking;
use anyhow::{Context, Result};
use chrono::Utc;
use helper::env_var;
use helper::monitoring::MetricsExporter;
use model::domain::sla::Sla;
use model::dto::function::Proposed;
use model::view::auction::AccumulatedLatency;
use model::BidId;
use nutype::nutype;
use std::sync::Arc;
use uom::si::f64::{Information, Ratio};
use uuid::Uuid;

use super::function::Function;

#[nutype(
    derive(PartialEq, PartialOrd),
    validate(finite, greater_or_equal = 0.0)
)]
pub struct PricingRatio(f64);

env_var!(PRICING_CPU);
env_var!(PRICING_CPU_INITIAL);
env_var!(PRICING_MEM);
env_var!(PRICING_MEM_INITIAL);
env_var!(PRICING_GEOLOCATION);
env_var!(RATIO_AA);
env_var!(RATIO_BB);
env_var!(RATIO_CC);
env_var!(ELECTRICITY_PRICE);

struct ComputedBid {
    pub(crate) name:          String,
    #[allow(dead_code)]
    pub(crate) available_ram: Information,
    #[allow(dead_code)]
    pub(crate) available_cpu: Ratio,
    #[allow(dead_code)]
    pub(crate) used_ram:      Information,
    #[allow(dead_code)]
    pub(crate) used_cpu:      Ratio,
    pub(crate) bid:           f64,
}

pub struct Auction {
    resource_tracking: Arc<ResourceTracking>,
    db:                Arc<FunctionTracking>,
    metrics:           Arc<MetricsExporter>,
    #[allow(dead_code)]
    function:          Arc<Function>,
}

impl Auction {
    pub fn new(
        resource_tracking: Arc<ResourceTracking>,
        db: Arc<FunctionTracking>,
        metrics: Arc<MetricsExporter>,
        function: Arc<Function>,
    ) -> Self {
        Self { resource_tracking, db, metrics, function }
    }

    /// Get a suitable (free enough) node to potentially run the designated SLA
    async fn get_a_node(
        &self,
        sla: &Sla,
    ) -> Result<Option<(String, Information, Ratio, Information, Ratio)>> {
        for node in self.resource_tracking.get_nodes() {
            let (used_ram, used_cpu) =
                self.resource_tracking.get_used(node).await.with_context(
                    || {
                        format!(
                            "Failed to get used resources from tracking data \
                             for node {}",
                            node
                        )
                    },
                )?;
            let (available_ram, available_cpu) = self
                .resource_tracking
                .get_available(node)
                .await
                .with_context(|| {
                    format!(
                        "Failed to get available resources from tracking \
                         data for node {}",
                        node
                    )
                })?;
            if super::function::satisfiability_check(
                &used_ram,
                &used_cpu,
                &available_ram,
                &available_cpu,
                sla,
            ) {
                return Ok(Some((
                    node.clone(),
                    used_ram,
                    used_cpu,
                    available_ram,
                    available_cpu,
                )));
            }
        }
        Ok(None)
    }

    #[cfg(feature = "linear_rates")]
    async fn compute_bid(
        &self,
        sla: &Sla,
        _accumulated_latency: &AccumulatedLatency,
    ) -> Result<Option<ComputedBid>> {
        use helper::env_load;

        // let pricing_cpu =
        //     env_load!(PricingRatio, PRICING_CPU, f64).into_inner();
        let pricing_cpu_initial =
            env_load!(PricingRatio, PRICING_CPU_INITIAL, f64).into_inner();
        // let pricing_mem =
        //     env_load!(PricingRatio, PRICING_MEM, f64).into_inner();
        let pricing_mem_initial =
            env_load!(PricingRatio, PRICING_MEM_INITIAL, f64).into_inner();

        let Some((name, used_ram, used_cpu, available_ram, available_cpu)) =
            self.get_a_node(sla)
                .await
                .context("Failed to found a suitable node for the sla")?
        else {
            return Ok(None);
        };

        let ram_ratio_sla: f64 = (sla.memory / available_ram).into();
        let cpu_ratio_sla: f64 = (sla.cpu / available_cpu).into();
        // let ram_ratio: f64 = ((used_ram + sla.memory) /
        // available_ram).into(); let cpu_ratio: f64 = ((used_cpu +
        // sla.cpu) / available_cpu).into();
        let bid: f64 = ram_ratio_sla * pricing_mem_initial
            + cpu_ratio_sla * pricing_cpu_initial;

        trace!("price on {:?} is {:?}", name, bid);

        Ok(Some(ComputedBid {
            name,
            bid,
            available_cpu,
            available_ram,
            used_cpu,
            used_ram,
        }))
    }

    #[cfg(feature = "quadratic_rates")]
    async fn compute_bid(
        &self,
        sla: &Sla,
        _accumulated_latency: &AccumulatedLatency,
    ) -> Result<Option<ComputedBid>> {
        use crate::service::function::UnprovisionEvent;
        use chrono::Duration;
        use helper::env_load;
        use helper::uom_helper::cpu_ratio::cpu;
        use uom::si::time::second;

        let Some((name, used_ram, used_cpu, available_ram, available_cpu)) =
            self.get_a_node(sla)
                .await
                .context("Failed to found a suitable node for the sla")?
        else {
            return Ok(None);
        };

        let aa = env_load!(PricingRatio, RATIO_AA, f64).into_inner();
        let bb = env_load!(PricingRatio, RATIO_BB, f64).into_inner();
        let now = Utc::now();
        let mut utilisation = 0.0;
        for UnprovisionEvent { timestamp, sla, node, .. } in
            self.function.get_utilisation_variations().await.iter()
        {
            let duration = if *timestamp > now {
                *timestamp - now
            } else {
                Duration::microseconds(0)
            };
            let (_available_ram, available_cpu) = self
                .resource_tracking
                .get_available(node)
                .await
                .with_context(|| {
                    format!(
                        "Failed to get available resources from tracking \
                         data for node {}",
                        node
                    )
                })?;

            let duration = duration.num_seconds() as f64;
            utilisation += (sla.cpu / available_cpu).get::<cpu>() * duration;
        }
        let sla_cpu = (sla.cpu / available_cpu).get::<cpu>();
        let sla_duration = sla.duration.get::<second>();
        let electricity_price =
            env_load!(PricingRatio, ELECTRICITY_PRICE, f64).into_inner();
        let bid = electricity_price
            * sla_cpu
            * (2.0 * aa * utilisation + (aa * sla_cpu + bb) * sla_duration);

        trace!("(quadratic) price on is {:?}", bid);

        Ok(Some(ComputedBid {
            name,
            bid,
            available_cpu,
            available_ram,
            used_cpu,
            used_ram,
        }))
    }

    #[cfg(feature = "maxcpu")]
    async fn compute_bid_maxcpu(
        &self,
        sla: &Sla,
        accumulated_latency: &AccumulatedLatency,
    ) -> Result<Option<(String, f64, f64)>> {
        match self.compute_bid(sla, accumulated_latency).await? {
            Some(computed) => {
                // The more the cpu is used the lower the price and the easiest
                // to win
                let cpu_ratio_sla: f64 =
                    (computed.used_cpu / computed.available_cpu).into();
                let bid: f64 = cpu_ratio_sla;
                // The normal "bid" is also the price usually, but not in that
                // valuation method
                let price = computed.bid;

                trace!("(random) price on {:?} is {:?}", computed.name, bid);
                Ok(Some((computed.name, bid, price)))
            }
            None => Ok(None),
        }
    }

    pub async fn bid_on(
        &self,
        sla: Sla,
        accumulated_latency: &AccumulatedLatency,
    ) -> Result<Option<(BidId, Proposed)>> {
        #[cfg(not(feature = "maxcpu"))]
        let Some(ComputedBid { name, bid, .. }) = self
            .compute_bid(&sla, accumulated_latency)
            .await
            .context("Failed to compute bid for sla")?
        else {
            return Ok(None);
        };
        #[cfg(not(feature = "maxcpu"))]
        let price = bid;

        #[cfg(feature = "maxcpu")]
        let Some((name, bid, price)) = self
            .compute_bid_maxcpu(&sla, accumulated_latency)
            .await
            .context("Failed to compute bid for sla")?
        else {
            return Ok(None);
        };
        let node = name;
        let record = Proposed::new(bid, sla, node);
        self.db.insert(record.clone());
        let id = Uuid::new_v4();
        let id = BidId::from(id);
        self.metrics
            .observe(BidGauge {
                bid,
                price,
                function_name: record.sla.function_live_name.clone(),
                sla_id: record.sla.id.to_string(),
                bid_id: id.to_string(),
                timestamp: Utc::now(),
            })
            .await?;
        Ok(Some((id, record)))
    }
}
