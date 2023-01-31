use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;

use bytes::Bytes;
use model::dto::auction::BidRecord;
use model::dto::faas::ProvisionedRecord;
use model::BidId;
use openfaas::models::{FunctionDefinition, Limits};
use openfaas::{DefaultApi, DefaultApiClient};

use crate::repository::provisioned::Provisioned as ProvisionedRepository;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    OpenFaaS(#[from] openfaas::Error<String>),
    #[error("BidId {0} not found")]
    IdNotFound(BidId),
}

#[async_trait]
pub trait FaaSBackend: Debug + Sync + Send {
    /// Provision the function from the bid description
    /// Return the function's name
    async fn provision_function(
        &self,
        id: BidId,
        bid: BidRecord,
    ) -> Result<String, Error>;

    async fn get_metrics(&self, id: &BidId) -> Result<Option<Bytes>, Error>;

    fn get_provisioned_function(
        &self,
        id: &BidId,
    ) -> Option<ProvisionedRecord>;

    fn get_provisioned_functions(&self) -> Vec<BidId>;
}

#[derive(Debug)]
pub struct OpenFaaSBackend {
    client:                Arc<DefaultApiClient>,
    provisioned_functions: Arc<dyn ProvisionedRepository>,
}

impl OpenFaaSBackend {
    pub fn new(
        client: Arc<DefaultApiClient>,
        provisioned_functions: Arc<dyn ProvisionedRepository>,
    ) -> Self {
        Self { client, provisioned_functions }
    }
}

#[async_trait]
impl FaaSBackend for OpenFaaSBackend {
    async fn provision_function(
        &self,
        id: BidId,
        bid: BidRecord,
    ) -> Result<String, Error> {
        let function_name = format!("{}-{}", bid.sla.function_live_name, id);

        let definition = FunctionDefinition {
            image: bid.sla.function_image.to_owned(),
            service: function_name.to_owned(),
            limits: Some(Limits {
                memory: bid.sla.memory,
                cpu:    bid.sla.cpu,
            }),
            ..Default::default()
        };

        self.client.system_functions_post(definition).await?;

        self.provisioned_functions.insert(
            id,
            ProvisionedRecord { bid, function_name: function_name.to_owned() },
        );

        Ok(function_name)
    }

    async fn get_metrics(&self, id: &BidId) -> Result<Option<Bytes>, Error> {
        let record = self
            .get_provisioned_function(id)
            .ok_or_else(|| Error::IdNotFound(id.clone()))?;

        Ok(self.client.functions_get_metrics(&record.function_name).await?)
    }

    fn get_provisioned_function(
        &self,
        id: &BidId,
    ) -> Option<ProvisionedRecord> {
        self.provisioned_functions.get(id)
    }

    fn get_provisioned_functions(&self) -> Vec<BidId> {
        self.provisioned_functions.get_all()
    }
}
