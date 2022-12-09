#[derive(thiserror::Error, Debug)]
pub enum ControllerError {
    #[error(transparent)]
    OpenFaaS(#[from] crate::service::faas::Error),
    #[error(transparent)]
    Auction(#[from] crate::service::auction::Error),
    #[error(transparent)]
    Function(#[from] crate::service::function_life::Error),
    #[error(transparent)]
    Router(#[from] crate::service::routing::Error),
    #[error(transparent)]
    NodeLife(#[from] crate::service::node_life::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub(crate) mod auction;
pub(crate) mod node;
pub(crate) mod routing;
