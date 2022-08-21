#![feature(future_join)]
#[macro_use]
extern crate log;
extern crate rocket;

use std::env;
use std::sync::Arc;

use rocket_okapi::openapi_get_routes;
use rocket_okapi::swagger_ui::*;

use crate::handler::*;
use crate::repository::fog_node::FogNodeImpl;

mod controller;
mod handler;
mod repository;
mod service;

async fn rocket() {
    let fog_node = Arc::new(FogNodeImpl::new());
    let fog_node_communication =
        Arc::new(crate::repository::node_communication::NodeCommunicationThroughRoutingImpl::new(
            fog_node.clone(),
        ));
    let auction_process =
        Arc::new(crate::repository::auction::SecondPriceAuction::new());

    // Services
    let auction_service = Arc::new(service::auction::AuctionImpl::new(
        auction_process,
        fog_node_communication.clone(),
    ));
    let fog_node_network_service =
        Arc::new(service::fog_node_network::FogNodeNetworkHashTreeImpl::new(
            fog_node.clone(),
        ));
    let faas_service = Arc::new(service::faas::FogNodeFaaSImpl::new(
        fog_node,
        fog_node_network_service.clone(),
        fog_node_communication,
    ));

    let _ = rocket::build()
        .manage(auction_service as Arc<dyn crate::service::auction::Auction>)
        .manage(
            fog_node_network_service
                as Arc<dyn crate::service::fog_node_network::FogNodeNetwork>,
        )
        .manage(faas_service as Arc<dyn crate::service::faas::FogNodeFaaS>)
        .mount(
            "/",
            make_swagger_ui(&SwaggerUIConfig {
                url: "/api/openapi.json".to_owned(),
                ..Default::default()
            }),
        )
        .mount(
            "/api/",
            openapi_get_routes![
                put_function,
                post_register_node,
                get_functions,
                get_fog,
                health
            ],
        )
        .ignite()
        .await
        .expect("Rocket failed to ignite")
        .launch()
        .await
        .expect("Rocket launch failed");
}

fn main() {
    env::set_var("RUST_LOG", "info, market=trace");
    env_logger::init();

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build runtime failed")
        .block_on(rocket());
}
