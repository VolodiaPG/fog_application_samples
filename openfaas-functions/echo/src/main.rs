#![feature(async_closure)]

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use actix_web::dev::Service;
use actix_web::{
    middleware, web, App, HttpMessage, HttpRequest, HttpResponse, HttpServer,
};
#[cfg(feature = "jaeger")]
use actix_web_opentelemetry::RequestTracing;
use anyhow::{Context, Result};
use chrono::serde::ts_microseconds;
use chrono::{DateTime, Utc};
use helper::monitoring::{
    InfluxAddress, InfluxBucket, InfluxOrg, InfluxToken, InstanceName,
    MetricsExporter,
};
use helper::{env_load, env_var};
use helper_derive::influx_observation;
use model::domain::sla::Sla;
#[cfg(feature = "jaeger")]
use opentelemetry::global;
#[cfg(feature = "jaeger")]
use opentelemetry::sdk::propagation::TraceContextPropagator;
use reqwest_middleware::ClientBuilder;
#[cfg(feature = "jaeger")]
use reqwest_tracing::TracingMiddleware;
use serde::Deserialize;
use std::sync::Arc;
use tracing::subscriber::set_global_default;
use tracing::{debug, warn, Subscriber};
#[cfg(feature = "jaeger")]
use tracing_actix_web::TracingLogger;
use tracing_forest::ForestLayer;
use tracing_log::LogTracer;
use tracing_subscriber::prelude::__tracing_subscriber_SubscriberExt;
use tracing_subscriber::{EnvFilter, Registry};

const VAR_SLA: &str = "SLA";
env_var!(INFLUX_ADDRESS);
env_var!(INFLUX_TOKEN);
env_var!(INFLUX_ORG);
env_var!(INFLUX_BUCKET);
env_var!(TOTO);

/// timestamp at which the function is fully operational
#[influx_observation]
struct FinishedBooting {
    #[influxdb(field)]
    value:  f64,
    #[influxdb(tag)]
    sla_id: String,
    #[influxdb(tag)]
    toto:   String,
}
/// SLA that passed here
#[influx_observation]
struct Latency {
    #[influxdb(field)]
    value:  f64,
    #[influxdb(tag)]
    sla_id: String,
    #[influxdb(tag)]
    tag:    String,
}
/// SLA that passed here
#[influx_observation]
struct LatencyHeader {
    #[influxdb(field)]
    value:  f64,
    #[influxdb(tag)]
    sla_id: String,
    #[influxdb(tag)]
    tag:    String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct IncomingPayload {
    #[serde(with = "ts_microseconds")]
    sent_at: DateTime<Utc>,
    tag:     String,
}

async fn handle(
    req: HttpRequest,
    payload: web::Json<IncomingPayload>,
    metrics: web::Data<MetricsExporter>,
    sla_id: web::Data<String>,
) -> HttpResponse {
    let first_byte_received =
        *req.extensions().get::<DateTime<Utc>>().unwrap();

    let data = &payload.0;

    let elapsed = first_byte_received - data.sent_at;

    if let Err(err) = metrics
        .observe(Latency {
            value:     elapsed.num_milliseconds().abs() as f64 / 1000.0,
            sla_id:    sla_id.to_string(),
            tag:       payload.tag.clone(),
            timestamp: first_byte_received,
        })
        .await
    {
        warn!("Failed to save metrics: {:?}", err);
        return HttpResponse::InternalServerError()
            .body(format!("Failed to save metrics: {:?}", err));
    }

    let proxy_timestamp = req.headers().get("Proxy-Timestamp");

    if let Some(timestamp) = proxy_timestamp {
        let timestamp = timestamp.to_str();
        if let Ok(timestamp) = timestamp {
            if let Ok(timestamp) = timestamp.parse::<i64>() {
                let elapsed =
                    first_byte_received.timestamp_millis() - timestamp;

                if let Err(err) = metrics
                    .observe(LatencyHeader {
                        value:     elapsed as f64 / 1000.0,
                        sla_id:    sla_id.to_string(),
                        tag:       payload.tag.clone(),
                        timestamp: first_byte_received,
                    })
                    .await
                {
                    warn!("Failed to save metrics: {:?}", err);
                    return HttpResponse::InternalServerError()
                        .body(format!("Failed to save metrics: {:?}", err));
                }
            }
        }
    }
    HttpResponse::Ok().finish()
}

/// Compose multiple layers into a `tracing`'s subscriber.
pub fn get_subscriber(
    _name: String,
    env_filter: String,
) -> impl Subscriber + Send + Sync {
    // Env variable LOG_CONFIG_PATH points at the path where

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(env_filter));

    #[cfg(feature = "jaeger")]
    let collector_ip = std::env::var("COLLECTOR_IP")
        .unwrap_or_else(|_| "localhost".to_string());
    #[cfg(feature = "jaeger")]
    let collector_port = std::env::var("COLLECTOR_PORT")
        .unwrap_or_else(|_| "14268".to_string());
    #[cfg(feature = "jaeger")]
    let tracing_leyer = tracing_opentelemetry::OpenTelemetryLayer::new(
        opentelemetry_jaeger::new_collector_pipeline()
            .with_endpoint(format!(
                "http://{}:{}/api/traces",
                collector_ip, collector_port
            ))
            .with_reqwest()
            .with_service_name(_name)
            .install_batch(opentelemetry::runtime::Tokio)
            .unwrap(),
    );

    let reg = Registry::default().with(env_filter);

    #[cfg(feature = "jaeger")]
    let reg = reg.with(tracing_leyer);

    reg.with(ForestLayer::default())
}

/// Register a subscriber as global default to process span data.
///
/// It should only be called once!
pub fn init_subscriber(subscriber: impl Subscriber + Send + Sync) {
    LogTracer::init().expect("Failed to set logger");
    set_global_default(subscriber).expect("Failed to set subscriber");
}

#[tokio::main]
async fn main() -> Result<()> {
    #[cfg(feature = "jaeger")]
    global::set_text_map_propagator(TraceContextPropagator::new());

    let subscriber = get_subscriber("market".into(), "trace".into());
    init_subscriber(subscriber);

    debug!("Tracing initialized.");

    #[cfg(feature = "jaeger")]
    let http_client = Arc::new(
        ClientBuilder::new(reqwest::Client::new())
            .with(TracingMiddleware::default())
            .build(),
    );
    #[cfg(not(feature = "jaeger"))]
    let http_client =
        Arc::new(ClientBuilder::new(reqwest::Client::new()).build());

    let Ok(sla_raw) = std::env::var(VAR_SLA) else {
        panic!("{} env variable not found", VAR_SLA);
    };

    let Ok(sla) = serde_json::from_str::<Sla>(&sla_raw) else {
        panic!("Cannot read and deserialize {} env variable", VAR_SLA);
    };

    let metrics = Arc::new(
        MetricsExporter::new(
            env_load!(InfluxAddress, INFLUX_ADDRESS),
            env_load!(InfluxOrg, INFLUX_ORG),
            env_load!(InfluxToken, INFLUX_TOKEN),
            env_load!(InfluxBucket, INFLUX_BUCKET),
            InstanceName::new(sla.id.to_string())?,
        )
        .await
        .expect("Cannot build the InfluxDB2 database connection"),
    );

    let sla_id_raw = sla.id;
    let metrics_raw = metrics.clone();

    let http_client = web::Data::from(http_client);
    let metrics = web::Data::from(metrics);
    let sla_id = web::Data::from(Arc::new(sla_id_raw.to_string()));

    let server = HttpServer::new(move || {
        let app = App::new().wrap(middleware::Compress::default());

        #[cfg(feature = "jaeger")]
        let app =
            app.wrap(TracingLogger::default()).wrap(RequestTracing::new());

        app.wrap_fn(|req, srv| {
            // Store the instant when the first byte was received in
            // request extensions
            req.extensions_mut().insert(Utc::now());

            // Call the next middleware or handler
            srv.call(req)
        })
        .app_data(web::Data::clone(&http_client))
        .app_data(web::Data::clone(&metrics))
        .app_data(web::Data::clone(&sla_id))
        .service(web::scope("/").route("", web::post().to(handle)))
    });

    metrics_raw
        .observe(FinishedBooting {
            value:     0.0,
            sla_id:    sla_id_raw.to_string(),
            toto:      std::env::var(TOTO)?,
            timestamp: Utc::now(),
        })
        .await?;

    server.bind(("0.0.0.0", 3000))?.run().await?;

    // Ensure all spans have been reported
    #[cfg(feature = "jaeger")]
    opentelemetry::global::shutdown_tracer_provider();

    Ok(())
}
