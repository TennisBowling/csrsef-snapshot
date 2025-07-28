use serde::Deserialize;
use sqlx::{postgres::{PgPool, PgPoolOptions}, FromRow};
use std::{env, time::Duration, error::Error};
use tracing_subscriber::EnvFilter;
use tokio::fs;
use qdrant_client::qdrant::{PointStruct, UpsertPointsBuilder};
use qdrant_client::Qdrant;
use uuid::Uuid;

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct EmbeddingResponse {
    object: String,
    embedding: Vec<f32>,
    index: u8,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct OkResponse {
    object: String,
    data: [EmbeddingResponse; 1],
    model: String,
    usage: serde_json::Value,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
pub struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    param: Option<String>,
    code: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum Response {
    Success(OkResponse),
    Error(ErrorResponse),
}

#[derive(FromRow, Debug)]
pub struct InProgressRow {
    pub custom_id: String,
    pub code: String,
    pub ast_data: String,
    pub ast_depth: i32,
    pub full_path: String,
}


async fn process_batch_file(pool: &PgPool, qdrant: &Qdrant, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(file_path).await?;
    let mut processed = 0;
    let mut failed = 0;
    let total = contents.lines().count();

    tracing::info!("Processing {} lines from batch file", total);

    for line in contents.lines() {
        let result: serde_json::Value = serde_json::from_str(line)?;
        let custom_id = result["custom_id"].as_str().unwrap_or_default();
        let status_code = result["response"]["status_code"].as_u64().unwrap_or(0);

        if status_code == 200 {
            let response_body: OkResponse = match serde_json::from_value(result["response"]["body"].clone()) {
                Ok(resp) => resp,
                Err(e) => {
                    tracing::error!("Couldn't deserialize batch response for {}: {:?}", custom_id, e);
                    failed += 1;
                    continue;
                }
            };

            let embedding = &response_body.data[0].embedding;

            // Get the original data from in_progress_batches
            let in_progress_row: InProgressRow = match sqlx::query_as::<_, InProgressRow>(
                "SELECT custom_id, code, ast_data, ast_depth, full_path FROM in_progress_batches WHERE custom_id = $1"
            )
            .bind(custom_id)
            .fetch_one(pool)
            .await {
                Ok(row) => row,
                Err(e) => {
                    tracing::error!("Couldn't find entry in in_progress_batches for {}: {:?}", custom_id, e);
                    failed += 1;
                    continue;
                }
            };

            // Create Qdrant point with UUID
            let point_id = Uuid::new_v4();
            let point = PointStruct::new(
                point_id.to_string(), // Use UUID as string ID
                embedding.clone(),
                [
                    ("code", in_progress_row.code.into()),
                    ("ast_data", in_progress_row.ast_data.into()),
                    ("ast_depth", (in_progress_row.ast_depth as i64).into()),
                    ("full_path", in_progress_row.full_path.into()),
                ],
            );

            // Insert into Qdrant
            if let Err(e) = qdrant
                .upsert_points(UpsertPointsBuilder::new("code_embeddings", vec![point]).wait(false))
                .await
            {
                tracing::error!("Error inserting into Qdrant for {}: {:?}", custom_id, e);
                failed += 1;
                continue;
            }


            // Clean up in_progress_batches
            if let Err(e) = sqlx::query("DELETE FROM in_progress_batches WHERE custom_id = $1")
                .bind(custom_id)
                .execute(pool)
                .await {
                    tracing::error!("Couldn't delete from in_progress_batches for {}: {:?}", custom_id, e);
                }

            processed += 1;
            if processed % 1000 == 0 {
                tracing::info!("Processed {}/{} entries ({} failed)", processed, total, failed);
            }
        } else {
            tracing::error!(
                "Failed request for custom_id {}: {:?}",
                custom_id,
                result["error"]
            );
            failed += 1;
        }
    }

    tracing::info!("Processing complete. Successfully processed: {}, Failed: {}", processed, failed);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set up logging
    let filter = EnvFilter::try_new("info,sqlx=error").unwrap_or_else(|_| EnvFilter::new("info"));
    let subscriber = tracing_subscriber::fmt::Subscriber::builder()
        .with_env_filter(filter)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Setting default subscriber failed");

    // Get file path from command line arguments
    let file_path = match env::args().nth(1) {
        Some(path) => path,
        None => {
            eprintln!("Usage: {} <jsonl_file>", env::args().next().unwrap());
            std::process::exit(1);
        }
    };

    // Connect to the database
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .acquire_timeout(Duration::from_secs(30))
        .connect("postgres://tennisbowling:tennispass@192.168.1.10/tennisbowling")
        .await?;

    println!("connected to postgres");

    let qdrant_client = Qdrant::from_url("http://192.168.1.10:6334")
        .timeout(Duration::from_secs(60))
        .build()?;

    println!("connected to qdrant");


    // Process the batch file
    process_batch_file(&pool, &qdrant_client, &file_path).await?;

    Ok(())
}