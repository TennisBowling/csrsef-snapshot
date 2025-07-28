use anyhow::Result;
use futures::stream::{self, StreamExt};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use qdrant_client::qdrant::{PointStruct, UpsertPointsBuilder};
use qdrant_client::Qdrant;
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPoolOptions;
use sqlx::{FromRow, PgPool};
use std::env;
use std::time::Duration;
use tokio::sync::Semaphore;
use std::sync::Arc;

#[derive(FromRow, Debug)]
struct CodeEmbedding {
    id: i32,
    code: String,
    ast_data: String,
    ast_depth: i32,
    embedding: Vec<f32>,
    full_path: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct PayloadData {
    code: String,
    ast_data: String,
    ast_depth: i32,
    full_path: Option<String>,
}

async fn get_total_rows(pool: &PgPool) -> Result<i64> {
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM code_embeddings_raw")
        .fetch_one(pool)
        .await?;
    Ok(count.0)
}

async fn fetch_batch(
    pool: &PgPool,
    offset: i64,
    batch_size: i64,
    worker_id: usize,
) -> Result<Vec<CodeEmbedding>> {
    let rows = sqlx::query_as::<_, CodeEmbedding>(
        r#"
        SELECT id, code, ast_data::text, ast_depth, embedding, full_path 
        FROM code_embeddings_raw 
        WHERE id BETWEEN $1 AND $2
        ORDER BY id
        "#,
    )
    .bind(offset)
    .bind(offset + batch_size - 1)
    .fetch_all(pool)
    .await?;
    
    println!("Worker {} fetched {} rows at offset {}", worker_id, rows.len(), offset);
    Ok(rows)
}

async fn process_batch(
    client: &Qdrant,
    rows: Vec<CodeEmbedding>,
    worker_id: usize,
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }

    let points: Vec<_> = rows
        .into_iter()
        .map(|row| {
            PointStruct::new(
                row.id as u64,
                row.embedding,
                [
                    ("code", row.code.into()),
                    ("ast_data", row.ast_data.into()),
                    ("ast_depth", (row.ast_depth as i64).into()),
                    (
                        "full_path",
                        row.full_path.unwrap_or_else(|| "".to_string()).into(),
                    ),
                ],
            )
        })
        .collect();

    client
        .upsert_points(UpsertPointsBuilder::new("code_embeddings", points).wait(false))
        .await?;
    
    println!("Worker {} processed batch", worker_id);
    Ok(())
}

async fn worker_task(
    pool: Arc<PgPool>,
    client: Arc<Qdrant>,
    worker_id: usize,
    start_offset: i64,
    end_offset: i64,
    batch_size: i64,
    pb: ProgressBar,
    semaphore: Arc<Semaphore>,
) -> Result<()> {
    let mut current_offset = start_offset;

    while current_offset < end_offset {
        // Acquire semaphore permit
        let _permit = semaphore.acquire().await?;

        let actual_batch_size = std::cmp::min(batch_size, end_offset - current_offset);
        let rows = fetch_batch(&pool, current_offset, actual_batch_size, worker_id).await?;
        
        if rows.is_empty() {
            break;
        }

        process_batch(&client, rows, worker_id).await?;
        pb.inc(actual_batch_size as u64);
        current_offset += actual_batch_size;
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Configuration
    const NUM_WORKERS: usize = 22;
    const BATCH_SIZE: i64 = 10000;
    const MAX_CONCURRENT_REQUESTS: usize = 5; // Limit concurrent requests to Qdrant

    // Get skip value from environment variable
    let skip = env::var("SKIP")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(0);

    // Initialize clients
    let client = Arc::new(
        Qdrant::from_url("http://192.168.1.10:6334")
            .timeout(Duration::from_secs(60))
            .build()?,
    );

    let pool = Arc::new(
        PgPoolOptions::new()
            .max_connections(NUM_WORKERS as u32 + 1) // +1 for the main connection
            .connect("postgres://tennisbowling:tennispass@192.168.1.10/tennisbowling")
            .await?,
    );

    // Get total number of rows
    let total_rows = get_total_rows(&pool).await?;
    let rows_per_worker = (total_rows - skip) / NUM_WORKERS as i64;

    // Create progress bars
    let mp = MultiProgress::new();
    let progress_style = ProgressStyle::default_bar()
        .template("{spinner:.green} Worker {prefix:.cyan} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-");

    // Create semaphore to limit concurrent requests
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS));

    // Spawn worker tasks
    let handles = stream::iter(0..NUM_WORKERS)
        .map(|worker_id| {
            let pool = Arc::clone(&pool);
            let client = Arc::clone(&client);
            let semaphore = Arc::clone(&semaphore);
            
            let start_offset = skip + (worker_id as i64 * rows_per_worker);
            let end_offset = if worker_id == NUM_WORKERS - 1 {
                total_rows
            } else {
                skip + ((worker_id + 1) as i64 * rows_per_worker)
            };

            let pb = mp.add(ProgressBar::new((end_offset - start_offset) as u64));
            pb.set_style(progress_style.clone());
            pb.set_prefix(worker_id.to_string());

            tokio::spawn(async move {
                if let Err(e) = worker_task(
                    pool,
                    client,
                    worker_id,
                    start_offset,
                    end_offset,
                    BATCH_SIZE,
                    pb,
                    semaphore,
                )
                .await
                {
                    eprintln!("Worker {} error: {}", worker_id, e);
                }
            })
        })
        .collect::<Vec<_>>()
        .await;

    // Wait for all workers to complete
    for handle in handles {
        handle.await?;
    }

    println!("Migration completed");
    pool.close().await;

    Ok(())
}