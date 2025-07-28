use rayon::prelude::*;
use sha2::{Sha256, Digest};
use base64ct::{Base64, Encoding};
use sqlx::{postgres::{PgPool, PgPoolOptions}, Executor};
use std::time::{Duration, Instant};
use indicatif::{ProgressBar, ProgressStyle};
use tracing_subscriber::EnvFilter;
use std::collections::HashSet;

#[derive(sqlx::FromRow, Clone)]
struct CodeRow {
    code: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup logging
    let filter = EnvFilter::try_new("debug,hyper=info,sqlx=error")
        .unwrap_or_else(|_| EnvFilter::new("debug"));

    let subscriber = tracing_subscriber::fmt::Subscriber::builder()
        .with_env_filter(filter)
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;
    tracing::info!("Starting hash sync");

    let pool = PgPoolOptions::new()
        .max_connections(50)
        .acquire_timeout(Duration::from_secs(30))
        .connect("postgres://tennisbowling:tennispass@127.0.0.1/tennisbowling")
        .await?;

    let start = Instant::now();

    // Load existing hashes into a HashSet for faster lookups
    let existing_hashes: HashSet<String> = sqlx::query_scalar("SELECT hash FROM completed_hashes")
        .fetch_all(&pool)
        .await?
        .into_iter()
        .collect();

    tracing::info!("Loaded {} existing hashes", existing_hashes.len());

    // Get all distinct codes
    let rows: Vec<CodeRow> = sqlx::query_as(
        "SELECT DISTINCT code FROM code_embeddings LIMIT 100000"
    )
    .fetch_all(&pool)
    .await?;

    let total_rows = rows.len();
    tracing::info!("Found {} rows to process", total_rows);

    let pb = ProgressBar::new(total_rows as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    // Process in chunks to avoid overwhelming the database
    let chunk_size = 10000;
    let chunks: Vec<Vec<CodeRow>> = rows.chunks(chunk_size).map(|c| c.to_vec()).collect();

    let mut total_new_hashes = 0;

    for (i, chunk) in chunks.iter().enumerate() {
        // Compute hashes in parallel using Rust's SHA-256 and filter out existing ones
        let new_hashes: Vec<String> = chunk.par_iter()
            .map(|row| {
                Base64::encode_string(&Sha256::digest(&row.code))
            })
            .filter(|hash| !existing_hashes.contains(hash))
            .collect();

        let new_count = new_hashes.len();
        if new_count > 0 {
            // Use COPY for faster inserts
            let mut copy_stmt = String::with_capacity(new_hashes.len() * 100);
            for hash in new_hashes {
                copy_stmt.push_str(&format!("{}\ttrue\t{}\n", hash, chrono::Utc::now()));
            }

            let mut tx = pool.begin().await?;
            tx.execute("COPY completed_hashes (hash, completed, created_at) FROM STDIN")
                .await?;
            
            sqlx::query(&copy_stmt)
                .execute(&mut *tx)
                .await?;
            
            tx.commit().await?;

            total_new_hashes += new_count;
            tracing::info!("Chunk {}: Inserted {} new hashes", i + 1, new_count);
        }

        pb.inc(chunk.len() as u64);
    }

    pb.finish_with_message("Processing complete");
    tracing::info!("Done in {:?}s. Added {} new hashes", start.elapsed().as_secs_f32(), total_new_hashes);
    
    Ok(())
}