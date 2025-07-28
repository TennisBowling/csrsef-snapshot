use serde::Deserialize;
use serde_json::json;
use sha2::{Sha256, Digest};
use tokio::{sync::Mutex, time::sleep};
use tracing_subscriber::EnvFilter;
use std::{env, error::Error, fs, path::Path, sync::{atomic::{AtomicU32, Ordering}, Arc}, time::{Duration, Instant}};
use bitcode::{Encode, Decode};
use sqlx::{
    postgres::{PgPool, PgPoolOptions}, sqlite::SqlitePoolOptions, FromRow
};
use qdrant_client::qdrant::{point_id::PointIdOptions, CountPointsBuilder, Direction, OrderByBuilder, PointStruct, ScrollPointsBuilder, UpsertPointsBuilder};
use qdrant_client::Qdrant;
use base64ct::{Base64, Encoding};
use ast_processor::ASTProcessor;
use tokio_util::task::TaskTracker;
use indicatif::{ProgressBar, ProgressStyle};
use uuid::Uuid;
use tokio::signal;

#[derive(Encode, Decode)]
pub struct Status {
    name: String,
    status: bool,
}

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
pub struct GithubRow {
    pub row_id: u32,
    pub code: String,
    pub full_path: String,
}

#[derive(Clone)]
pub struct BatchProcessor {
    pool: PgPool,
    sqlite: sqlx::SqlitePool,
    client: Arc<reqwest::Client>,
    status_tracker: StatusTracker,
    qdrant: Arc<Qdrant>,
}

impl BatchProcessor {
    pub async fn new(
        pool: PgPool,
        sqlite: sqlx::SqlitePool,
        status_tracker: StatusTracker,
        client: Arc<reqwest::Client>,
        qdrant: Arc<Qdrant>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let processor = Self {
            pool,
            sqlite,
            client,
            status_tracker,
            qdrant,
        };
        processor.create_in_progress_table().await?;
        Ok(processor)
    }

    async fn create_in_progress_table(&self) -> Result<(), Box<dyn std::error::Error>> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS in_progress_batches (
                custom_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL,
                code TEXT NOT NULL,
                ast_data TEXT NOT NULL,
                ast_depth INTEGER NOT NULL,
                full_path TEXT NOT NULL
            );
            "#,
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    #[tracing::instrument(skip_all)]
    pub async fn process_large_batch(&self, row_id: Arc<Mutex<i64>>, remaining_batches: Arc<Mutex<u32>>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        *remaining_batches.lock().await += 1;

        // Fetch 31k rows from SQLite
        let row_id_temp = *row_id.lock().await;
        tracing::info!("Getting 31000 rows from sqlite");
        let rows = sqlx::query_as::<_, GithubRow>("SELECT * FROM github_data WHERE row_id BETWEEN $1 AND $2;")
            .bind(row_id_temp)
            .bind(row_id_temp + 31000)
            .fetch_all(&self.sqlite)
            .await?;

        tracing::info!("Got {} rows from sqlite", rows.len());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let in_progress_data = Arc::new(Mutex::new(Vec::new()));

        let tracker = TaskTracker::new();

        tracing::info!("Making jsonl");
        for row in rows {
            let status_tracker = self.status_tracker.clone();
            let requests_clone = requests.clone();
            let in_progress_data_clone = in_progress_data.clone();

            tracker.spawn(async move {
                let processor = ASTProcessor::new(3, 4, 8).unwrap();

                if status_tracker.is_completed(&row.code.clone()).await.unwrap() {
                    return;
                }

                let snippets = match processor.process_file(&row.code.clone()) {
                    Ok(snippets) => snippets,
                    Err(_) => return,
                };

                for ast in snippets {
                    let code = ast.code.clone();
                    let hash = Base64::encode_string(&Sha256::digest(&code));

                    if status_tracker.is_completed(&code).await.unwrap() {
                        continue;
                    }
                    status_tracker.set_completed(&code).await.unwrap();

                    let request = json!({
                        "custom_id": hash,
                        "method": "POST",
                        "url": "/v1/embeddings",
                        "body": {
                            "input": code,
                            "model": "text-embedding-3-small"
                        }
                    }).to_string();
                    let mut requests = requests_clone.lock().await;
                    if requests.contains(&request) {
                        continue;
                    }
                    requests.push(request);
                    drop(requests);

                    in_progress_data_clone.lock().await.push((
                        hash,
                        code,
                        ast.ast_data,
                        ast.ast_depth,
                        row.full_path.clone(),
                    ));
                }
            });
        }

        tracker.close();
        tracker.wait().await;

        let in_progress_data = Arc::into_inner(in_progress_data).unwrap().into_inner();
        let requests = Arc::into_inner(requests).unwrap().into_inner();

        // Prepare JSONL content
        let jsonl = requests.join("\n");
        let jsonl_bytes = jsonl.into_bytes();
        let bytes_size = jsonl_bytes.len();

        // Upload file to OpenAI
        let form = reqwest::multipart::Form::new()
            .text("purpose", "batch")
            .part(
                "file",
                reqwest::multipart::Part::bytes(jsonl_bytes).file_name("batch.jsonl"),
            );

        tracing::info!("done making jsonl, total of {} bytes", bytes_size);
        let file_upload = self
            .client
            .post("https://api.openai.com/v1/files")
            .header(
                "Authorization",
                "Bearer sk-",
            )
            .multipart(form)
            .send()
            .await?;

        let file_id: serde_json::Value = file_upload.json().await?;
        let file_id = file_id["id"].as_str().ok_or("Failed to get file ID")?;
        tracing::info!("Uploaded file with ID: {}", file_id);

        // Create batch
        let batch_body = json!({
            "input_file_id": file_id,
            "endpoint": "/v1/embeddings",
            "completion_window": "24h"
        });

        let batch_create = self
            .client
            .post("https://api.openai.com/v1/batches")
            .header(
                "Authorization",
                "Bearer sk-",
            )
            .json(&batch_body)
            .send()
            .await?;

        let batch_info: serde_json::Value = batch_create.json().await?;
        let batch_id = batch_info["id"].as_str().unwrap().to_owned();

        tracing::info!("uploaded file and created batch");

        // Insert into in_progress_batches
        let tracker = TaskTracker::new();

        for (custom_id, code, ast_data, ast_depth, full_path) in in_progress_data {
            let pool = self.pool.clone();
            let batch_id = batch_id.clone();
            tracker.spawn(async move {
                sqlx::query(
                    r#"
                    INSERT INTO in_progress_batches (custom_id, batch_id, code, ast_data, ast_depth, full_path)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (custom_id) DO NOTHING
                    "#,
                )
                .bind(custom_id)
                .bind(batch_id)
                .bind(code)
                .bind(ast_data)
                .bind(ast_depth)
                .bind(full_path)
                .execute(&pool)
                .await
                .unwrap();
            });
        }

        tracker.close();
        tracker.wait().await;
        tracing::info!("inserted all into in progress batches");

        // Poll for batch completion
        let batch_id = batch_id.to_string();
        loop {
            let batch_status: serde_json::Value = self
                .client
                .get(format!("https://api.openai.com/v1/batches/{}", batch_id))
                .header(
                    "Authorization",
                    "Bearer sk-",
                )
                .send()
                .await?
                .json()
                .await?;

            let status = batch_status["status"].as_str().unwrap_or("");
            match status {
                "completed" => break,
                "failed" | "expired" | "cancelled" => {
                    return Err("Batch processing failed".into());
                }
                _ => {
                    tracing::debug!("Batch status: {}", status);
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }
            }
        }
        tracing::info!("Batch ready");

        // Retrieve results
        let batch_status: serde_json::Value = self
            .client
            .get(format!("https://api.openai.com/v1/batches/{}", batch_id))
            .header(
                "Authorization",
                "Bearer sk-",
            )
            .send()
            .await?
            .json()
            .await?;
        tracing::info!("downloading batch results");
        let output_file_id = batch_status["output_file_id"].as_str().ok_or("No output file")?;
        let results = self
            .client
            .get(format!("https://api.openai.com/v1/files/{}/content", output_file_id))
            .header(
                "Authorization",
                "Bearer sk-",
            )
            .timeout(Duration::from_secs(1080))
            .send()
            .await?
            .text()
            .await?;
        tracing::info!("got results from batch, processing");

        // Process results
        let completed = Arc::new(AtomicU32::new(0));
        let total = results.lines().count();

        let tracker = TaskTracker::new();

        for line in results.lines() {
            let line = line.to_owned();
            let pool = self.pool.clone();
            let completed_clone = completed.clone();
            let qdrant = self.qdrant.clone();

            tracker.spawn(async move {
                let result: serde_json::Value = serde_json::from_str(&line).unwrap();
                let custom_id = result["custom_id"].as_str().unwrap_or_default();
                let status_code = result["response"]["status_code"].as_u64().unwrap_or(0);

                if status_code == 200 {
                    let response_body: OkResponse = match serde_json::from_value(result["response"]["body"].clone()) {
                        Ok(resp) => resp,
                        Err(e) => {
                            tracing::error!("Couldn't deserialize batch response: {:?}", e);
                            return;
                        }
                    };

                    let embedding = &response_body.data[0].embedding;

                    let (code, ast_data, ast_depth, full_path): (String, String, i32, String) = match sqlx::query_as(
                        "SELECT code, ast_data, ast_depth, full_path FROM in_progress_batches WHERE custom_id = $1",
                    )
                    .bind(custom_id)
                    .fetch_one(&pool)
                    .await {
                        Ok(tuple) => tuple,
                        Err(e) => {
                            tracing::error!("Couldn't get from in progress batches: {:?}", e);
                            return;
                        }
                    };

                    // Create Qdrant point with UUID
                    let point_id = Uuid::new_v4();
                    let point = PointStruct::new(
                        point_id.to_string(), // Use UUID as string ID
                        embedding.clone(),
                        [
                            ("code", code.into()),
                            ("ast_data", ast_data.into()),
                            ("ast_depth", (ast_depth as i64).into()),
                            ("full_path", full_path.into()),
                        ],
                    );

                    // Insert into Qdrant
                    if let Err(e) = qdrant
                        .upsert_points(UpsertPointsBuilder::new("code_embeddings", vec![point]).wait(false))
                        .await
                    {
                        tracing::error!("Error inserting into Qdrant: {:?}", e);
                        return;
                    }

                    match sqlx::query("DELETE FROM in_progress_batches WHERE custom_id = $1")
                        .bind(custom_id)
                        .execute(&pool)
                        .await {
                            Ok(_) => {},
                            Err(e) => {
                                tracing::error!("Couldn't delete from in progress batch: {:?}", e);
                                return;
                            }
                        };

                    completed_clone.fetch_add(1, Ordering::SeqCst);
                }
                else {
                    tracing::error!(
                        "Failed request for custom_id {}: {:?}",
                        custom_id,
                        result["error"]
                    );
                }
            });
        }
        tracker.close();
        tracker.wait().await;

        let completed = Arc::into_inner(completed).unwrap().into_inner();
        tracing::info!("Done with batch, inserted {} embeddings out of {}, {}", completed, total, completed as f32 / total as f32);

        let remaining = *remaining_batches.lock().await;
        if remaining == 1 {
            save_progress(*row_id.lock().await).unwrap();
            std::process::exit(0);
        }
        else {
            *remaining_batches.lock().await -= 1;
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct StatusTracker {
    pool: PgPool,
}

impl StatusTracker {
    pub async fn new(pool: PgPool) -> Result<Self, Box<dyn std::error::Error>> {
        // Create the table if it doesn't exist
        sqlx::query("CREATE TABLE IF NOT EXISTS completed_hashes (
                hash TEXT PRIMARY KEY,
                completed BOOLEAN NOT NULL DEFAULT true,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );")
            .execute(&pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS completed_hashes_hash_idx ON completed_hashes(hash);")
        .execute(&pool)
        .await?;

        Ok(Self { pool })
    }

    pub async fn migrate_from_bitcode(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Only migrate if the bitcode file exists
        if let Ok(file) = tokio::fs::read("./completed.bitcode").await {
            if !file.is_empty() {
                let statuses: Vec<Status> = bitcode::decode(&file)?;

                // Use a transaction for better performance
                let mut tx = self.pool.begin().await?;

                for status in statuses {
                    sqlx::query(
                        "INSERT INTO completed_hashes (hash, completed)
                         VALUES ($1, $2)
                         ON CONFLICT (hash) DO NOTHING"
                    )
                    .bind(&status.name)
                    .bind(status.status)
                    .execute(&mut *tx)
                    .await?;
                }

                tx.commit().await?;

                // Optionally backup and remove the old bitcode file
                tokio::fs::rename("./completed.bitcode", "./completed.bitcode.bak").await?;
            }
        }

        Ok(())
    }

    pub async fn is_completed(&self, code: &str) -> Result<bool, Box<dyn std::error::Error>> {
        let hash = Base64::encode_string(&Sha256::digest(code));

        let result = sqlx::query_scalar::<_, bool>(
            "SELECT completed FROM completed_hashes WHERE hash = $1"
        )
        .bind(&hash)
        .fetch_optional(&self.pool)
        .await?;

        Ok(result.unwrap_or(false))
    }

    pub async fn set_completed(&self, code: &str) -> Result<(), Box<dyn std::error::Error>> {
        let hash = Base64::encode_string(&Sha256::digest(code));

        sqlx::query(
            "INSERT INTO completed_hashes (hash, completed)
             VALUES ($1, true)
             ON CONFLICT (hash) DO UPDATE SET completed = true"
        )
        .bind(&hash)
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

const PROGRESS_FILE: &str = "./batch_progress.txt";

fn save_progress(row_id: i64) -> Result<(), Box<dyn std::error::Error>> {
    fs::write(PROGRESS_FILE, row_id.to_string())?;
    Ok(())
}

fn read_progress() -> Result<i64, Box<dyn std::error::Error>> {
    if Path::new(PROGRESS_FILE).exists() {
        let contents = fs::read_to_string(PROGRESS_FILE)?;
        Ok(contents.trim().parse()?)
    } else {
        Ok(0)
    }
}


const BATCH_SIZE: i64 = 400;    // 950 every 6s = 9500rpm, limit 10k


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filter_string = "debug,hyper=info,sqlx=error,h2=info,tower=info".to_string();

    let filter = EnvFilter::try_new(filter_string).unwrap_or_else(|_| EnvFilter::new("debug"));

    let subscriber = tracing_subscriber::fmt::Subscriber::builder()
        .with_env_filter(filter)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("Setting default subscriber failed");
    tracing::info!("Starting inserter");

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .acquire_timeout(Duration::from_secs(500000000))
        .connect("postgres://tennisbowling:tennispass@192.168.1.10/tennisbowling")
        .await?;

    let sqlite = SqlitePoolOptions::new()
        .max_connections(30)
        .acquire_timeout(Duration::from_secs(500000000))
        .connect("./github.sqlite")
        .await?;

    let qdrant = Arc::new(
        Qdrant::from_url("http://192.168.1.10:6334")
            .timeout(Duration::from_secs(60))
            .build()?,
    );


    let start = Instant::now();

    // Get total count for progress bar
    let total_count: i64 = sqlx::query_scalar("SELECT MAX(row_id) FROM github_data;")
        .fetch_one(&sqlite)
        .await?;

    let pb = ProgressBar::new(total_count as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    let status_tracker = StatusTracker::new(pool.clone()).await?;

    let client = Arc::new(reqwest::Client::new());
    let tracker = TaskTracker::new();

    let skip: i64 = match env::args().nth(2) {
        Some(arg) => arg.parse().unwrap_or(0),
        None => read_progress().unwrap_or(0),
    };

    let row_id = Arc::new(Mutex::new(skip));
    pb.inc(*row_id.lock().await as u64);

    let remaining_batches = Arc::new(Mutex::new(0));


    // batch processor 1
    let batch_processor = BatchProcessor::new(
        pool.clone(),
        sqlite.clone(),
        status_tracker.clone(),
        client.clone(),
        qdrant.clone(),
    )
    .await
    .unwrap();

    let row_id_clone = row_id.clone();
    let remaining_batches_clone = remaining_batches.clone();
    tokio::spawn(async move {
        if let Err(e) = batch_processor.process_large_batch(row_id_clone, remaining_batches_clone.clone()).await {
            tracing::error!("Batch processing failed: {:?}", e);
            *remaining_batches_clone.lock().await -= 1;
        }
    });

    *row_id.lock().await += 31000;
    pb.inc(31000);

    sleep(Duration::from_secs(300)).await;  // 5m to upload

    // 2
    let batch_processor2 = BatchProcessor::new(
        pool.clone(),
        sqlite.clone(),
        status_tracker.clone(),
        client.clone(),
        qdrant.clone(),
    )
    .await
    .unwrap();

    let row_id_clone = row_id.clone();
    let remaining_batches_clone = remaining_batches.clone();
    tokio::spawn(async move {
        if let Err(e) = batch_processor2.process_large_batch(row_id_clone, remaining_batches_clone.clone()).await {
            tracing::error!("Batch processing failed: {:?}", e);
            *remaining_batches_clone.lock().await -= 1;
        }
    });

    *row_id.lock().await += 31000;
    pb.inc(31000);

    sleep(Duration::from_secs(300)).await;  // 5m to upload

    // 3
    let batch_processor3 = BatchProcessor::new(
        pool.clone(),
        sqlite.clone(),
        status_tracker.clone(),
        client.clone(),
        qdrant.clone(),
    )
    .await
    .unwrap();

    let row_id_clone = row_id.clone();
    let remaining_batches_clone = remaining_batches.clone();
    tokio::spawn(async move {
        if let Err(e) = batch_processor3.process_large_batch(row_id_clone, remaining_batches_clone.clone()).await {
            tracing::error!("Batch processing failed: {:?}", e);
            *remaining_batches_clone.lock().await -= 1;
        }
    });

    *row_id.lock().await += 31000;
    pb.inc(31000);

    sleep(Duration::from_secs(300)).await;  // 5m to upload

    // 4
    let batch_processor4 = BatchProcessor::new(
        pool.clone(),
        sqlite.clone(),
        status_tracker.clone(),
        client.clone(),
        qdrant.clone(),
    )
    .await
    .unwrap();

    let row_id_clone = row_id.clone();
    let remaining_batches_clone = remaining_batches.clone();
    tokio::spawn(async move {
        if let Err(e) = batch_processor4.process_large_batch(row_id_clone, remaining_batches_clone.clone()).await {
            tracing::error!("Batch processing failed: {:?}", e);
            *remaining_batches_clone.lock().await -= 1;
        }
    });

    *row_id.lock().await += 31000;
    pb.inc(31000);

    sleep(Duration::from_secs(300)).await;  // 5m to upload

    // 5
    let batch_processor5 = BatchProcessor::new(
        pool.clone(),
        sqlite.clone(),
        status_tracker.clone(),
        client.clone(),
        qdrant.clone(),
    )
    .await
    .unwrap();

    let row_id_clone = row_id.clone();
    let remaining_batches_clone = remaining_batches.clone();
    tokio::spawn(async move {
        if let Err(e) = batch_processor5.process_large_batch(row_id_clone, remaining_batches_clone.clone()).await {
            tracing::error!("Batch processing failed: {:?}", e);
            *remaining_batches_clone.lock().await -= 1;
        }
    });

    *row_id.lock().await += 31000;
    pb.inc(31000);

    sleep(Duration::from_secs(300)).await;  // 5m to upload

    let row_id_temp = *row_id.lock().await;
    let mut batches: Vec<GithubRow> = sqlx::query_as("SELECT * FROM github_data WHERE row_id BETWEEN $1 AND $2;")
        .bind(row_id_temp)
        .bind(row_id_temp + BATCH_SIZE - 1)
        .fetch_all(&sqlite)
        .await
        .unwrap();

    *row_id.lock().await += BATCH_SIZE;

    tracing::info!("Loaded initial {}", BATCH_SIZE);

    // Setup Ctrl+C handler
    let row_id_ctrl_c = row_id.clone();
    tokio::spawn(async move {
        signal::ctrl_c().await.unwrap();
        tracing::info!("Ctrl+C received, saving progress and exiting...");
        save_progress(*row_id_ctrl_c.lock().await).unwrap();
        std::process::exit(0);
    });


    // Iterate through row groups
    loop {
        if batches.is_empty() {
            tokio::time::sleep(Duration::from_secs(12)).await;
            tracing::debug!("Loading {} more", BATCH_SIZE);
            let row_id_temp = *row_id.lock().await;
            batches = sqlx::query_as("SELECT * FROM github_data WHERE row_id BETWEEN $1 AND $2;")
                .bind(row_id_temp)
                .bind(row_id_temp + BATCH_SIZE - 1)
                .fetch_all(&sqlite)
                .await
                .unwrap();

            if batches.is_empty() {
                tracing::info!("No more data to load");
                break;
            }

            *row_id.lock().await += BATCH_SIZE;
        }

        let current_batches = batches.drain(..).collect::<Vec<_>>();
        for row in current_batches {
            let pool_clone = pool.clone();
            let status_tracker = status_tracker.clone();
            let tracker_clone = tracker.clone();
            let client_clone = client.clone();
            let qdrant = qdrant.clone();
            let pb = pb.clone();
            let remaining_batches_clone_for_exit = remaining_batches.clone();
            let row_id_clone_for_exit = row_id.clone();

            tracker.spawn(async move {
                let processor = ASTProcessor::new(3, 4, 8).unwrap();

                if status_tracker.is_completed(&row.code).await.unwrap() {
                    pb.inc(1);
                    return;
                }

                let snippet = match processor.process_file(&row.code) {
                    Ok(snippet) => snippet,
                    Err(_) => {
                        status_tracker.set_completed(&row.code).await.unwrap();
                        pb.inc(1);
                        return;
                    }
                };

                for ast in snippet {
                    if status_tracker.is_completed(&ast.code).await.unwrap() {
                        pb.inc(1);
                        return;
                    }

                    let code_clone = ast.code.clone();
                    let client_clone = client_clone.clone();
                    let task = tracker_clone.spawn(async move {
                        let response = match client_clone.post("https://api.openai.com/v1/embeddings")
                            .header("Authorization", "Bearer sk-")
                            .header("Content-Type", "application/json")
                            .body(serde_json::to_string(&json!({"input": code_clone, "model": "text-embedding-3-small"})).unwrap())
                            .send()
                            .await {
                                Ok(res) => res,
                                Err(e) => {
                                    if format!("{:?}", e) == r##"reqwest::Error { kind: Request, url: "https://api.openai.com/v1/embeddings", source: hyper_util::client::legacy::Error(Connect, ConnectError("dns error", Os { code: 22, kind: InvalidInput, message: "Invalid argument" })) }"## {
                                        return Err(());
                                    }
                                    tracing::error!("Request failed: {:?}", e);
                                    return Err(());
                                }
                            };
                        let text = response.text().await.unwrap();

                        match serde_json::from_str::<Response>(&text) {
                            Ok(embeddings) => Ok(embeddings),
                            Err(e) => {
                                tracing::error!("Error on parsing json response: {:?}. response = {:?}", e, text);
                                Err(())
                            }
                        }
                    });

                    let res = match task.await {
                        Ok(res) => {
                            match res {
                                Ok(res) => match res {
                                    Response::Success(ok) => ok,
                                    Response::Error(error_response) => {
                                        if error_response.error.message.starts_with("This model's maximum context") {
                                            status_tracker.is_completed(&ast.code).await.unwrap();
                                            continue;
                                        }
                                        tracing::error!("Error with openai: {:?}", error_response);
                                        status_tracker.is_completed(&ast.code).await.unwrap();
                                        continue;
                                    },
                                },
                                Err(_) => {
                                    status_tracker.is_completed(&ast.code).await.unwrap();
                                    continue;
                                }
                            }
                        },
                        Err(_) => {
                            status_tracker.is_completed(&ast.code).await.unwrap();
                            continue;
                        }
                    };

                    // Create Qdrant point with UUID
                    let point_id = Uuid::new_v4();
                    let point = PointStruct::new(
                        point_id.to_string(), // Store UUID as string
                        res.data[0].embedding.clone(),
                        [
                            ("code", ast.code.clone().into()),
                            ("ast_data", ast.ast_data.into()),
                            ("ast_depth", (ast.ast_depth as i64).into()),
                            ("full_path", row.full_path.clone().into()),
                        ],
                    );

                    if let Err(e) = qdrant
                        .upsert_points(UpsertPointsBuilder::new("code_embeddings", vec![point]).wait(false))
                        .await {
                            tracing::error!("Error inserting into Qdrant: {:?}", e);
                            status_tracker.is_completed(&ast.code).await.unwrap();
                            continue;
                        }

                    status_tracker.is_completed(&ast.code).await.unwrap();
                }
                pb.inc(1);
            });
        }

        tracing::info!("Done processing {}", BATCH_SIZE)
    }

    tracing::info!("Created everything, now just waiting for tasks to finish");

    tracker.close();
    tracker.wait().await;

    pb.finish_with_message("Processing complete");
    save_progress(*row_id.lock().await).unwrap(); // Save progress on normal completion too
    std::process::exit(0); // Exit after completing all tasks

    tracing::info!("Done in {:?}s", start.elapsed().as_secs_f32());
    Ok(())
}