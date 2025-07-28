use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    routing::get,
    Router,
};
use rand::seq::SliceRandom;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{sqlite::SqlitePool, Row};
use std::{
    collections::{HashMap, HashSet}, net::SocketAddr, sync::Arc
};
use tokio::sync::{broadcast, RwLock, Mutex};
use tower_http::cors::CorsLayer;
use uuid::Uuid;
use sha2::{Sha256, Digest};
use polars::prelude::*;
use std::fs;
use tracing::{info, error};

// Constants matching the Python implementation
const INITIAL_LOAD_SIZE: i64 = 100_000;
const LOAD_INCREMENT_SIZE: i64 = 20_000;
const CHECKPOINT_SIZE: usize = 1000;

// Structures matching the Python implementation's data model
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkBatch {
    batch_id: String,
    rows: Vec<DataRow>,
    assigned_to: Option<String>,
    assigned_time: Option<DateTime<Utc>>,
    completed: bool,
    skipped: bool,
    chunk_id: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_snake_case)]
struct DataRow {
    Body: Option<String>,
    Score: Option<i64>,
    Title: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClientMessage {
    command: String,
    batch_id: Option<String>,
    results: Option<Vec<serde_json::Value>>,
    model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StatusUpdate {
    command: String,
    total_batches: i64,
    completed_batches: i64,
    connected_clients: usize,
    current_results_buffer: usize,
    remaining_batches: usize,
    active_batches: usize,
}

// Main server state structure
struct LLMServer {
    work_batches: Arc<RwLock<HashMap<String, WorkBatch>>>,
    completed_results: Arc<Mutex<Vec<serde_json::Value>>>,
    clients: Arc<RwLock<HashMap<String, broadcast::Sender<Message>>>>,
    current_dataset_position: Arc<Mutex<i64>>,
    is_loading_data: Arc<RwLock<bool>>,
    all_data_processed: Arc<RwLock<bool>>,
    db_pool: SqlitePool,
    dataset_pool: SqlitePool,
    batch_size: i64,
    target_rows: i64,
    processed_batches_cache: Arc<RwLock<HashSet<String>>>,
}

impl LLMServer {
    async fn new(
        batch_size: i64,
        target_rows: i64,
        db_path: &str,
        dataset_path: &str,
    ) -> Result<Arc<Self>, sqlx::Error> {
        // Create directories
        fs::create_dir_all("checkpoints").unwrap_or_default();

        // Initialize SQLite connections
        let db_pool = SqlitePool::connect(&format!("sqlite:{}", db_path)).await?;
        let dataset_pool = SqlitePool::connect(&format!("sqlite:{}", dataset_path)).await?;

        // Initialize database tables
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS completed_batches (
                batch_id TEXT PRIMARY KEY
            );
            CREATE TABLE IF NOT EXISTS blacklisted_batches (
                batch_id TEXT PRIMARY KEY
            );
            CREATE TABLE IF NOT EXISTS skipped_batches (
                batch_id TEXT PRIMARY KEY
            );
            "#,
        )
        .execute(&db_pool)
        .await?;

        let processed_batches = Arc::new(RwLock::new(sqlx::query(
            r#"
            SELECT batch_id FROM completed_batches
            UNION
            SELECT batch_id FROM blacklisted_batches
            UNION
            SELECT batch_id FROM skipped_batches
            "#
        )
        .fetch_all(&db_pool)
        .await?
        .iter()
        .map(|row| row.get::<String, _>("batch_id"))
        .collect::<HashSet<_>>()));

        Ok(Arc::new(Self {
            work_batches: Arc::new(RwLock::new(HashMap::new())),
            completed_results: Arc::new(Mutex::new(Vec::new())),
            clients: Arc::new(RwLock::new(HashMap::new())),
            current_dataset_position: Arc::new(Mutex::new(0)),
            is_loading_data: Arc::new(RwLock::new(false)),
            all_data_processed: Arc::new(RwLock::new(false)),
            db_pool,
            dataset_pool,
            batch_size,
            target_rows,
            processed_batches_cache: processed_batches
        }))
    }

    fn generate_deterministic_batch_id(&self, rows: &[DataRow]) -> String {
        let mut hasher = Sha256::new();
        let serialized = serde_json::to_string(&rows).unwrap();
        hasher.update(serialized.as_bytes());
        format!("{:x}", hasher.finalize())[..32].to_string()
    }

    async fn is_batch_processed(&self, batch_id: &str) -> bool {
        // Check cache first
        let cache_hit = {
            let cache = self.processed_batches_cache.read().await;
            cache.contains(batch_id)
        };
        
        if cache_hit {
            info!("Batch {} found in cache", batch_id);
            return true;
        }
    
        // If not in cache, check DB
        let result = sqlx::query(r#"
            SELECT 1 FROM completed_batches WHERE batch_id = ?
            UNION ALL
            SELECT 1 FROM blacklisted_batches WHERE batch_id = ?
            UNION ALL
            SELECT 1 FROM skipped_batches WHERE batch_id = ?
            "#,)
            .bind(batch_id)
            .bind(batch_id)
            .bind(batch_id)
            .fetch_optional(&self.db_pool)
            .await;
    
        let is_processed = result.unwrap_or(None).is_some();
        if is_processed {
            info!("Batch {} found in database", batch_id);
            // Update cache
            self.processed_batches_cache.write().await.insert(batch_id.to_string());
        }
        
        is_processed
    }

    async fn process_batch_results(
        &self,
        batch_id: String,
        results: Vec<serde_json::Value>,
        model: Option<String>,
    ) -> Result<(), sqlx::Error> {
        // First, verify the batch exists and mark it as completed
        let batch_completed = {
            let mut work_batches = self.work_batches.write().await;
            if let Some(batch) = work_batches.get_mut(&batch_id) {
                batch.completed = true;
                true
            } else {
                false
            }
        }; // Release work_batches lock immediately

        if !batch_completed {
            return Ok(());
        }

        // Save to DB outside of any locks
        sqlx::query("INSERT OR IGNORE INTO completed_batches (batch_id) VALUES (?)")
            .bind(&batch_id)
            .execute(&self.db_pool)
            .await?;

        // Process results
        let mut results_with_model = results;
        for result in &mut results_with_model {
            if let serde_json::Value::Object(map) = result {
                map.insert(
                    "model_used".to_string(),
                    serde_json::Value::String(model.clone().unwrap_or_else(|| "unknown".to_string()))
                );
            }
        }

        // Add to completed results with minimal lock time
        {
            let mut completed_results = self.completed_results.lock().await;
            completed_results.extend(results_with_model);
            
            // Check if we need to checkpoint
            if completed_results.len() >= CHECKPOINT_SIZE {
                // Clone the results and clear the buffer while holding the lock
                let results_to_checkpoint = completed_results.clone();
                completed_results.clear();
                
                // Release the lock before checkpointing
                drop(completed_results);
                
                // Perform checkpoint with the cloned data
                self.checkpoint_results_with_data(results_to_checkpoint).await;
            }
        }

        // Check completion status
        let completed_batches = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM completed_batches")
            .fetch_one(&self.db_pool)
            .await?;

        if completed_batches >= self.target_rows / self.batch_size 
           && *self.all_data_processed.read().await {
            // Checkpoint any remaining results
            let remaining_results = {
                let mut completed_results = self.completed_results.lock().await;
                let results = completed_results.clone();
                completed_results.clear();
                results
            };

            if !remaining_results.is_empty() {
                self.checkpoint_results_with_data(remaining_results).await;
            }

            self.save_results().await;
        }

        Ok(())
    }

    async fn checkpoint_results_with_data(&self, results: Vec<serde_json::Value>) {
        if results.is_empty() {
            return;
        }

        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
        let checkpoint_file = format!("checkpoints/checkpoint_{}.parquet", timestamp);

        info!("Creating checkpoint with {} results", results.len());

        // Convert to DataFrame and save
        let mut df = DataFrame::new(vec![
            Series::new("data".into(), results.iter().map(|v| v.to_string()).collect::<Vec<_>>()).into()
        ]).unwrap();

        let mut file = std::fs::File::create(&checkpoint_file).unwrap();
        ParquetWriter::new(&mut file).finish(&mut df).unwrap();

        info!("Checkpoint saved to {}", checkpoint_file);
    }

    async fn load_data_chunk(&self, num_rows: i64) -> Result<(), sqlx::Error> {
        info!("Starting load_data_chunk with num_rows: {}", num_rows);
        let mut is_loading = self.is_loading_data.write().await;
        if *is_loading {
            info!("Data loading already in progress, skipping...");
            return Ok(());
        }
        
        *is_loading = true;
        
        // Use drop to ensure the lock is released even if there's an error
        let _guard = scopeguard::guard((), |_| {
            let is_loading_data = self.is_loading_data.clone();
            let future = async move {
                let mut is_loading = is_loading_data.write().await;
                *is_loading = false;
            };
            tokio::spawn(future);
        });
    
        let current_position = *self.current_dataset_position.lock().await;
        info!("Loading up to {} rows from position {}", num_rows, current_position);

        let rows = sqlx::query(
            r#"
            SELECT Body, Score, Title, row_id
            FROM posts 
            WHERE row_id > ? 
            ORDER BY row_id 
            LIMIT ?
            "#,
        )
        .bind(current_position)
        .bind(num_rows)
        .fetch_all(&self.dataset_pool)
        .await?;

        if rows.is_empty() {
            *self.all_data_processed.write().await = true;
            *is_loading = false;
            return Ok(());
        }

        let mut last_row_id = current_position;
        let mut records = Vec::new();

        for row in rows {
            let data_row = DataRow {
                Body: row.get("Body"),
                Score: row.get("Score"),
                Title: row.get("Title"),
            };
            last_row_id = row.get("row_id");
            records.push(data_row);
        }

        let mut new_batches = 0;
        let mut work_batches = self.work_batches.write().await;

        for chunk in records.chunks(self.batch_size as usize) {
            let batch_id = self.generate_deterministic_batch_id(chunk);
            let is_processed = self.is_batch_processed(&batch_id).await;
            info!("Checking batch {}: processed = {}", batch_id, is_processed); // Add this line
            if !is_processed {
                if !work_batches.contains_key(&batch_id) {
                    let batch = WorkBatch {
                        batch_id: batch_id.clone(),
                        rows: chunk.to_vec(),
                        assigned_to: None,
                        assigned_time: None,
                        completed: false,
                        skipped: false,
                        chunk_id: Some(current_position),
                    };
                    work_batches.insert(batch_id, batch);
                    new_batches += 1;
                }
            }
        }

        *self.current_dataset_position.lock().await = last_row_id;
        info!("Loaded {} new rows, created {} new batches", records.len(), new_batches);
        *is_loading = false;
        Ok(())
    }

    async fn get_next_batch(server: Arc<Self>, client_id: &str) -> Option<WorkBatch> {
        // Add debugging
        let total_batches = server.work_batches.read().await.len();
        let unprocessed = server.work_batches.read().await
            .values()
            .filter(|b| !b.completed && !b.skipped)
            .count();
        info!("Total batches: {}, Unprocessed: {}", total_batches, unprocessed);
    
        // Try to get work a few times before giving up
        for _ in 0..3 {
            let mut candidates = {
                let work_batches_guard = server.work_batches.read().await;
                let processed_cache = server.processed_batches_cache.read().await;
                
                work_batches_guard
                    .values()
                    .filter(|batch| {
                        !batch.completed && 
                        !batch.skipped && 
                        !processed_cache.contains(&batch.batch_id) &&
                        batch.assigned_to.is_none()  // Only get unassigned batches
                    })
                    .take(LOAD_INCREMENT_SIZE as usize)
                    .map(|batch| batch.batch_id.clone())
                    .collect::<Vec<_>>()
            };
    
            if candidates.is_empty() {
                // Try to load more data
                if let Err(e) = server.load_data_chunk(LOAD_INCREMENT_SIZE).await {
                    error!("Failed to load data chunk: {}", e);
                }
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                continue;
            }
    
            candidates.shuffle(&mut rand::thread_rng());
    
            for batch_id in candidates {
                if !server.is_batch_processed(&batch_id).await {
                    let mut work_batches = server.work_batches.write().await;
                    
                    if let Some(batch) = work_batches.get_mut(&batch_id) {
                        if batch.assigned_to.is_none() || 
                           batch.assigned_time.map(|t| (Utc::now() - t).num_seconds() > 90000).unwrap_or(true) {
                            batch.assigned_to = Some(client_id.to_string());
                            batch.assigned_time = Some(Utc::now());
                            return Some(batch.clone());
                        }
                    }
                }
            }
        
            let _ = server.load_data_chunk(LOAD_INCREMENT_SIZE).await;
        }
    
        None
    }

    async fn cleanup_stale_assignments(&self) {
        let mut work_batches = self.work_batches.write().await;
        for batch in work_batches.values_mut() {
            if let Some(assigned_time) = batch.assigned_time {
                if (Utc::now() - assigned_time).num_seconds() > 90000 {
                    batch.assigned_to = None;
                    batch.assigned_time = None;
                }
            }
        }
    }

    async fn checkpoint_results(&self) {
        let mut completed_results = self.completed_results.lock().await;
        if completed_results.is_empty() {
            return;
        }

        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
        let checkpoint_file = format!("checkpoints/checkpoint_{}.parquet", timestamp);

        info!("Creating checkpoint with {} results", completed_results.len());

        // Convert to DataFrame and save
        let mut df = DataFrame::new(vec![
            Series::new("data".into(), completed_results.iter().map(|v| v.to_string()).collect::<Vec<_>>()).into()
        ]).unwrap();

        let mut file = std::fs::File::create(&checkpoint_file).unwrap();
        ParquetWriter::new(&mut file).finish(&mut df).unwrap();

        completed_results.clear();
        info!("Checkpoint saved to {}", checkpoint_file);
    }

    async fn save_results(&self) {
        info!("Saving final results...");
        let mut all_results_df = DataFrame::default();

        for entry in fs::read_dir("checkpoints").unwrap() {
            let path = entry.unwrap().path();
            if path.extension().unwrap_or_default() == "parquet" {
                info!("Loading checkpoint file: {:?}", path);
                let file = std::fs::File::open(&path).unwrap();
                let df = ParquetReader::new(file).finish().unwrap();
                all_results_df.vstack(&df).unwrap();
                fs::remove_file(&path).unwrap();
            }
        }

        if !all_results_df.is_empty() {
            let mut file = std::fs::File::create("so_ai_solved.parquet").unwrap();
            ParquetWriter::new(&mut file).finish(&mut all_results_df).unwrap();
            info!("Final results saved to so_ai_solved.parquet");
        }
    }

    async fn handle_client_message(
        server: Arc<LLMServer>,
        msg: Message,
        client_id: String,
    ) -> Result<Option<Message>, serde_json::Error> {
        if let Message::Text(text) = msg {
            let client_msg: ClientMessage = serde_json::from_str(&text)?;

            match client_msg.command.as_str() {
                "get_work" => {
                    info!("Client {} requested work", client_id);
                    if let Some(batch) = LLMServer::get_next_batch(server.clone(), &client_id).await {
                        info!("Giving batch {} to client {}", batch.batch_id, client_id);
                        let response = serde_json::json!({
                            "command": "process_batch",
                            "batch_id": batch.batch_id,
                            "data": batch.rows
                        });
                        Ok(Some(Message::Text(response.to_string().into())))
                    } else {
                        info!("No work for client {}", client_id);
                        Ok(Some(Message::Text(r#"{"command":"no_work"}"#.to_string().into())))
                    }
                }
                "submit_results" => {
                    if let (Some(batch_id), Some(results)) = (client_msg.batch_id, client_msg.results) {
                        info!("Processing batch {} from client {}, model {:?}", batch_id, client_id, client_msg.model);
                        
                        if let Err(e) = server.process_batch_results(
                            batch_id,
                            results,
                            client_msg.model
                        ).await {
                            error!("Error processing batch results: {}", e);
                        }
                    }
                    Ok(None)
                }
                "no_return" => {
                    if let Some(batch_id) = client_msg.batch_id {
                        info!("Client {} marked batch {} as skipped", client_id, batch_id);
                        let mut work_batches = server.work_batches.write().await;
                        if let Some(batch) = work_batches.get_mut(&batch_id) {
                            batch.skipped = true;
                            
                            // Save to DB
                            for table in ["skipped_batches", "blacklisted_batches"] {
                                sqlx::query(&format!("INSERT OR IGNORE INTO {} (batch_id) VALUES (?)", table))
                                    .bind(&batch_id)
                                    .execute(&server.db_pool)
                                    .await
                                    .unwrap();
                            }
                        }
                    }
                    Ok(None)
                }
                "status" => {
                    let total_batches = server.target_rows / server.batch_size;
                    let completed_batches = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM completed_batches")
                        .fetch_one(&server.db_pool)
                        .await
                        .unwrap_or(0);

                    let work_batches = server.work_batches.read().await;
                    let active_batches = work_batches
                        .values()
                        .filter(|b| 
                            !b.completed && 
                            !b.skipped && 
                            b.assigned_to.is_some() && 
                            b.assigned_time
                                .map(|t| (Utc::now() - t).num_seconds() <= 90000)
                                .unwrap_or(false)
                        )
                        .count();

                    let status = StatusUpdate {
                        command: "status_update".to_string(),
                        total_batches,
                        completed_batches,
                        connected_clients: server.clients.read().await.len(),
                        current_results_buffer: server.completed_results.lock().await.len(),
                        remaining_batches: work_batches
                            .values()
                            .filter(|b| !b.completed)
                            .count(),
                        active_batches,
                    };
                    info!("Status update: {:?}", status);
                    Ok(Some(Message::Text(serde_json::to_string(&status)?.into())))
                }
                _ => Ok(None),
            }
        } else {
            Ok(None)
        }
    }

    async fn handle_websocket(
        server: Arc<LLMServer>,
        mut socket: WebSocket,
    ) {
        let client_id = Uuid::new_v4().to_string();
        info!("New client connected: {}", client_id);

        // Set up message channel for this client
        let (tx, _rx) = broadcast::channel(100);
        server.clients.write().await.insert(client_id.clone(), tx);

        while let Some(result) = socket.recv().await {
            match result {
                Ok(msg) => {
                    match LLMServer::handle_client_message(server.clone(), msg, client_id.clone()).await {
                        Ok(Some(response)) => {
                            if let Err(e) = socket.send(response).await {
                                error!("Error sending message to client {}: {}", client_id, e);
                                break;
                            }
                        }
                        Ok(None) => {}
                        Err(e) => {
                            error!("Error processing message from client {}: {}", client_id, e);
                        }
                    }
                }
                Err(e) => {
                    error!("Error receiving message from client {}: {}", client_id, e);
                    break;
                }
            }
        }

        // Clean up client connection
        server.clients.write().await.remove(&client_id);
        info!("Client {} disconnected", client_id);
    }
}

// Initialize tracing for logging
fn setup_logging() {
    use tracing_subscriber::{fmt, EnvFilter};
    let subscriber = fmt()
        .with_env_filter(EnvFilter::from_default_env()
            .add_directive(tracing::Level::INFO.into()))
        .with_target(false)
        .with_thread_ids(true)
        .with_line_number(true)
        .with_file(true)
        .compact()
        .init();
}

#[tokio::main]
async fn main() {
    setup_logging();

    let server = LLMServer::new(
        1, // batch_size
        1_517_595, // target_rows
        "batches.db",
        "so_dataset.sqlite",
    )
    .await.unwrap();

    // Load initial dataset
    server.load_data_chunk(INITIAL_LOAD_SIZE).await.unwrap();

    let server_clone = server.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(300)).await;
            server_clone.cleanup_stale_assignments().await;
        }
    });

    // Set up the Axum router
    let app = Router::new()
        .route("/", get(
            move |ws: WebSocketUpgrade| async move {
                ws.on_upgrade(move |socket| LLMServer::handle_websocket(server.clone(), socket))
            }
        ))
        .layer(CorsLayer::permissive());

    // Start the server
    let addr: SocketAddr = "0.0.0.0:8765".parse().unwrap();
    info!("Starting server on {}", addr);
    
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => listener,
        Err(e) => {
            tracing::error!("Unable to bind to {}: {}", addr, e);
            return;
        }
    };

    axum::serve(listener, app).await.unwrap();
}