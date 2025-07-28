use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::Deserialize;
use serde_json::json;
use sha2::{Sha256, Digest};
use std::{collections::HashMap, fs, path::Component, sync::Arc};

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Row;
use bitcode::{Encode, Decode};
use sqlx::{
    postgres::{PgPool, PgPoolOptions},
    sqlite::SqlitePoolOptions,
    FromRow,
};
use tracing::{error, info, Level};
use base64ct::{Base64, Encoding};
use ast_processor::ASTProcessor;

#[derive(Encode, Decode)]
pub struct Status {
    name: String,
    status: bool,
}

#[derive(Deserialize)]
pub struct EmbeddingResponse {
    object: String,
    embedding: Vec<f32>,
    index: u8,
}

#[derive(Deserialize)]
pub struct Response {
    object: String,
    data: [EmbeddingResponse; 1],
    model: String,
    usage: serde_json::Value,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pool = PgPoolOptions::new()
        .max_connections(20)
        .connect("postgres://tennisbowling:tennispass@192.168.1.10/tennisbowling")
        .await?;

    // Open the parquet file
    let file = fs::File::open("./datasets/github_processed.parquet")?;
    let reader = SerializedFileReader::new(file)?;
    let metadata = reader.metadata();
    let row_groups = metadata.row_groups();

    // Configure batch size for reading
    let batch_size = 1000;
    let mut current_row = 0;
    let max_rows = 1_000;

    let file = tokio::fs::read("./completed.bitcode").await?;
    let completed_raw: Vec<Status> = bitcode::decode(&file)?;
    drop(file);
    let mut completed = HashMap::with_capacity(completed_raw.len());

    completed_raw.iter().for_each(|item| {
        completed.insert(item.name.clone(), item.status);
    });
    drop(completed_raw);

    let client = reqwest::Client::new();

    let count = 0;

    // Iterate through row groups
    for i in 0..row_groups.len() {
        let mut row_group_reader = reader.get_row_group(i)?;
        
        // Read rows in batches
        while current_row < max_rows {
            let mut batch = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                if let Some(row) = row_group_reader.next() {
                    batch.push(row?);
                    current_row += 1;
                    if current_row >= max_rows {
                        break;
                    }
                } else {
                    break;
                }
            }

            if batch.is_empty() {
                break;
            }

            count += 1;
            if count > 100 {
                // just like wait until all the last ten are done
            }

            let processor = ASTProcessor::new().unwrap();

            for row in batch {
                let tx = pool.begin().await.unwrap();
                
                let code = row.get_string("code").unwrap_or_default();
                let full_path = row.get_string("full_path").unwrap_or_default();

                tokio::spawn(async move {
                    let hash = Base64::encode_string(&Sha256::digest(code));
                    if *completed.get(&hash).unwrap_or(&false) {
                        return;
                    }

                    let task = tokio::spawn(async move { 
                        return client.post("https://api.openai.com/v1/embeddings")
                            .bearer_auth("OPENAI_API_KEY")
                            .header("Content-Type", "application/json")
                            .body(format!(r#"{{\"input\": {}, \"model\": \"text-embedding-3-small\"}}"#, code))
                            .send()
                            .await
                            .unwrap()
                            .json::<Response>()
                            .await
                            .unwrap();
                    });

                    let ast = match processor.process_code(&code) {
                        Ok(ast) => ast,
                        Err(_) => return
                    };

                    let res = match task.await {
                        Ok(res) => res,
                        Err(_) => return
                    };

                    sqlx::query(&format!("INSERT INTO code_embeddings (code, ast_data, ast_depth, embedding, repo) VALUES ($1, $2, $3, '{:?}', $4);", res.data[0].embedding))
                        .bind(code)
                        .bind(ast.ast_data)
                        .bind(ast.ast_depth)
                        .bind(full_path)
                        .execute(&mut *tx)
                        .await
                        .unwrap();
                });
            }
        }
    }

    Ok(())
}