use std::env;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, ListArray, StringArray, Float32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use qdrant_client::qdrant::vectors_output::VectorsOptions;
use qdrant_client::{
    Qdrant,
    qdrant::{
        ScrollPointsBuilder,
        {PointId, point_id::PointIdOptions},
    },
};

// Configuration constants
const BATCH_SIZE: u32 = 20000; // Number of points to retrieve in each batch
const OUTPUT_DIR: &str = "./parquet_output"; // Directory to store output files

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let skip_numeric = args.iter().any(|arg| arg == "--skip");
    
    if skip_numeric {
        println!("Skipping numeric IDs and starting with UUID processing");
    }
    
    // Create output directory if it doesn't exist
    std::fs::create_dir_all(OUTPUT_DIR)?;
    
    // Connect to Qdrant
    let client = Qdrant::from_url("http://192.168.1.10:6334").build()?;
    println!("Connected to Qdrant");
    
    // Get collection info to determine vector dimension
    let collection_info = client.collection_info("code_embeddings").await?;
    println!("Retrieved collection info for code_embeddings");
    
    // Define the schema for the parquet files
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("code", DataType::Utf8, true),
        Field::new("full_path", DataType::Utf8, true),
        Field::new("ast_depth", DataType::Int64, true),
        Field::new("ast_data", DataType::Utf8, true),
        Field::new(
            "vector", 
            DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
            true
        ),
    ]));
    
    // Set up the scroll request
    let mut scroll_request = ScrollPointsBuilder::new("code_embeddings")
        .limit(BATCH_SIZE)
        .with_payload(true)
        .with_vectors(true);
    
    // If skipping numeric IDs, set an initial offset to a UUID format
    if skip_numeric {
        // Create a dummy UUID point ID to start processing from UUIDs
        // This uses a placeholder UUID that should come after numeric IDs
        let uuid_point_id = PointId {
            point_id_options: Some(PointIdOptions::Uuid(
                "00000000-0000-0000-0000-000000000000".to_string()
            )),
        };
        scroll_request = scroll_request.offset(uuid_point_id);
    }
    
    let mut batch_num = 0;
    let mut total_points = 0;
    let mut offset = None;
    
    loop {
        println!("Fetching batch {}", batch_num);
        
        // Update the offset if we have one from the previous iteration
        if let Some(offset_value) = offset {
            scroll_request = ScrollPointsBuilder::new("code_embeddings")
                .limit(BATCH_SIZE)
                .with_payload(true)
                .with_vectors(true)
                .offset(offset_value);
        }
        
        // Scroll through points - create a new request for each batch
        let response = client.scroll(scroll_request).await?;

        scroll_request = ScrollPointsBuilder::new("code_embeddings")
            .limit(BATCH_SIZE)
            .with_payload(true)
            .with_vectors(true);
        
        // The result is directly the points in this API version
        let points = response.result;
        if points.is_empty() {
            println!("No more points to fetch");
            break;
        }
        
        let num_points = points.len();
        println!("Retrieved {} points", num_points);
        
        // Count UUIDs vs numeric IDs in this batch
        let mut uuid_count = 0;
        let mut numeric_count = 0;
        
        for point in &points {
            if let Some(id_opt) = &point.id {
                if let Some(point_id_opt) = &id_opt.point_id_options {
                    match point_id_opt {
                        PointIdOptions::Num(_) => numeric_count += 1,
                        PointIdOptions::Uuid(_) => uuid_count += 1,
                    }
                }
            }
        }
        
        println!("Batch contains {} numeric IDs and {} UUIDs", numeric_count, uuid_count);
        
        // Prepare vectors for Arrow arrays
        let mut ids = Vec::with_capacity(num_points);
        let mut codes = Vec::with_capacity(num_points);
        let mut full_paths = Vec::with_capacity(num_points);
        let mut ast_depths = Vec::with_capacity(num_points);
        let mut ast_datas = Vec::with_capacity(num_points);
        let mut vectors = Vec::with_capacity(num_points);
        
        for point in &points {
            // Extract ID
            let id = match &point.id {
                Some(id_opt) => {
                    if let Some(point_id_opt) = &id_opt.point_id_options {
                        match point_id_opt {
                            PointIdOptions::Num(num) => *num as i64,
                            PointIdOptions::Uuid(uuid) => {
                                println!("Processing UUID id: {}", uuid);
                                0 // Still use 0 for parquet, but we continue processing
                            },
                        }
                    } else {
                        0
                    }
                },
                None => 0,
            };
            
            ids.push(id);
            
            // Extract payload - direct access, no Option handling
            if point.payload.is_empty() {
                println!("Warning: Point without payload found (ID: {})", id);
                codes.push(String::new());
                full_paths.push(String::new());
                ast_depths.push(0);
                ast_datas.push(String::new());
                continue;
            }
            
            // Extract code
            let code = point.payload.get("code")
                .and_then(|v| v.kind.as_ref())
                .and_then(|k| match k {
                    qdrant_client::qdrant::value::Kind::StringValue(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            codes.push(code);
            
            // Extract full_path
            let full_path = point.payload.get("full_path")
                .and_then(|v| v.kind.as_ref())
                .and_then(|k| match k {
                    qdrant_client::qdrant::value::Kind::StringValue(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            full_paths.push(full_path);
            
            // Extract ast_depth
            let ast_depth = point.payload.get("ast_depth")
                .and_then(|v| v.kind.as_ref())
                .and_then(|k| match k {
                    qdrant_client::qdrant::value::Kind::IntegerValue(i) => Some(*i as i64),
                    _ => None,
                })
                .unwrap_or_default();
            ast_depths.push(ast_depth);
            
            // Extract ast_data
            let ast_data = point.payload.get("ast_data")
                .and_then(|v| v.kind.as_ref())
                .and_then(|k| match k {
                    qdrant_client::qdrant::value::Kind::StringValue(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            ast_datas.push(ast_data);
            
            // Extract vector - updated for better handling of named vectors
            let vector = match &point.vectors {
                Some(vectors_map) => {
                    match &vectors_map.vectors_options {
                        Some(options) => {
                            match options {
                                VectorsOptions::Vector(vec) => {
                                    vec.data.clone()
                                },
                                VectorsOptions::Vectors(named_vec) => {
                                    // Properly handle named vectors - extract the first one
                                    if !named_vec.vectors.is_empty() {
                                        let first_vector = named_vec.vectors.values().next().unwrap();
                                        first_vector.data.clone()
                                    } else {
                                        Vec::new()
                                    }
                                }
                            }
                        },
                        None => {
                            Vec::new()
                        }
                    }
                },
                None => {
                    println!("Warning: No vectors found for point ID: {}", id);
                    Vec::new()
                }
            };
            vectors.push(vector);
        }
        
        // Create Arrow arrays
        let id_array = Arc::new(Int64Array::from(ids)) as ArrayRef;
        let code_array = Arc::new(StringArray::from(codes)) as ArrayRef;
        let full_path_array = Arc::new(StringArray::from(full_paths)) as ArrayRef;
        let ast_depth_array = Arc::new(Int64Array::from(ast_depths)) as ArrayRef;
        let ast_data_array = Arc::new(StringArray::from(ast_datas)) as ArrayRef;
        
        // Create vector list array - use modern array creation API
        let vector_array = {
            // Combine all vectors into a single array
            let mut all_values = Vec::new();
            let mut offsets = Vec::with_capacity(vectors.len() + 1);
            offsets.push(0);
            
            for vector in &vectors {
                all_values.extend_from_slice(vector);
                offsets.push(all_values.len() as i32);
            }
            
            let values = Float32Array::from(all_values);
            let field = Arc::new(Field::new("item", DataType::Float32, false));
            
            // Convert offsets to proper buffer format
            let offset_buffer = arrow::buffer::ScalarBuffer::from(offsets);
            
            Arc::new(ListArray::try_new(
                field,
                arrow::buffer::OffsetBuffer::new(offset_buffer),
                Arc::new(values) as ArrayRef,
                None,
            )?) as ArrayRef
        };
        
        // Create record batch
        let record_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                id_array,
                code_array,
                full_path_array,
                ast_depth_array,
                ast_data_array,
                vector_array,
            ],
        )?;
        
        // Write to parquet file
        let output_path = Path::new(OUTPUT_DIR)
            .join(format!("code_embeddings_batch_{}.parquet", batch_num));
        
        let file = File::create(output_path)?;
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        
        writer.write(&record_batch)?;
        writer.close()?;
        
        println!("Wrote batch {} with {} points to parquet file", batch_num, num_points);
        
        total_points += num_points;
        
        // Update offset for next batch using the id of the last point
        if let Some(last_point) = points.last() {
            if let Some(id_opt) = &last_point.id {
                println!("Using offset for next batch: {:?}", id_opt);
                offset = Some(id_opt.clone());
                batch_num += 1;
            } else {
                println!("No more batches to fetch - last point has no ID");
                break;
            }
        } else {
            println!("No more batches to fetch - empty result");
            break;
        }
    }
    
    println!("Export complete. Exported {} points in {} batches.", total_points, batch_num + 1);
    
    Ok(())
}