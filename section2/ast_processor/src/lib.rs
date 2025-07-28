use pyo3::prelude::*;
use pyo3::types::PyModule;
use anyhow::{Result, anyhow};
use rayon::iter::{IntoParallelIterator, ParallelIterator};


// WHAT IS EVEN GOING ON
// Basically we run this to make multiple GILs at the same time to parse faster
const PY_CODE: &str = r#"
import ast
import json
import warnings
from typing import Tuple, Dict, Any, List, Set

warnings.filterwarnings("ignore")


class ASTProcessor:
    def __init__(self, min_ast_depth: int = 3, min_node_types: int = 4, min_lines: int = 8):
        self.min_ast_depth = min_ast_depth
        self.min_node_types = min_node_types
        self.min_lines = min_lines

    def get_ast_depth(self, node: ast.AST) -> int:
        if not isinstance(node, ast.AST):
            return 0
        return 1 + max((self.get_ast_depth(child) for child in ast.iter_child_nodes(node)), default=0)
    
    def get_node_types(self, node: ast.AST) -> Set[type]:
        types = {type(node)}
        for child in ast.iter_child_nodes(node):
            types.update(self.get_node_types(child))
        return types
    
    def is_valid_block(self, node: ast.AST) -> Tuple[bool, int]:
        if all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(node)):
            return False, 0
            
        depth = self.get_ast_depth(node)
        unique_types = len(self.get_node_types(node))
        
        return (depth >= self.min_ast_depth and 
                unique_types >= self.min_node_types), depth
    
    def ast_to_dict(self, node: ast.AST) -> Dict[str, Any]:
        if isinstance(node, ast.AST):
            fields = {}
            for field in node._fields:
                value = getattr(node, field)
                if isinstance(value, list):
                    fields[field] = [self.ast_to_dict(x) for x in value]
                else:
                    fields[field] = self.ast_to_dict(value)
            return {
                '_type': node.__class__.__name__,
                '_fields': fields
            }
        elif isinstance(node, (str, int, float, bool, type(None))):
            return node
        else:
            raise TypeError(f"Unexpected type: {type(node)}")

    def process_code(self, code: str) -> Tuple[str, int]:
        try:
            tree = ast.parse(code)
            ast_data = self.ast_to_dict(tree)
            depth = self.get_ast_depth(tree)
            return json.dumps(ast_data), depth
        except SyntaxError as e:
            raise ValueError(f"Invalid Python code: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing code: {str(e)}")
            
    def get_source_segment(self, code: str, node: ast.AST) -> str:
        """Safely extract complete source segment for a node"""
        try:
            # Get the line numbers
            start_lineno = node.lineno - 1  # Convert to 0-based indexing
            end_lineno = node.end_lineno if hasattr(node, 'end_lineno') else start_lineno + 1
            
            # Split the code into lines and extract the relevant ones
            lines = code.splitlines()
            relevant_lines = lines[start_lineno:end_lineno]
            
            # Handle indentation for the first line
            if hasattr(node, 'col_offset'):
                first_line = relevant_lines[0][node.col_offset:]
                relevant_lines[0] = first_line
            
            return '\n'.join(relevant_lines)
        except Exception as e:
            print(f"Error in get_source_segment: {e}")
            return ""
            
    def extract_code_blocks(self, code: str) -> List[Tuple[str, str, int]]:
        try:
            tree = ast.parse(code.strip())
        except Exception as e:
            #print(f"Parse error: {e}")
            return []
            
        blocks = []
        current_lines = []
        
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Process any accumulated standalone lines
                if current_lines:
                    standalone_block = '\n'.join(current_lines).strip()
                    try:
                        parsed_block = ast.parse(standalone_block)
                        if len(standalone_block.splitlines()) >= self.min_lines:
                            is_valid, depth = self.is_valid_block(parsed_block)
                            if is_valid:
                                ast_json = json.dumps(self.ast_to_dict(parsed_block))
                                blocks.append((standalone_block, ast_json, depth))
                    except Exception as e:
                        #print(f"Error processing standalone block: {e}")
                        pass
                    current_lines = []
                
                # Process the function/class definition
                try:
                    node_code = self.get_source_segment(code, node)
                    if node_code and len(node_code.splitlines()) >= self.min_lines:
                        is_valid, depth = self.is_valid_block(node)
                        if is_valid:
                            ast_json = json.dumps(self.ast_to_dict(node))
                            blocks.append((node_code, ast_json, depth))
                except Exception as e:
                    #print(f"Error processing node: {e}")
                    continue
            else:
                try:
                    line = self.get_source_segment(code, node)
                    if line:
                        current_lines.append(line)
                except Exception as e:
                    #print(f"Error getting source segment: {e}")
                    continue
        
        # Process any remaining standalone lines
        if current_lines:
            standalone_block = '\n'.join(current_lines).strip()
            try:
                parsed_block = ast.parse(standalone_block)
                if len(standalone_block.splitlines()) >= self.min_lines:
                    is_valid, depth = self.is_valid_block(parsed_block)
                    if is_valid:
                        ast_json = json.dumps(self.ast_to_dict(parsed_block))
                        blocks.append((standalone_block, ast_json, depth))
            except Exception as e:
                #print(f"Error processing final standalone block: {e}")
                pass
        
        return blocks
"#;

#[derive(Debug, Clone)]
pub struct ASTResult {
    pub code: String,
    pub ast_data: String,
    pub ast_depth: i32,
}

#[derive(Clone)]
pub struct ASTProcessor {
    py_code: &'static str,
    min_ast_depth: i32,
    min_node_types: i32,
    min_lines: i32,
}

impl ASTProcessor {
    pub fn new(min_ast_depth: i32, min_node_types: i32, min_lines: i32) -> Result<Self> {
        Ok(ASTProcessor {
            py_code: PY_CODE,
            min_ast_depth,
            min_node_types,
            min_lines,
        })
    }

    fn create_processor(&self) -> Result<PyObject> {
        Python::with_gil(|py| {
            let module = PyModule::from_code(py, self.py_code, "ast_processor.py", "ast_processor")?;
            let processor_class = module.getattr("ASTProcessor")?;
            let processor = processor_class.call1((self.min_ast_depth, self.min_node_types, self.min_lines))?;
            Ok(processor.into())
        }).map_err(|e: PyErr| anyhow!("Python initialization error: {}", e))
    }

    pub fn process_file(&self, code: &str) -> Result<Vec<ASTResult>> {
        if code.trim().is_empty() {
            return Ok(Vec::new());
        }

        Python::with_gil(|py| {
            let processor = self.create_processor()?;
            
            // Release GIL while processing
            py.allow_threads(|| {
                Python::with_gil(|py| -> Result<Vec<ASTResult>> {
                    let blocks = processor
                        .call_method1(py, "extract_code_blocks", (code,))?
                        .extract::<Vec<(String, String, i32)>>(py)?;

                    Ok(blocks
                        .into_iter()
                        .map(|(code, ast_data, depth)| ASTResult {
                            code,
                            ast_data,
                            ast_depth: depth,
                        })
                        .collect())
                })
            })
        }).map_err(|e| anyhow!("File processing error: {}", e))
    }

    pub fn process_files_parallel(&self, codes: Vec<String>) -> Vec<Result<Vec<ASTResult>>> {
        codes.into_par_iter()
            .map(|code| {
                Python::with_gil(|py| {
                    py.allow_threads(|| self.process_file(&code))
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_file() -> Result<()> {
        let processor = ASTProcessor::new(3, 4, 8)?;
        
        let test_code = r#"
def complex_function():
    x = 1
    y = 2
    result = 0
    for i in range(10):
        if i > 5:
            result += x + y
        else:
            result += i
    return result

class ComplexClass:
    def __init__(self):
        self.value = 0
        
    def complex_method(self):
        for i in range(10):
            if i > 5:
                self.value += i
            else:
                self.value -= i
        return self.value

# Standalone block
total = 0
for i in range(20):
    if i % 2 == 0:
        total += i * 2
    else:
        total += i // 2
print(total)
"#;
        
        let results = processor.process_file(test_code)?;
        
        println!("Number of results: {}", results.len());
        for (i, result) in results.iter().enumerate() {
            println!("Block {}: ", i);
            println!("Code: {}", result.code);
            println!("Depth: {}", result.ast_depth);
        }
        
        assert!(!results.is_empty(), "Should have found at least one valid block");
        
        for result in &results {
            assert!(result.ast_depth >= 3, "AST depth should be at least 3");
            assert!(!result.code.is_empty(), "Code should not be empty");
            assert!(!result.ast_data.is_empty(), "AST data should not be empty");
            
            // Parse the AST data to verify it's valid JSON
            let ast_json: Value = serde_json::from_str(&result.ast_data)
                .expect("AST data should be valid JSON");
            
            // Verify it has the expected structure
            assert!(ast_json.is_object(), "AST data should be a JSON object");
        }
        
        Ok(())
    }
}