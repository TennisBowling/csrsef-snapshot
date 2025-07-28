import ast
import time
import zss
from typing import List, Any, Optional
import openai
from qdrant_client import QdrantClient


class ASTComparer:
    
    def __init__(self, normalize_whitespace: bool = True, 
                 ignore_variable_names: bool = True,
                 ignore_literal_values: bool = False):
        self.normalize_whitespace = normalize_whitespace
        self.ignore_variable_names = ignore_variable_names
        self.ignore_literal_values = ignore_literal_values
        
    def _get_children(self, node: ast.AST) -> List[ast.AST]:
        if not isinstance(node, ast.AST):
            return []
        
        return [
            child for child in ast.iter_child_nodes(node)
            if isinstance(child, ast.AST) and
            not isinstance(child, (ast.Load, ast.Store, ast.Del))
        ]
    
    def _get_node_label(self, node: ast.AST) -> str:
        if not isinstance(node, ast.AST):
            return str(node)
            
        label = node.__class__.__name__
        
        if isinstance(node, ast.Name):
            if not self.ignore_variable_names:
                label += f"_{node.id}"
        elif isinstance(node, ast.Constant):
            if not self.ignore_literal_values:
                label += f"_{type(node.value).__name__}_{node.value}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                label += f"_{node.func.id}"
        elif isinstance(node, ast.BinOp):
            label += f"_{node.op.__class__.__name__}"
        elif isinstance(node, ast.Compare):
            ops = '_'.join(op.__class__.__name__ for op in node.ops)
            label += f"_{ops}"
            
        return label
    
    def _node_distance(self, a: str, b: str) -> int:
        if a == b:
            return 0
        
        a_parts = a.split('_')
        b_parts = b.split('_')
        
        if a_parts[0] != b_parts[0]:
            return 10
            
        return sum(1 for i in range(min(len(a_parts), len(b_parts)))
                  if a_parts[i] != b_parts[i])
    
    def compare(self, code1: str, code2: str) -> float:
        try:
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)
            
            distance = zss.simple_distance(
                tree1, tree2,
                get_children=self._get_children,
                get_label=self._get_node_label,
                label_dist=self._node_distance
            )
            
            size1 = sum(1 for _ in ast.walk(tree1))
            size2 = sum(1 for _ in ast.walk(tree2))
            max_size = max(size1, size2)
            
            return 1 - (distance / (max_size * 10))
            
        except SyntaxError:
            raise ValueError("Invalid Python code provided")

if __name__ == "__main__":
    comparer = ASTComparer(
        normalize_whitespace=True,
        ignore_variable_names=True,
        ignore_literal_values=False
    )
    qdrant = QdrantClient(url="http://mediacenter2:6333")
    oai = openai.OpenAI(api_key="")
    
    
    code1 = """def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n-1)"""

    code2 = """def fact(x):
        if x <= 1:
            return 1
        return x * fact(x-1)"""
    start = time.time()
    similarity = comparer.compare(code1, code2)
    end = time.time()
    print(f"Similarity score: {similarity:.2f}, final: {end - start}")


    embedding = oai.embeddings.create(input=code1, model="text-embedding-3-small").data[0].embedding

    matches = qdrant.query_points(
        collection_name="code_embeddings",
        query=embedding,
        limit=100
    )

    

    ted_results = []
    for point in matches.points:
        matched_code = point.payload['code']
        if matched_code:
            ted = comparer.compare(code1, matched_code)
            ted_results.append((ted, matched_code))

    ted_results.sort(key=lambda item: item[0], reverse=True)
    print([x[0] for x in ted_results])

    
    print("Top 3 Tree Edit Distances:")
    for ted, matched_code in ted_results[3:]:
        print(f"TED: {ted:.2f}")
        print("Matched Code:\n", matched_code)
        print("-" * 30)