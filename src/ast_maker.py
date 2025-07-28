import ast
import json
from typing import Tuple, Any, Dict
import argparse

class ASTAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.depth = 0
        self.max_depth = 0
        
    def generic_visit(self, node: ast.AST) -> None:
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)
        self.depth -= 1

def node_to_dict(node: ast.AST) -> Dict[str, Any]:
    if isinstance(node, ast.AST):
        fields = {}
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                fields[field] = [node_to_dict(item) for item in value]
            elif isinstance(value, ast.AST):
                fields[field] = node_to_dict(value)
            else:
                fields[field] = value
        return {
            'node_type': node.__class__.__name__,
            'fields': fields
        }
    elif isinstance(node, list):
        return [node_to_dict(item) for item in node]
    else:
        return node

def analyze_code(code: str) -> Tuple[Dict[str, Any], int]:
    try:
        tree = ast.parse(code)
        
        analyzer = ASTAnalyzer()
        analyzer.visit(tree)
        ast_depth = analyzer.max_depth
        
        ast_data = node_to_dict(tree)
        
        return ast_data, ast_depth
        
    except SyntaxError as e:
        raise ValueError(f"Invalid Python code: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error analyzing code: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Python code and output AST data and depth.')
    parser.add_argument('code', help='Python code to analyze (as string)')
    parser.add_argument('--output', '-o', help='Output file path (optional)')
    
    args = parser.parse_args()
    
    try:
        ast_data, ast_depth = analyze_code(args.code)
        
        result = {
            'ast_data': ast_data,
            'ast_depth': ast_depth
        }
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results written to {args.output}")
        else:
            print(json.dumps(result, indent=2))
            
    except ValueError as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()