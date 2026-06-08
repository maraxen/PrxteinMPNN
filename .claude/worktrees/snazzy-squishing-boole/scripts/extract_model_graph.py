"""Extract import relationships from src/prxteinmpnn for Mermaid graph generation."""
import ast
import json
from pathlib import Path

ROOT = Path("src/prxteinmpnn")

def extract_local_imports(path: Path) -> list[str]:
    """Return list of local prxteinmpnn module names imported by path."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and "prxteinmpnn" in (node.module or ""):
                imports.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "prxteinmpnn" in alias.name:
                    imports.append(alias.name)
    return sorted(set(imports))

def module_name(path: Path) -> str:
    """Convert file path to dotted module name."""
    rel = path.relative_to(Path("src"))
    return str(rel.with_suffix("")).replace("/", ".")

def count_classes_and_functions(path: Path) -> tuple[list[str], list[str]]:
    """Return (class_names, top_level_function_names) defined in file."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return [], []
    functions = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    return classes, functions

results = {}
for py_file in sorted(ROOT.rglob("*.py")):
    if "__pycache__" in str(py_file):
        continue
    mod = module_name(py_file)
    lines = len(py_file.read_text().splitlines())
    classes, functions = count_classes_and_functions(py_file)
    imports = extract_local_imports(py_file)
    results[mod] = {
        "path": str(py_file),
        "lines": lines,
        "classes": classes,
        "functions": functions,
        "local_imports": imports,
    }

print(json.dumps(results, indent=2))
