"""Keep the independent edge package free of backend implementation imports."""
from pathlib import Path
import ast
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_client_source_does_not_import_backend_package() -> None:
    imported: set[str] = set()
    for path in (ROOT / "src" / "home_cortex_client").glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    assert not any(name == "home_cortex" or name.startswith("home_cortex.") for name in imported)


def test_import_does_not_eagerly_load_opencv() -> None:
    script = "import home_cortex_client,sys; assert 'cv2' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], cwd=ROOT / "src", check=True)
