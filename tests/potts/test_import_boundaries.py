"""Import boundary enforcement for aminx.potts modules.

Tests verify that aminx.potts.{model,poe,sampling} do NOT import from
forbidden modules (aminx.inference.decode, aminx.host.plan, aminx.types.stages,
aminx.inference.logits). The designer module is exempt from this restriction.

Reference: ADR 260605_potts-parallel-not-stageset.md
"""

import ast
from pathlib import Path
from typing import List, Set

import pytest


# Module import definitions
FORBIDDEN_IMPORTS = {
    "aminx.inference.decode",
    "aminx.host.plan",
    "aminx.types.stages",
    "aminx.inference.logits",
}

# Files that must NOT import from FORBIDDEN_IMPORTS
GUARDED_FILES = {
    "src/aminx/potts/model.py",
    "src/aminx/potts/poe.py",
    "src/aminx/potts/sampling.py",
}

# Files exempt from the restriction (may import anything)
EXEMPT_FILES = {
    "src/aminx/potts/designer.py",
}


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to extract all import names from a module."""

    def __init__(self) -> None:
        self.imports: Set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        """Handle: import a.b.c"""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handle: from a.b.c import x, y, z"""
        if node.module:
            self.imports.add(node.module)
        self.generic_visit(node)


def extract_imports(filepath: Path) -> Set[str]:
    """Extract all module imports from a Python file via AST."""
    try:
        code = filepath.read_text()
        tree = ast.parse(code)
        visitor = ImportVisitor()
        visitor.visit(tree)
        return visitor.imports
    except Exception as e:
        pytest.fail(f"Failed to parse {filepath}: {e}")


def find_forbidden_imports(imports: Set[str], forbidden: Set[str]) -> Set[str]:
    """Find which imported modules are in the forbidden set."""
    violations = set()
    for imp in imports:
        # Check exact match and prefixes (e.g., "aminx.inference.decode" matches
        # both "aminx.inference.decode" and "aminx.inference.decode.submodule")
        for forbidden_module in forbidden:
            if imp == forbidden_module or imp.startswith(forbidden_module + "."):
                violations.add(imp)
                break
    return violations


def get_guarded_file_path(relative_path: str) -> Path:
    """Resolve a relative path from project root."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / relative_path


@pytest.mark.potts
class TestImportBoundaries:
    """Verify that potts modules respect import boundaries."""

    @pytest.mark.parametrize(
        "guarded_file",
        sorted(GUARDED_FILES),
        ids=lambda x: x.split("/")[-1],
    )
    def test_guarded_module_no_forbidden_imports(self, guarded_file: str) -> None:
        """Guarded modules must not import from forbidden set.

        Args:
            guarded_file: Relative path to a guarded module.

        Raises:
            AssertionError: If any forbidden import is found, with message
                citing the ADR.
        """
        filepath = get_guarded_file_path(guarded_file)

        # Skip if file doesn't exist yet (allows test to pass before all modules are created)
        if not filepath.exists():
            pytest.skip(f"Module not yet created: {guarded_file}")

        imports = extract_imports(filepath)
        violations = find_forbidden_imports(imports, FORBIDDEN_IMPORTS)

        assert not violations, (
            f"Module {guarded_file} violates import boundary (ADR 260605_potts-parallel-not-stageset.md):\n"
            f"  Forbidden imports found: {', '.join(sorted(violations))}\n"
            f"  Guarded modules must not import from: {', '.join(sorted(FORBIDDEN_IMPORTS))}"
        )

    def test_exempt_module_not_guarded(self) -> None:
        """Verify that exempt modules are not in the guarded list."""
        overlap = GUARDED_FILES & EXEMPT_FILES
        assert not overlap, (
            f"Exempt modules should not be guarded: {overlap}\n"
            f"If a module needs exemption, remove it from GUARDED_FILES."
        )

    def test_all_forbidden_modules_listed(self) -> None:
        """Verify forbidden modules are the expected set."""
        # This test ensures that future changes to the ADR are reflected here.
        expected = {
            "aminx.inference.decode",
            "aminx.host.plan",
            "aminx.types.stages",
            "aminx.inference.logits",
        }
        assert (
            FORBIDDEN_IMPORTS == expected
        ), f"Forbidden imports list mismatch. Update to match ADR 260605_potts-parallel-not-stageset.md"

    def test_guarded_files_aligned_with_adr(self) -> None:
        """Verify guarded modules match ADR specification."""
        expected = {
            "src/aminx/potts/model.py",
            "src/aminx/potts/poe.py",
            "src/aminx/potts/sampling.py",
        }
        assert (
            GUARDED_FILES == expected
        ), f"Guarded files list mismatch. Update to match ADR 260605_potts-parallel-not-stageset.md"

    def test_exempt_files_aligned_with_adr(self) -> None:
        """Verify exempt modules match ADR specification."""
        expected = {
            "src/aminx/potts/designer.py",
        }
        assert (
            EXEMPT_FILES == expected
        ), f"Exempt files list mismatch. Update to match ADR 260605_potts-parallel-not-stageset.md"


# xtrax internal module boundary enforcement
XTRAX_BANNED_INTERNAL = {
    "xtrax.stages.bundle",
    "xtrax.stages.protocols",
    "xtrax.tiling.bucket",
    "xtrax.tiling.dedup",
    "xtrax.tiling.dispatch",
    "xtrax.tiling.plan",
    "xtrax.tiling.strategy",
    "xtrax.engine.engine",
    "xtrax.engine.io",
    "xtrax.safety.manager",
}

XTRAX_FIELD_NAMES = {
    "atom_37",
    "residue_index",
    "tie_group_map",
}


@pytest.mark.potts
class TestXtraxBoundaries:
    """Verify xtrax import boundary enforcement and aminx field separation."""

    def test_aminx_potts_no_xtrax_internals(self) -> None:
        """Verify aminx.potts modules only import from xtrax public APIs.

        Scans all .py files in src/aminx/potts/ and verifies no import
        matches the internal xtrax modules (implementation files that
        should only be accessed via public __init__ re-exports).

        Raises:
            AssertionError: If any internal xtrax import is found.
        """
        potts_root = Path(__file__).parent.parent.parent / "src" / "aminx" / "potts"

        # Collect all violations across all potts modules
        all_violations = {}
        for py_file in potts_root.glob("*.py"):
            imports = extract_imports(py_file)
            violations = find_forbidden_imports(imports, XTRAX_BANNED_INTERNAL)
            if violations:
                all_violations[py_file.name] = violations

        assert not all_violations, (
            f"aminx.potts modules import from xtrax internal modules (ADR 260605):\n"
            + "\n".join(
                f"  {filename}: {', '.join(sorted(viol))}"
                for filename, viol in sorted(all_violations.items())
            )
            + f"\nUse public __init__ re-exports instead, e.g., "
            f"`from xtrax.stages import ...` not `from xtrax.stages.bundle import ...`"
        )

    def test_xtrax_no_aminx_field_names(self) -> None:
        """Verify xtrax codebase does not reference aminx-specific field names.

        These field names (atom_37, residue_index, tie_group_map) are specific
        to aminx and should not appear in xtrax implementation. This enforces
        the separation that xtrax operates on generic structures.

        Raises:
            AssertionError: If any forbidden field name is found in xtrax source.
            pytest.skip: If xtrax source is not available at expected path.
        """
        xtrax_root = Path("/home/marielle/projects/xtrax/src/xtrax")

        # Skip if xtrax is not available (allowed in some CI environments)
        if not xtrax_root.exists():
            pytest.skip("xtrax source not found at expected path")

        violations = {}
        for py_file in xtrax_root.rglob("*.py"):
            code = py_file.read_text()
            for field_name in XTRAX_FIELD_NAMES:
                if field_name in code:
                    rel_path = py_file.relative_to(xtrax_root.parent)
                    if rel_path not in violations:
                        violations[rel_path] = []
                    violations[rel_path].append(field_name)

        assert not violations, (
            f"xtrax source contains aminx-specific field names (ADR 260605):\n"
            + "\n".join(
                f"  {filepath}: {', '.join(sorted(set(fields)))}"
                for filepath, fields in sorted(violations.items())
            )
            + f"\nxtrax must not reference aminx-specific structures. "
            f"Remove or parameterize these references."
        )
