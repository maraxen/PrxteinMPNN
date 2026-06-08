#!/bin/bash
# scripts/check_model_boundary.sh
# Enforces that _inference is not imported from outside model/ without annotation.

EXIT_CODE=0

echo "Checking for unannotated internal model imports..."

# Find all python files.
if command -v fd > /dev/null; then
  FILES=$(fd -e py . src tests)
else
  FILES=$(find src tests -name "*.py")
fi

for f in $FILES; do
  # Skip files inside src/prxteinmpnn/model/ (they are allowed to import from _inference)
  if [[ "$f" == "src/prxteinmpnn/model/"* ]]; then
    continue
  fi
  # Violation 1: Importing from _inference without annotation
  if grep -E "from prxteinmpnn\.model\._inference" "$f" | grep -v "# prxteinmpnn-internal-import" > /dev/null; then
    echo "VIOLATION: $f contains unannotated import from _inference"
    grep -nE "from prxteinmpnn\.model\._inference" "$f" | grep -v "# prxteinmpnn-internal-import"
    EXIT_CODE=1
  fi
  
  # Violation 2: Importing from moved files (they should be under _inference now)
  if grep -E "from prxteinmpnn\.model\.(ar_scan|ar_exact|ar_exact_ligand|score_exact_ligand)" "$f" | grep -v "# prxteinmpnn-internal-import" > /dev/null; then
    echo "VIOLATION: $f contains direct import of internal model file (move to _inference and annotate)"
    grep -nE "from prxteinmpnn\.model\.(ar_scan|ar_exact|ar_exact_ligand|score_exact_ligand)" "$f" | grep -v "# prxteinmpnn-internal-import"
    EXIT_CODE=1
  fi
done

if [ $EXIT_CODE -eq 0 ]; then
  echo "Boundary check passed."
else
  echo "Boundary check failed."
fi

exit $EXIT_CODE
