"""Export the parity HTML report to PDF using an available rendering engine."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def export_html_to_pdf(*, html_path: Path, pdf_path: Path) -> None:
  """Render HTML to PDF using wkhtmltopdf or weasyprint."""
  if not html_path.exists():
    msg = f"Missing HTML report: {html_path}"
    raise FileNotFoundError(msg)

  pdf_path.parent.mkdir(parents=True, exist_ok=True)
  if wkhtmltopdf := shutil.which("wkhtmltopdf"):
    command = [wkhtmltopdf, "--quiet", str(html_path), str(pdf_path)]
    subprocess.run(command, check=True)  # noqa: S603
    return

  try:
    from weasyprint import HTML
  except ModuleNotFoundError as error:
    msg = (
      "No supported HTML->PDF engine found. Install `wkhtmltopdf` on PATH "
      "or `weasyprint` in the active environment."
    )
    raise RuntimeError(msg) from error

  HTML(filename=str(html_path), base_url=str(html_path.parent)).write_pdf(str(pdf_path))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--html", default="docs/parity/parity_report.html", help="Path to parity HTML report.")
  parser.add_argument("--pdf", default="docs/parity/parity_report.pdf", help="Output PDF path.")
  args = parser.parse_args()

  export_html_to_pdf(
    html_path=Path(args.html).resolve(),
    pdf_path=Path(args.pdf).resolve(),
  )
  print(f"Wrote {Path(args.pdf).resolve()}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

