# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = source
BUILDDIR     = build

help:
		@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean livehtml

%: Makefile
		@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
		rm -rf $(BUILDDIR)/*

livehtml:
		@command -v sphinx-autobuild >/dev/null 2>&1 || { \
				echo "sphinx-autobuild not found. Please install with: pip install sphinx-autobuild"; \
				echo "Or install all dev dependencies with: pip install -e \".[dev]\""; \
				exit 1; \
		}
		sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)