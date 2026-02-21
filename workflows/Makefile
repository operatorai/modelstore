REPO_ROOT=$(shell cd ../ && pwd)

.PHONY: setup install uninstall refresh

install: uninstall
	@$(REPO_ROOT)/bin/_uv_install
	find requirements/ -name "*.txt" -type f -exec uv pip install -r '{}' ';'

uninstall:
	@$(REPO_ROOT)/bin/_uv_uninstall

refresh:
	@echo "\n 🔵  Refreshing installation of modelstore"
	uv pip install --upgrade pip setuptools wheel
	uv pip uninstall -y modelstore
	uv pip install --no-cache-dir -e $(REPO_ROOT)

setup:
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
