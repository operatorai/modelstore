VIRTUALENV_NAME=modelstore.$(shell pwd | rev | cut -d '/' -f 1 | rev)
REPO_ROOT=$(shell cd ../ && pwd)

.PHONY: setup pyenv pyenv-uninstall refresh

pyenv: pyenv-uninstall
	@$(REPO_ROOT)/bin/_pyenv_install $(VIRTUALENV_NAME)
	find requirements/ -name "*.txt" -type f -exec pip install -r '{}' ';'

refresh:
	@echo "\n 🔵  Refreshing installation of modelstore"
	pip install --upgrade pip setuptools wheel	
	pip uninstall -y modelstore
	pip install --no-cache-dir -e $(REPO_ROOT)

pyenv-uninstall:
	@$(REPO_ROOT)/bin/_pyenv_uninstall $(VIRTUALENV_NAME)

setup:
	pip install --upgrade pip setuptools wheel	
	pip install -r requirements.txt
