VIRTUALENV_NAME=$(shell pwd | rev | cut -d '/' -f 1 | rev)-dev

.PHONY: library test setup install uninstall release-test release-prod clean update

uninstall:
	@./bin/_pyenv_uninstall $(VIRTUALENV_NAME)

setup:
	@./bin/_brew_install

install: uninstall
	@./bin/_pyenv_install $(VIRTUALENV_NAME)

update:
	@./bin/_pyenv_update

build : test
	@./bin/_build_library

test:
	@python -m pytest tests/

release-test: build
	@./bin/_release_test

release-prod:
	@./bin/_release_prod

clean:
	@./bin/_cleanup
