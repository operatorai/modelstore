VIRTUALENV_NAME=$(shell pwd | rev | cut -d '/' -f 1 | rev)-dev

.PHONY: library test install uninstall release-test release-prod clean

uninstall:
	@./bin/_remove_pyenv

install: uninstall
	@./bin/_setup_brew
	@./bin/_setup_pyenv $(VIRTUALENV_NAME)

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
