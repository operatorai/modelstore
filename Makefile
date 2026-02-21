.PHONY: uninstall
uninstall:
	@./bin/_uv_uninstall

.PHONY: setup
setup:
	@./bin/_brew_install

.PHONY: install
install: uninstall
	@./bin/_uv_install

.PHONY: update
update:
	@./bin/_uv_update

.PHONY: build
build:
	@./bin/_build_library

.PHONY: test
test:
	@docker build . -t modelstore-dev
	@docker run -it --rm modelstore-dev

.PHONY: release-test
release-test: build
	@./bin/_release_test

.PHONY: release-prod
release-prod:
	@./bin/_release_prod

.PHONY: cleanup
clean:
	@./bin/_cleanup
