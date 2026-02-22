include ../Makefile-example

DOCKER_IMAGE ?= modelstore-examples
MODELSTORE_IN ?= filesystem
ML_FRAMEWORK ?= sklearn

.PHONY: run docker-build docker-run

run:
	@./run-all.sh

docker-build:
	@docker build -f Dockerfile -t $(DOCKER_IMAGE) .

docker-run:
	@docker run --rm \
		-e MODEL_STORE_ROOT_PREFIX=/tmp/modelstore \
		$(DOCKER_IMAGE) \
		--modelstore-in $(MODELSTORE_IN) \
		--ml-framework $(ML_FRAMEWORK)
