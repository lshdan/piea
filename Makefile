NAME=piea

.PHONY: build

build:
	docker build -t ${NAME} .

run-devel:
	docker run --name ${NAME}-dev \
	-d -v $(CURDIR):/app \
	-v /tmp:/tmp \
	-u $(shell id -u):$(shell id -g) \
	${NAME} sleep infinity
