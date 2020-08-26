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

run-piea-cpu:
	docker run --rm \
	-v $(CURDIR):/app \
	-u $(shell id -u):$(shell id -g) \
	${NAME} \
	python -m piea.app --guider=mobilenet \
        --tgt=demo/output \
        --index=index.txt \
        --loss=2 \
        --src=demo/input 


run-piea-gpu:
	docker run --rm \
	-v $(CURDIR):/app \
	--gpus all \
	-u $(shell id -u):$(shell id -g) \
	${NAME} \
	python -m piea.app --guider=mobilenet \
        --tgt=demo/output \
        --index=index.txt \
        --loss=2 \
        --src=demo/input 
		