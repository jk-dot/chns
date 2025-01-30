SHELL := /bin/zsh

clean:
	@setopt nullglob; \
	for dir in output; do \
		rm -f $$dir/*.{out,log,fls,blg,fdb_latexmk,aux,bbl,bcf,run.xml,synctex.gz}; \
	done

# Variables
IMAGE_NAME = firedrake-zsh
CONTAINER_NAME = firedrake-container
SHARED_DIR = $(shell pwd)/shared

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the Docker container interactively
run:
	docker run -it --rm \
		--name $(CONTAINER_NAME) \
		-v $(SHARED_DIR):/home/firedrake/shared \
		$(IMAGE_NAME)

# Run the Docker container with X11 forwarding (for GUI applications)
run-gui:
	docker run -it --rm \
		--name $(CONTAINER_NAME) \
		-v $(SHARED_DIR):/home/firedrake/shared \
		-e DISPLAY=$(DISPLAY) \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		$(IMAGE_NAME)

# Help message
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build      Build the Docker image"
	@echo "  run        Run the Docker container interactively"
	@echo "  run-gui    Run the Docker container with X11 forwarding (for GUI applications)"
	@echo "  clean      Remove the Docker image"
	@echo "  help       Show this help message"