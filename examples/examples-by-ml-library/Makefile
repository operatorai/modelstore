include ../Makefile-example

.PHONY: install run

install: uninstall
	@uv venv
	@uv pip install --upgrade pip setuptools wheel
	@uv pip install -r requirements.txt
	@uv pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

run:
	@./run-all.sh
	