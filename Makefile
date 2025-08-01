DATA_DIR := data
BASE_URL := https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1
PREFIX := QuALITY.v1.0.1.htmlstripped

PARTS := train val test

download-data:
	mkdir -p $(DATA_DIR)
	for part in $(PARTS); do \
		curl -L $(BASE_URL)/$(PREFIX).$$part -o $(DATA_DIR)/$(PREFIX).$$part; \
	done