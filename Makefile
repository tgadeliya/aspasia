DATA_DIR := data
BASE_URL := https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1
PREFIX := QuALITY.v1.0.1.htmlstripped
PARTS := train dev test
RUFF_LINE_LENGTH := 88

download-data:
	mkdir -p $(DATA_DIR)
	for part in $(PARTS); do \
		curl -L $(BASE_URL)/$(PREFIX).$$part -o $(DATA_DIR)/$(PREFIX).$$part; \
	done

fix: lint_docs ruff_fmt ruff_check

ruff_fmt:
	uvx ruff format 

ruff_check:
	uvx ruff check

lint_docs:
	uvx docformatter --in-place -r \
		--wrap-summaries=$(RUFF_LINE_LENGTH) \
		--wrap-descriptions=$(RUFF_LINE_LENGTH) \
		.