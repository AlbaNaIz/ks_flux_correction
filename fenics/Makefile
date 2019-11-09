PYTHON = python3
TESTS := $(wildcard ks_test_*.py) $(wildcard fc-ks_test_*.py)

.PHONY: $(TESTS)

# Run all tests
test: $(TESTS)

# Rule for each .py test
${CURDIR}/$(TESTS):
	$(PYTHON) $@
	@echo "..."
