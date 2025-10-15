# Makefile for PyNexus development

.PHONY: help install test clean docs format check

help:
	@echo "PyNexus Development Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make install     - Install package in development mode"
	@echo "  make test        - Run tests"
	@echo "  make format      - Format code with black and isort"
	@echo "  make check       - Check code style with flake8"
	@echo "  make docs        - Build documentation"
	@echo "  make clean       - Clean build artifacts"

install:
	pip install -e .[dev]

test:
	pytest

format:
	black pynexus tests
	isort pynexus tests

check:
	flake8 pynexus tests
	black --check pynexus tests
	isort --check-only pynexus tests

docs:
	@echo "Documentation building not yet configured"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete