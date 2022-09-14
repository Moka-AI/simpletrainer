SHELL := /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "style        : run black & isort."
	@echo "lint         : run pyright to lint the code."
	@echo "test         : run pytest cases."
	@echo "requirement  : generate requirements.txt file."
	@echo "push         : prepare for git push."
	@echo "publish      : publish the package to pypi."
	@echo "clean        : cleans all unnecessary files."

.PHONY: style
style:
	isort --profile black .
	blue -l 128 .

.PHONY: lint
lint:
	pyright .

.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -rf dist
	rm -f .coverage
	rm -rf htmlcov

.PHONY: requirement
requirement:
	poetry export --without-hashes > requirements.txt
	poetry export --only dev --without-hashes > dev-requirements.txt

.PHONY: push
push: style clean requirement
	echo "Ready to push"

.PHONY: publish
publish: style clean requirement
	poetry publish --build

.PHONY: test
test:
	python -m pytest --cov simpletrainer -v -p no:warnings .
