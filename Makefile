.PHONY: build docs test

BUILDDIR := $(PWD)
PYCHECKDIRS := examples tests src utils scripts notebooks
JSCHECKDIRS := src
DOCDIR := docs

# run checks on all files for the repo
quality:
	@echo "Running python quality checks";
	black --check $(PYCHECKDIRS);
	isort --check-only $(PYCHECKDIRS);
	flake8 $(PYCHECKDIRS);
	@echo "Running js/jsx quality checks";
	yarn prettier --check $(JSCHECKDIRS);

# style the code according to accepted standards for the repo
style:
	@echo "Running python styling";
	black $(PYCHECKDIRS);
	isort $(PYCHECKDIRS);
	@echo "Running js/jsx styling";
	yarn prettier --write $(JSCHECKDIRS);

# run tests for the repo
test:
	@echo "Running python tests";
	@pytest;

# create docs
docs:
	sphinx-apidoc -o "$(DOCDIR)/source/" src/sparsezoo;
	cd $(DOCDIR) && $(MAKE) html;

# creates wheel file
build:
	python3 setup.py sdist bdist_wheel

# clean package
clean:
	rm -fr .pytest_cache;
	rm -fr docs/_build docs/build;
	rm -fr node_modules;
	find . -not -path "./.venv/*" | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -fr;
	find . -not -path "./.venv/*" | grep .rst | xargs rm -fr;
