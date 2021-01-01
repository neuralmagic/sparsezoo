BUILDDIR := $(PWD)
CHECKDIRS := examples tests src utils
DOCDIR := docs

# run checks on all files for the repo
quality:
	black --check $(CHECKDIRS);
	isort --check-only $(CHECKDIRS);
	flake8 $(CHECKDIRS);

# style the code according to accepted standards for the repo
style:
	black $(CHECKDIRS);
	isort $(CHECKDIRS);

# run tests for the repo
test:
	@pytest;

# create docs
docs:
	sphinx-apidoc -o "$(DOCDIR)/source/" src/sparsezoo;
	cd $(DOCDIR) && $(MAKE) html;

# clean package
clean:
	rm -fr .pytest_cache;
	rm -fr docs/_build docs/build;
	rm -fr node_modules;
	find . -not -path "./.venv/*" | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -fr;
	find . -not -path "./.venv/*" | grep .rst | xargs rm -fr;
