.PHONY: build docs test

BUILDDIR := $(PWD)
PYCHECKDIRS := examples tests src utils scripts notebooks
PYCHECKGLOBS := 'examples/**/*.py' 'scripts/**/*.py' 'src/**/*.py' 'tests/**/*.py' 'utils/**/*.py' setup.py
DOCDIR := docs
MDCHECKGLOBS := 'docs/**/*.md' 'docs/**/*.rst' 'examples/**/*.md' 'notebooks/**/*.md' 'scripts/**/*.md'
MDCHECKFILES := CODE_OF_CONDUCT.md CONTRIBUTING.md DEVELOPING.md README.md

BUILD_ARGS :=  # set nightly to build nightly release
TARGETS := ""  # targets for running pytests: full,efficientnet,inception,resnet,vgg,ssd,yolo
PYTEST_ARGS ?= ""
ifneq ($(findstring full,$(TARGETS)),full)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/sparsezoo/models/test_zoo_extensive.py
endif
ifneq ($(findstring efficientnet,$(TARGETS)),efficientnet)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/sparsezoo/models/classification/test_efficientnet.py
endif
ifneq ($(findstring inception,$(TARGETS)),inception)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/sparsezoo/models/classification/test_inception.py
endif
ifneq ($(findstring resnet,$(TARGETS)),resnet)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/sparsezoo/models/classification/test_resnet.py
endif
ifneq ($(findstring vgg,$(TARGETS)),vgg)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/sparsezoo/models/classification/test_vgg.py
endif
ifneq ($(findstring ssd,$(TARGETS)),ssd)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/sparsezoo/models/detection/test_ssd.py
endif
ifneq ($(findstring yolo,$(TARGETS)),yolo)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/sparsezoo/models/detection/test_yolo.py
endif


# run checks on all files for the repo
quality:
	@echo "Running copyright checks";
	python utils/copyright.py quality $(PYCHECKGLOBS) $(JSCHECKGLOBS) $(MDCHECKGLOBS) $(MDCHECKFILES)
	@echo "Running python quality checks";
	black --check $(PYCHECKDIRS);
	isort --check-only $(PYCHECKDIRS);
	flake8 $(PYCHECKDIRS);

# style the code according to accepted standards for the repo
style:
	@echo "Running copyrighting";
	python utils/copyright.py style $(PYCHECKGLOBS) $(JSCHECKGLOBS) $(MDCHECKGLOBS) $(MDCHECKFILES)
	@echo "Running python styling";
	black $(PYCHECKDIRS);
	isort $(PYCHECKDIRS);

# run tests for the repo
test:
	@echo "Running python tests";
	SPARSEZOO_TEST_MODE="true" NM_DISABLE_ANALYTICS="true" pytest tests $(PYTEST_ARGS);

# create docs
docs:
	@echo "Running docs creation";
	python utils/docs_builder.py --src $(DOCDIR) --dest $(DOCDIR)/build/html;

docsupdate:
	@echo "Runnning update to api docs";
	find $(DOCDIR)/api | grep .rst | xargs rm -rf;
	sphinx-apidoc -o "$(DOCDIR)/api" src/sparsezoo;

# creates wheel file
build:
	@echo "Building python package";
	python3 setup.py sdist bdist_wheel $(BUILD_ARGS)

# clean package
clean:
	rm -rf .pytest_cache;
	rm -rf docs/_build docs/build;
	rm -rf build;
	rm -rf dist;
	rm -rf src/sparsezoo.egg-info;
	find $(PYCHECKDIRS) | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf;
