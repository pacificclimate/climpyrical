export PIP_INDEX_URL=https://pypi.pacificclimate.org/simple

# Setup environment
ifeq ($(TMPDIR),)
VENV_PATH := /tmp/climpyrical-venv
else
VENV_PATH := $(TMPDIR)/climpyrical-venv
endif

# Makefile Vars
SHELL:=/bin/bash
PYTHON=${VENV_PATH}/bin/python3
PIP=${VENV_PATH}/bin/pip

.PHONY: all
all: apt install test

.PHONY: apt
apt:
	sudo apt-get install -y \
		proj-bin \
		libproj-dev \
		r-base

.PHONY: clean
clean:
	rm -rf $(VENV_PATH)

.PHONY: install
install: venv
	${PIP} install -U pip
	${PIP} install -r requirements.txt -r test_requirements.txt
	${PIP} install -e .
	Rscript install_pkgs.R r_requirements.txt

.PHONY: docs
docs: venv
	${PIP} install pdoc3==0.9.1
	source ${VENV_PATH}/bin/activate && \
	pdoc --html -o docs --force climpyrical/*.py

.PHONY: test
test: venv
	${PYTHON} -m pytest -v

.PHONY: venv
venv:
	test -d $(VENV_PATH) || python3 -m venv $(VENV_PATH)
