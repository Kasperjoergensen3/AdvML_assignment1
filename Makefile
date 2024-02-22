.PHONY: create_environment install_requirements clean run_partA

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = AdvML1
PYTHON_VERSION = 3.11
ifeq ($(OS),Windows_NT)
    PYTHON_INTERPRETER = python
else
    PYTHON_INTERPRETER = $(shell command -v python3 || command -v python)
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) --no-default-packages -y

install_requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

run_partB:
#$(PYTHON_INTERPRETER) AMLsrc/train_PartB.py --config "AMLsrc/configs/partB_DDPM.yaml" --version v1
#$(PYTHON_INTERPRETER) AMLsrc/train_PartB.py --config "AMLsrc/configs/partB_Flow.yaml" --version v1
	$(PYTHON_INTERPRETER) AMLsrc/train_PartB.py --config "AMLsrc/configs/partB_VAE.yaml" --version v1
eval_partB:
	$(PYTHON_INTERPRETER) AMLsrc/summarize_partB.py
run_partA:
	sh partA.sh

partA_figures:
	$(PYTHON_INTERPRETER) src/summarize_partA.py


