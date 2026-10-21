#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = supply_chain_resilience
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python
DATASETS = dataco olist

#################################################################################
# ENVIRONNEMENT                                                                 #
#################################################################################

## Créer l'environnement conda du projet
.PHONY: create_environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> Activer avec : conda activate $(PROJECT_NAME)"

## Installer les dépendances
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

#################################################################################
# QUALITÉ                                                                       #
#################################################################################

## Vérifier le style (ruff)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Formater le code (ruff)
.PHONY: format
format:
	ruff check --fix
	ruff format

## Lancer les tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests

## Supprimer les fichiers Python compilés
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# PIPELINE (CRISP-DM)                                                           #
#################################################################################

## Data Preparation : données brutes -> data/processed/<jeu>.parquet, puis variables d'historique -> <jeu>_features.parquet
.PHONY: data
data:
	for d in $(DATASETS); do \
		$(PYTHON_INTERPRETER) -m $(PROJECT_NAME).preprocessing --dataset $$d && \
		$(PYTHON_INTERPRETER) -m $(PROJECT_NAME).features --dataset $$d || exit 1; \
	done

## Modeling : entraînement et Optuna -> models/ et reports/results/
.PHONY: train
train:
	for d in $(DATASETS); do $(PYTHON_INTERPRETER) -m $(PROJECT_NAME).train --dataset $$d; done

## Evaluation et Deployment : métriques, écarts, dérive, coût -> reports/results/
.PHONY: evaluate
evaluate:
	for d in $(DATASETS); do $(PYTHON_INTERPRETER) -m $(PROJECT_NAME).evaluate --dataset $$d; done

## Figures de l'article -> reports/figures/
.PHONY: figures
figures:
	$(PYTHON_INTERPRETER) -m $(PROJECT_NAME).plots

## Pipeline complet, de bout en bout
.PHONY: all
all: data train evaluate figures

#################################################################################
# AIDE                                                                          #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Commandes disponibles :\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
