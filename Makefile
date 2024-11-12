.DEFAULT_GOAL := help

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  reinstall_package  to install/reinstall the package"

#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y deepmicro || :
	@pip install -e .