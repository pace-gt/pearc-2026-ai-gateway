[![DOI](https://zenodo.org/badge/1153701571.svg)](https://doi.org/10.5281/zenodo.18554180)

# PEARC 2026: Supporting Materials for GT AI Gateway

## Overview

This repo contains config files and code to support PACE's paper submission to PEARC 2026,  "Service-Oriented Architecture for an On-Premises AI Gateway Without an On-Premises Cloud" (under review).  See the paper for detailed explanations.

Please note that these public artifacts are not copied verbatim from the versions deployed at our site.  In particular, directory paths and hostnames have been anonymized and secrets have been omitted.  Sites attempting to use our configurations and scripts should adapt them as needed.

## Contents

* `gateway/` contains the config files for LiteLLM and PostgreSQL
  - `config.yaml` is the LiteLLM config file
  - `litellm-postgres.pod` is the systemd unit for the pod that contains the LiteLLM and PostgreSQL containers
  - `litellm.container` is the systemd unit for the LiteLLM container
  - `postgres.container` is the sytemd unit for the PostgreSQL container
* `load-testing/` contains config files and scripts for peforming load testing via Locust
  - `locust.conf` is the Locust config file
  - `locust.py` is the script to run the load testing
* `triton/` contains the config files and scripts for running Triton Inference Server
  - `run-triton-server.sh` runs Triton and the API adapter via an Apptainer instance
  - `triton_sd_adapter.py` is the FastAPI-based API adapter
  - `triton.def` is the Apptainer `.def` file used to build the Apptainer image containing both Triton and the API adapter
* `vllm/` contains the config files and scripts for running vLLM
  - `run-vllm-apptainer.sh` runs vLLM via an Apptainer instance
  - `vllm.def` is the Apptainer `.def` file uses to build the Apptainer image for vLLM



