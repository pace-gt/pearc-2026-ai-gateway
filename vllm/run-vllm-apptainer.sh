#!/usr/bin/bash
set -e -o pipefail

# =============================================================================
#
# This script requires that the following env vars are defined for your site:
# 
#   APPTAINER_CONFIGDIR : Per-user Apptainer configuration, see https://apptainer.org/docs/user/latest/appendix.html#apptainer-s-environment-variables
#   CUDA_VISIBLE_DEVICES : Devices to use for this vLLM instance, see https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-visible-devices
#   HF_TOKEN : Auth token to download models from huggingface, see https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hftoken
#   VLLM_CACHE_ROOT : Directory for vLLM cache files, see https://docs.vllm.ai/en/v0.6.2/serving/env_vars.html
#   VLLM_SIF_IMAGE : Path to .sif image for vLLM
#   LITELLM_BASE_URL : Host and port to your site's LiteLLM instance, e.g. https://litellm.my-university.edu:8000
#   MODEL_MGMT_ACCT_KEY : A previously-generated API key for updating LiteLLM's model database.  It's recommended to use RBAC
#
# =============================================================================

function print_help {
  # Show a help message
  echo "\
Start or stop a vLLM instance on the localhost with Apptainer.  Valid subcommands:

    $(basename "$0") start --config CONFIG_YAML [--vllm-logging-config-path LOGGING_CONFIG_JSON] [...]

        Start a vLLM instance with the settings from CONFIG_YAML
	* CONFIG_YAML is the vLLM config file.  It includes the model name and other parameters
	  for running the model.  See: 
          https://docs.vllm.ai/en/stable/configuration/serve_args.html?h=config+file
	* LOGGING_CONFIG_JSON is a JSON for configuring vLLM's logging.  See:
	  https://docs.vllm.ai/en/stable/examples/others/logging_configuration.html

  
    $(basename "$0") stop --config CONFIG_YAML [...]
  
        Stop a vLLM instance that was spawned from CONFIG_YAML
	* CONFIG_YAML is the vLLM config file.  It includes the model name and other parameters
	  for running the model.
    
    $(basename "$0") list

        List all the vLLM instances that are running. Prints JSON output 
  
    $(basename "$0") shell
  
        Start a shell session with the vLLM container with no model setup.
        Useful for setting up credentials correctly, like with huggingface-cli login
    
    $(basename "$0") help

        Show this help message and exit

Here are some environment variables that affect Apptainer and vLLM. These are optional and
have reasonble defaults defined in the script.

    APPTAINER_CONFIGDIR
    CUDA_VISIBLE_DEVICES
    HF_HOME
    VLLM_API_KEY
    VLLM_CACHE_ROOT
    VLLM_LOGGING_CONFIG_PATH

Here are some optional command-line args.  They take precedence over corresponding env variables.

    --apptainer-configdir DIR
    --cuda-visible-devices LIST
    --hf-home DIR
    --vllm-cache-root DIR
    --vllm-sif-image PATH
    --litellm-base-url URL
    --vllm-logging-config-path PATH
  "
}

function main {
  # Runs this script
  subcmd="$1"

  case $subcmd in
    start | stop | list | shell )
      ;;
    help | -h | --help )
      print_help
      exit 0
      ;;
    * )
      echo "[ERROR] Missing or invalid subcommand.  See usage with --help"
      exit 1
      ;;
  esac
  shift

  while getopts ':h-:' VAL; do
    case "$VAL" in 
      h )
        print_help
        exit 0
        ;;
      - )
        case "$OPTARG" in 
          help )
            print_help
            exit 0
            ;;
          config )
            config="${!OPTIND}"
            ((OPTIND++))
            ;;
          apptainer-configdir )
            export APPTAINER_CONFIGDIR="${!OPTIND}"
            ((OPTIND++))
            ;;
          cuda-visible-devices )
            export CUDA_VISIBLE_DEVICES="${!OPTIND}"
            ((OPTIND++))
            ;;
          hf-home )
            export HF_HOME="${!OPTIND}"
            ((OPTIND++))
            ;;
          vllm-cache-root )
            export VLLM_CACHE_ROOT="${!OPTIND}"
            ((OPTIND++))
            ;;
          vllm-logging-config-path )
            export VLLM_LOGGING_CONFIG_PATH="${!OPTIND}"
            ((OPTIND++))
            ;;
          vllm-sif-image )
            export VLLM_SIF_IMAGE="${!OPTIND}"
            ((OPTIND++))
            ;;
          litellm-base-url )
            export LITELLM_BASE_URL="${!OPTIND}"
            ((OPTIND++))
            ;;
          * )
            echo "[ERROR] Invalid arg: --$OPTARG . See usage with --help"
            exit 1
            ;;
        esac
        ;;
      * ) 
        echo "[ERROR] Invalid arg: -$OPTARG . See usage with --help" 
        exit 1
        ;;
    esac
  done

  if [[ $subcmd == start || $subcmd == stop ]] && [[ -z $config ]]; then
    echo "[ERROR] Did not specify a --config . See usage with --help"
    exit 1
  fi

  # This helps disambiguate Apptainer logs
  export APPTAINER_CONFIGDIR="$APPTAINER_CONFIGDIR/$(config_to_instance "$config")"

  echo "[INFO] APPTAINER_CONFIGDIR='$APPTAINER_CONFIGDIR'"
  echo "[INFO] CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"
  echo "[INFO] HF_HOME='$HF_HOME'"
  echo "[INFO] LITELLM_BASE_URL='$LITELLM_BASE_URL'"
  echo "[INFO] VLLM_CACHE_ROOT='$VLLM_CACHE_ROOT'"
  echo "[INFO] VLLM_LOGGING_CONFIG_PATH='$VLLM_LOGGING_CONFIG_PATH'"
  echo "[INFO] VLLM_SIF_IMAGE='$VLLM_SIF_IMAGE'"

  case $subcmd in
    start ) start_vllm "$config" ;;
    stop ) stop_vllm "$config" ;;
    list ) list_vllm ;;
    shell ) shell_vllm ;;
  esac
}

function start_vllm {
  # Starts a vLLM instance for the given model
  config="$1"
  instance="$(config_to_instance "$config")"
  model="$(config_to_model "$config")"

  # Stop the instance if it's already running
  if [[ $(instance_exists "$config") = "true" ]]; then
    echo "[WARN] There is already an instance running for $config .  Stopping instance now..."
    stop_vllm "$config"
  fi

  #echo "[INFO] Hostname is: $(hostname)"

  # Choose an open port
  port=$(find_port)
  #echo "[INFO] Using port: $port"

  # Generate an API key, if it's not already set in the env
  if [[ -z $VLLM_API_KEY ]]; then
    export VLLM_API_KEY=$(create_passwd)
  fi
  #echo "[INFO] Using API key: $VLLM_API_KEY"

  # Create APPTAINER_CONFIGDIR if it doesn't exist
  mkdir -p "$APPTAINER_CONFIGDIR"

  # Start the instance
  apptainer instance run \
    --nv \
    --cleanenv \
    --env "HF_HOME=$HF_HOME" \
    --env "HF_HUB_CACHE=/tmp" \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT" \
    --env "VLLM_API_KEY=$VLLM_API_KEY" \
    --env "VLLM_LOGGING_CONFIG_PATH=$VLLM_LOGGING_CONFIG_PATH" \
    --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
    "$VLLM_SIF_IMAGE" \
    "$instance" \
    --config "$config" \
    --port "$port"
  
  # Detect any errors raised by apptainer
  if [[ $? -ne 0 ]]; then
    echo "[ERROR] Apptainer instance for $config did not start successfully"
    exit 1
  fi

  # Display the stdout and stderr files that Apptainer is using
  outfile=$(apptainer instance list --json | jq ".instances|INDEX(.instance).${instance}.logOutPath")
  echo "[INFO] STDOUT logfile: $outfile"
  errfile=$(apptainer instance list --json | jq ".instances|INDEX(.instance).${instance}.logErrPath")
  echo "[INFO] STDERR logfile: $errfile"

  # Unlike Docker or Podman, Apptainer doesn't actually have the means to verify whether a service 
  # in an Apptainer instance has started correctly.  Here, we verify that vLLM has started correctly
  # by polling it with an API request
  timeout=1200 # in seconds
  echo -n "[INFO] Apptainer doesn't actually know if vLLM is running successfully. To verify, let's poll vLLM using the REST API and timeout after ${timeout} seconds ..."
  if poll_vllm_health "$port" "$timeout"; then
    echo -e "\n[INFO] Success! vLLM responded to a REST API request"
  else
    echo -e "\n[ERROR] Uh-oh! vLLM did not respond any REST API requests after $timeout seconds. You might want to check the Apptainer logs and/or stop the instance"
    exit 1
  fi

  # Update or add model to LiteLLM DB
  litellm_update_model "$port" "$model"
}

function list_vllm {
  # Show all the Apptainer instances
  apptainer instance list --json
}

function shell_vllm {
  echo "[INFO] Hostname is: $(hostname)"

  # Generate an API key, if it's not already set in the env
  if [[ -z $VLLM_API_KEY ]]; then
    export VLLM_API_KEY=$(create_passwd)
  fi
  echo "[INFO] Using API key: $VLLM_API_KEY"

  # Start the instance
  apptainer shell \
    --nv \
    --cleanenv \
    --env "HF_HOME=$HF_HOME" \
    --env "HF_HUB_CACHE=/tmp" \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT" \
    --env "VLLM_API_KEY=$VLLM_API_KEY" \
    --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
    "$VLLM_SIF_IMAGE"
}

function instance_exists {
  # Output "true" if the instance for the given model exists
  config="$1"
  instance="$(config_to_instance "$config")"
  apptainer instance list --json | jq ".instances | INDEX(.instance) | has(\"${instance}\")"
}

function stop_vllm {
  # Stops a given vLLM instance
  config="$1"
  instance="$(config_to_instance "$config")"
  if [[ $(instance_exists "$config") = "true" ]] ; then
    apptainer instance stop "$instance"
    if [[ $? -eq 0 ]]; then
      echo "[INFO] Apptainer instance for $config was successfully stopped"
    else
      echo "[ERROR] Apptainer instance for $config was not successfully stopped"
      exit 1
    fi
  else
    echo "[WARN] There is no active instance for $config"
  fi
}

function config_to_instance {
  # Santize the name of a config file so Apptainer and jq are both happy with it
  echo  "$(basename $(basename $1 .yml) .yaml)" | tr './-' '_'
}

function config_to_model {
  # Extract model name from config YAML
  grep -oP 'model:\s+\K(.*)' "$1"
}

function find_port {
  # Finds an unused port. Adapted from Open OnDemand (https://github.com/OSC/ondemand)
  host=localhost
  min_port=49152
  max_port=65535
  port_range=($(shuf -i ${min_port}-${max_port}))
  for port in "${port_range[@]}"; do
    if port_used "${host}:${port}"; then
      continue
    fi
    echo "${port}"
    return 0 # success
  done
  echo "error: failed to find available port in range ${min_port}..${max_port}" >&2
  return 1 # failure
}

function port_used {
  # Verifies if a port is in use. Adapted from Open OnDemand (https://github.com/OSC/ondemand)
  port="${1#*:}"
  host=$((expr "${1}" : '\\(.*\\):' || echo "localhost") | awk 'END{print $NF}')
  port_used_nc "$host" "$port"
  status=$?
  if [[ "$status" == "0" ]] || [[ "$status" == "1" ]]; then
    return $status
  fi
  return 127
}

function port_used_nc {
  # Runs nc to see if a port is in use. Adapted from Open OnDemand(https://github.com/OSC/ondemand)
  nc -w 2 "$1" "$2" < /dev/null > /dev/null 2>&1
}

function create_passwd {
  # Creates an alphanumeric password. Adapted from Open OnDemand(https://github.com/OSC/ondemand)
  tr -cd 'a-zA-Z0-9' < /dev/urandom 2> /dev/null | head -c 32
}

function poll_vllm_health {
  # This polls a vLLM instance on localhost at a given port using the /health API endpoint.  
  # It polls every 2 seconds until the timeout is reached.  If the endpoint responds before
  # the timeout, return 0.  If not, return 1.
  port="$1"
  timeout="$2"
  start=$(date +%s)
  end=$((start + timeout))

  while [[ $(date +%s) -le $end ]]; do
    curl --fail --silent -X GET "http://0.0.0.0:${port}/health" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $VLLM_API_KEY"
    if [[ $? -eq 0 ]]; then
        return 0
    fi
    echo -n "."
    sleep 2
  done
  return 1
}

function litellm_new_model {
  # Add model to LiteLLM DB
  port="$1"
  model="$2"
  curl -X POST "$LITELLM_BASE_URL/model/new" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $MODEL_MGMT_ACCT_KEY" \
      --data  "{
          \"model_name\": \"${model}\",
          \"litellm_params\": {
              \"model\": \"hosted_vllm/${model}\",
              \"api_base\": \"http://$(hostname):${port}/v1\",
              \"api_key\": \"${VLLM_API_KEY}\",
              \"model_info\": {
                  \"litellm_provider\": \"hosted_vllm\"
              }
          }
      }" | jq
}

function litellm_delete_model {
  # Delete a named model from LiteLLM DB
  model="$1"

  found_ids=$(curl -X GET "$LITELLM_BASE_URL/model/info" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $MODEL_MGMT_ACCT_KEY" \
    | jq ".data | .[] | select(.model_name==\"${model}\") | .model_info.id")

  for id in $found_ids; do
    curl -X POST "$LITELLM_BASE_URL/model/delete" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $MODEL_MGMT_ACCT_KEY" \
        --data  "{
            \"id\": $id
        }" | jq
  done
}

function litellm_update_model {
  # Add model to LiteLLM DB
  port="$1"
  model="$2"

  # Get the any IDs with this model name
  found_ids=$(curl -X GET "$LITELLM_BASE_URL/model/info" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $MODEL_MGMT_ACCT_KEY" \
    | jq ".data | .[] | select(.model_name==\"${model}\") | .model_info.id")

  if [[ -z $found_ids ]]; then
    # If there aren't any models with this name, add a new one
    echo "[INFO] Adding new model '${model}' to ${LITELLM_BASE_URL}"
    curl -X POST "$LITELLM_BASE_URL/model/new" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $MODEL_MGMT_ACCT_KEY" \
        --data  "{
            \"model_name\": \"${model}\",
            \"litellm_params\": {
                \"model\": \"hosted_vllm/${model}\",
                \"api_base\": \"http://$(hostname):${port}/v1\",
                \"api_key\": \"${VLLM_API_KEY}\",
                \"model_info\": {
                    \"litellm_provider\": \"hosted_vllm\"
                }
            }
        }" | jq
  else
    # If there are any models with this name, update the host, port, and key
    for id in $found_ids; do 
      echo "[INFO] Updating existing model '${model}' at ${LITELLM_BASE_URL}"
      # Strip of quotes from JSON field
      id="${id%\"}"
      id="${id#\"}"
      curl -X PATCH "$LITELLM_BASE_URL/model/$id/update" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $MODEL_MGMT_ACCT_KEY" \
        --data "{ 
          \"litellm_params\": {
              \"api_base\": \"http://$(hostname):${port}/v1\",
              \"api_key\": \"${VLLM_API_KEY}\"
            }
          }" | jq
    done
  fi
}

# Run the main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
  sleep infinity
fi

