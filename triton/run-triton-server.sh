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
#   TRITON_SIF_IMAGE : Path to .sif image for Triton
#   LITELLM_BASE_URL : Host and port to your site's LiteLLM instance, e.g. https://litellm.my-university.edu:8000
#   MODEL_MGMT_ACCT_KEY : A previously-generated API key for updating LiteLLM's model database.  It's recommended to use RBAC
#
# =============================================================================

if [[ ! -d "$HF_HOME" ]]; then
  echo "[ERROR] HF_HOME directory does not exist: $HF_HOME"
  echo "[INFO] Creating HF_HOME directory..."
  mkdir -p "$HF_HOME"
fi

function print_help {
  # Show a help message
  echo "\
Start or stop a Triton + SD Adapter instance on the localhost with Apptainer.  Valid subcommands:
    $(basename "$0") start --config CONFIG_YAML [...]
        Start a Triton + SD Adapter instance with the settings from CONFIG_YAML
	* CONFIG_YAML is the Triton config file.  It includes the model name, repository path, and other parameters
	  for running the model.

    $(basename "$0") stop --config CONFIG_YAML [...]
        Stop a Triton + SD Adapter instance that was spawned from CONFIG_YAML
	* CONFIG_YAML is the Triton config file.  It includes the model name, repository path, and other parameters
	  for running the model.

    $(basename "$0") list
        List all the Triton + SD Adapter instances that are running. Prints JSON output

    $(basename "$0") shell
        Start a shell session with the Triton container with no model setup.
        Useful for setting up credentials correctly, like with huggingface-cli login

    $(basename "$0") help
        Show this help message and exit
Here are some environment variables that affect Apptainer and Triton. These are optional and
have reasonable defaults defined in the script.
    APPTAINER_CONFIGDIR
    CUDA_VISIBLE_DEVICES
    HF_HOME
    TRITON_ADAPTER_API_KEY
Here are some optional command-line args.  They take precedence over corresponding env variables.
    --apptainer-configdir DIR
    --config PATH (required for start/stop)
    --cuda-visible-devices LIST
    --hf-home DIR
    --triton-sif-image PATH
    --litellm-base-url URL
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
          triton-sif-image )
            export TRITON_SIF_IMAGE="${!OPTIND}"
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
  if [[ $subcmd == start || $subcmd == stop ]]; then
    export APPTAINER_CONFIGDIR="$APPTAINER_CONFIGDIR/$(config_to_instance "$config")"
  fi
  echo "[INFO] APPTAINER_CONFIGDIR='$APPTAINER_CONFIGDIR'"
  echo "[INFO] CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"
  echo "[INFO] HF_HOME='$HF_HOME'"
  echo "[INFO] LITELLM_BASE_URL='$LITELLM_BASE_URL'"
  echo "[INFO] TRITON_SIF_IMAGE='$TRITON_SIF_IMAGE'"
  case $subcmd in
    start ) start_triton "$config" ;;
    stop ) stop_triton "$config" ;;
    list ) list_triton ;;
    shell ) shell_triton ;;
  esac
}

function config_to_instance {
  # Sanitize the name of a config file so Apptainer and jq are both happy with it
  echo  "$(basename $(basename $1 .yml) .yaml)" | tr './-' '_'
}

function config_to_model {
  # Extract model name from config YAML
  grep -oP 'model:\s+\K(.*)' "$1"
}

function config_to_repository {
  # Extract model repository path from config YAML
  grep -oP 'model_repository:\s+\K(.*)' "$1"
}

function config_to_hf_home {
  # Extract HF home path from config YAML
  grep -oP 'hf_home:\s+\K(.*)' "$1"
}

function start_triton {
  # Starts Triton server and SD adapter instances for the given model
  config="$1"
  instance="$(config_to_instance "$config")"
  model="$(config_to_model "$config")"
  model_repository="$(config_to_repository "$config")"
  hf_home_config="$(config_to_hf_home "$config")"

  # Override HF_HOME if specified in config
  if [[ -n $hf_home_config ]]; then
    export HF_HOME="$hf_home_config"
  fi
  
  # Stop the instance if it's already running
  if [[ $(instance_exists "$config") = "true" ]]; then
    echo "[WARN] There is already an instance running for $config .  Stopping instance now..."
    stop_triton "$config"
  fi
  
  # Choose open ports for Triton server and adapter
  triton_port=$(find_port)
  adapter_port=$(find_port)
  echo "[INFO] Using Triton port: $triton_port"
  echo "[INFO] Using Adapter port: $adapter_port"
  
  # Generate an API key for the adapter, if it's not already set in the env
  if [[ -z $TRITON_ADAPTER_API_KEY ]]; then
    export TRITON_ADAPTER_API_KEY=$(create_passwd)
  fi
  echo "[INFO] Using Adapter API key: $TRITON_ADAPTER_API_KEY"
  
  # Create APPTAINER_CONFIGDIR if it doesn't exist
  mkdir -p "$APPTAINER_CONFIGDIR"
  
  # Start Triton server instance
  echo "[INFO] Starting Triton server instance..."
  apptainer instance run \
    --nv \
    --cleanenv \
    --bind "${model_repository}:/workspace" \
    --bind "${HF_HOME}:/workspace/.cache/huggingface" \
    --env "HF_HOME=/workspace/.cache/huggingface" \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
    "$TRITON_SIF_IMAGE" \
    "${instance}-server" \
    tritonserver --model-repository /workspace/diffusion-models \
    --model-control-mode explicit \
    --load-model "$model" \
    --http-port "$triton_port" \
    --grpc-port "$((triton_port + 1))" \
    --http-address 0.0.0.0
  
  # Detect any errors raised by apptainer for server
  if [[ $? -ne 0 ]]; then
    echo "[ERROR] Apptainer instance for Triton server did not start successfully"
    exit 1
  fi
  
  # Start adapter instance
  echo "[INFO] Starting SD adapter instance..."
  apptainer instance run \
    --nv \
    --cleanenv \
    --bind "${model_repository}:/workspace" \
    --bind "${HF_HOME}:/workspace/.cache/huggingface" \
    --env "HF_HOME=/workspace/.cache/huggingface" \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
    --env "TRITON_URL=localhost:$triton_port" \
    --env "ADAPTER_PORT=$adapter_port" \
    --env "MODEL_NAME=$model" \
    "$TRITON_SIF_IMAGE" \
    "${instance}-adapter" \
    python3 /workspace/triton_sd_adapter.py
  
  # Detect any errors raised by apptainer for adapter
  if [[ $? -ne 0 ]]; then
    echo "[ERROR] Apptainer instance for SD adapter did not start successfully"
    exit 1
  fi
  
  # Display the stdout and stderr files that Apptainer is using
  server_outfile=$(apptainer instance list --json | jq --arg instance "$instance" '.instances[] | select(.instance == ($instance + "-server")) | .logOutPath')
  echo "[INFO] Triton server STDOUT logfile: $server_outfile"
  server_errfile=$(apptainer instance list --json | jq --arg instance "$instance" '.instances[] | select(.instance == ($instance + "-server")) | .logErrPath')
  echo "[INFO] Triton server STDERR logfile: $server_errfile"

  adapter_outfile=$(apptainer instance list --json | jq --arg instance "$instance" '.instances[] | select(.instance == ($instance + "-adapter")) | .logOutPath')
  echo "[INFO] Adapter STDOUT logfile: $adapter_outfile"
  adapter_errfile=$(apptainer instance list --json | jq --arg instance "$instance" '.instances[] | select(.instance == ($instance + "-adapter")) | .logErrPath')
  echo "[INFO] Adapter STDERR logfile: $adapter_errfile"
  
  # Unlike Docker or Podman, Apptainer doesn't actually have the means to verify whether a service 
  # in an Apptainer instance has started correctly.  Here, we verify that both Triton and adapter 
  # have started correctly by polling their health endpoints
  timeout=1200 # in seconds
  echo -n "[INFO] Apptainer doesn't actually know if Triton and adapter are running successfully. To verify, let's poll both services using their REST API and timeout after ${timeout} seconds ..."
  
  if poll_triton_health "$triton_port" "$timeout" && poll_adapter_health "$adapter_port" "$timeout"; then
    echo -e "\n[INFO] Success! Both Triton server and SD adapter responded to REST API requests"
  else
    echo -e "\n[ERROR] Uh-oh! Either Triton server or SD adapter did not respond to REST API requests after $timeout seconds. You might want to check the Apptainer logs and/or stop the instances"
    exit 1
  fi
  
  # Update or add model to LiteLLM DB
  litellm_update_model "$adapter_port" "$model" "$config"
}

function list_triton {
  # Show all the Apptainer instances
  apptainer instance list --json
}

function shell_triton {
  echo "[INFO] Hostname is: $(hostname)"
  # Generate an API key, if it's not already set in the env
  if [[ -z $TRITON_ADAPTER_API_KEY ]]; then
    export TRITON_ADAPTER_API_KEY=$(create_passwd)
  fi
  echo "[INFO] Using Adapter API key: $TRITON_ADAPTER_API_KEY"
  # Start the shell instance
  apptainer shell \
    --nv \
    --cleanenv \
    --bind "${HF_HOME}:/workspace/.cache/huggingface" \
    --env "HF_HOME=/workspace/.cache/huggingface" \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
    "$TRITON_SIF_IMAGE"
}

function instance_exists {
  # Output "true" if the instance for the given config exists
  config="$1"
  instance="$(config_to_instance "$config")"
  server_exists=$(apptainer instance list --json | jq ".instances | INDEX(.instance) | has(\"${instance}-server\")")
  adapter_exists=$(apptainer instance list --json | jq ".instances | INDEX(.instance) | has(\"${instance}-adapter\")")

  if [[ $server_exists == "true" && $adapter_exists == "true" ]]; then
    echo "true"
  else
    echo "false"
  fi
}

function stop_triton {
  # Stops a given Triton server and adapter instances
  config="$1"
  instance="$(config_to_instance "$config")"
  
  if [[ $(instance_exists "$config") = "true" ]] ; then
    echo "[INFO] Stopping Triton server instance..."
    apptainer instance stop "${instance}-server"
    if [[ $? -eq 0 ]]; then
      echo "[INFO] Triton server instance stopped successfully"
    else
      echo "[ERROR] Triton server instance was not stopped successfully"
    fi

    echo "[INFO] Stopping SD adapter instance..."
    apptainer instance stop "${instance}-adapter"
    if [[ $? -eq 0 ]]; then
      echo "[INFO] SD adapter instance stopped successfully"
    else
      echo "[ERROR] SD adapter instance was not stopped successfully"
    fi
  else
    echo "[WARN] There are no active instances for stable_diffusion_1_5"
  fi
}

# config_to_instance and config_to_model functions removed - model is hardcoded to stable_diffusion_1_5

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

function poll_triton_health {
  # This polls a Triton instance on localhost at a given port using the /v2/health/ready API endpoint.
  # It polls every 2 seconds until the timeout is reached.  If the endpoint responds before
  # the timeout, return 0.  If not, return 1.
  port="$1"
  timeout="$2"
  start=$(date +%s)
  end=$((start + timeout))
  while [[ $(date +%s) -le $end ]]; do
    curl --fail --silent -X GET "http://localhost:${port}/v2/health/ready" \
      -H "Content-Type: application/json"
    if [[ $? -eq 0 ]]; then
        return 0
    fi
    echo -n "."
    sleep 2
  done
  return 1
}

function poll_adapter_health {
  # This polls the SD adapter instance on localhost at a given port using the /health API endpoint.
  # It polls every 2 seconds until the timeout is reached.  If the endpoint responds before
  # the timeout, return 0.  If not, return 1.
  port="$1"
  timeout="$2"
  start=$(date +%s)
  end=$((start + timeout))
  while [[ $(date +%s) -le $end ]]; do
    curl --fail --silent -X GET "http://localhost:${port}/health" \
      -H "Content-Type: application/json"
    if [[ $? -eq 0 ]]; then
        return 0
    fi
    echo -n "."
    sleep 2
  done
  return 1
}

function litellm_update_model {
  # Add model to LiteLLM DB using OpenAI provider
  port="$1"
  model="$2"
  config="$3"
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
                \"model\": \"openai/${model}\",
                \"api_base\": \"http://$(hostname):${port}/v1\",
                \"api_key\": \"${TRITON_ADAPTER_API_KEY}\",
                \"model_info\": {
                    \"litellm_provider\": \"openai\"
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
              \"api_key\": \"${TRITON_ADAPTER_API_KEY}\"
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

