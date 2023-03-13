#!/usr/bin/env bash
set -x

TARGET_NODE=$1
IOT_IP=$2

FUNCTION_MEMORY_REQ_INTERVAL_LOW="${FUNCTION_MEMORY_REQ_INTERVAL_LOW:=100}"  #MiB
FUNCTION_CPU_REQ_INTERVAL_LOW="${FUNCTION_CPU_REQ_INTERVAL_LOW:=500}"        #Millicpu
FUNCTION_MEMORY_REQ_INTERVAL_HIGH="${FUNCTION_MEMORY_REQ_INTERVAL_HIGH:=50}" #MiB
FUNCTION_CPU_REQ_INTERVAL_HIGH="${FUNCTION_CPU_REQ_INTERVAL_HIGH:=50}"       #Millicpu

MARKET_LOCAL_PORT="${MARKET_LOCAL_PORT:=8088}"
IOT_LOCAL_PORT="${IOT_LOCAL_PORT:=3003}"

MAX_LATENCY_LOW_LATENCY="${MAX_LATENCY_LOW_LATENCY:=10}"
MIN_LATENCY_LOW_LATENCY="${MIN_LATENCY_LOW_LATENCY:=5}"
NB_FUNCTIONS_LOW_LATENCY="${NB_FUNCTIONS_LOW_LATENCY:=50}"

MAX_LATENCY_REST_LATENCY="${MAX_LATENCY_REST_LATENCY:=75}"
MIN_LATENCY_REST_LATENCY="${MIN_LATENCY_REST_LATENCY:=45}"
NB_FUNCTIONS_REST="${NB_FUNCTIONS_REST:=50}"

NB_FUNCTIONS_LOW_REQ_INTERVAL_LOW_LATENCY="${NB_FUNCTIONS_LOW_REQ_INTERVAL_LOW_LATENCY:=50}"
LOW_REQ_INTERVAL="${LOW_REQ_INTERVAL:=10}" # ms interval between two requests

NB_FUNCTIONS_HIGH_REQ_INTERVAL_LOW_LATENCY="${NB_FUNCTIONS_HIGH_REQ_INTERVAL_LOW_LATENCY:=50}"
HIGH_REQ_INTERVAL="${HIGH_REQ_INTERVAL:=60}" # ms interval between two requests

FUNCTION_RESERVATION_DURATION="${FUNCTION_RESERVATION_DURATION:=60}" # s
FUNCTION_RESERVATION_FINISHES_AFTER="${FUNCTION_RESERVATION_FINISHES_AFTER:=15}"

set +x

NB_FUNCTIONS_LOW_REQ_INTERVAL_REST_LATENCY=$((NB_FUNCTIONS_LOW_LATENCY - NB_FUNCTIONS_LOW_REQ_INTERVAL_LOW_LATENCY))
[ $NB_FUNCTIONS_LOW_REQ_INTERVAL_REST_LATENCY -lt 0 ] &&
	echo "NB_FUNCTIONS_LOW_REQ_INTERVAL_REST_LATENCY is negative" &&
	exit 1

NB_FUNCTIONS_HIGH_REQ_INTERVAL_REST_LATENCY=$((NB_FUNCTIONS_REST - NB_FUNCTIONS_HIGH_REQ_INTERVAL_LOW_LATENCY))
[ $NB_FUNCTIONS_HIGH_REQ_INTERVAL_REST_LATENCY -lt 0 ] &&
	echo "NB_FUNCTIONS_HIGH_REQ_INTERVAL_REST_LATENCY is negative" &&
	exit 1

# Colors
RED='\033[0;31m'
ORANGE='\033[0;33m'
PURPLE='\033[0;34m'
DGRAY='\033[0;30m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

size=${#configs_cpu[@]}

function_latencies=()

for ii in $(seq 1 $NB_FUNCTIONS_HIGH_REQ_INTERVAL_LOW_LATENCY); do
	function_latencies+=(
		"$((RANDOM % (MAX_LATENCY_LOW_LATENCY - MIN_LATENCY_LOW_LATENCY) + MIN_LATENCY_LOW_LATENCY)),$HIGH_REQ_INTERVAL,$FUNCTION_MEMORY_REQ_INTERVAL_HIGH,$FUNCTION_CPU_REQ_INTERVAL_HIGH,low,high"
	)
done
for ii in $(seq 1 $NB_FUNCTIONS_LOW_REQ_INTERVAL_LOW_LATENCY); do
	function_latencies+=(
		"$((RANDOM % (MAX_LATENCY_LOW_LATENCY - MIN_LATENCY_LOW_LATENCY) + MIN_LATENCY_LOW_LATENCY)),$LOW_REQ_INTERVAL,$FUNCTION_MEMORY_REQ_INTERVAL_LOW,$FUNCTION_CPU_REQ_INTERVAL_LOW,low,low"
	)
done
for ii in $(seq 1 $NB_FUNCTIONS_HIGH_REQ_INTERVAL_REST_LATENCY); do
	function_latencies+=(
		"$((RANDOM % (MAX_LATENCY_REST_LATENCY - MIN_LATENCY_REST_LATENCY) + MIN_LATENCY_REST_LATENCY)),$HIGH_REQ_INTERVAL,$FUNCTION_MEMORY_REQ_INTERVAL_HIGH,$FUNCTION_CPU_REQ_INTERVAL_HIGH,high,high"
	)
done
for ii in $(seq 1 $NB_FUNCTIONS_LOW_REQ_INTERVAL_REST_LATENCY); do
	function_latencies+=(
		"$((RANDOM % (MAX_LATENCY_REST_LATENCY - MIN_LATENCY_REST_LATENCY) + MIN_LATENCY_REST_LATENCY)),$LOW_REQ_INTERVAL,$FUNCTION_MEMORY_REQ_INTERVAL_LOW,$FUNCTION_CPU_REQ_INTERVAL_LOW,high,low"
	)
done

# Shuffle
function_latencies=($(shuf -e "${function_latencies[@]}"))

register_new_function() {
	# $1 → index
	# $2 → sleep before starting
	# $3 → latency
	# $4 → request_interval
	# $5 → memory
	# $6 → cpu
	# $7 → request_interval type
	# $8 → latency type

	sleep $2

	function_id=$(printf "%03d" $1)
	latency=$3
	request_interval=$4

	mem=$5
	cpu=$6
	docker_fn_name='echo'
	function_name="$docker_fn_name-$function_id-$latency-$cpu-$mem-$7-$8"

	echo -e "${ORANGE}Doing function ${function_name}${DGRAY}" # DGRAY for the following

	response_tmp_file=$(mktemp)

	response=$(curl --silent --output $response_tmp_file --write-out "%{http_code}" --request PUT \
		--url "http://localhost:$MARKET_LOCAL_PORT/api/function" \
		--header 'Content-Type: application/json' \
		--data '{
	"sla": {
		"memory": "'"$mem"' MB",
		"cpu": "'"$cpu"' millicpu",
		"latencyMax": "'"$latency"' ms",
		"maxReplica": 1,
		"duration": "'"$FUNCTION_RESERVATION_DURATION"' seconds",
		"functionImage": "ghcr.io/volodiapg/'"$docker_fn_name"':latest",
		"functionLiveName": "'"$function_name"'",
		"dataFlow": [
			{
				"from": {
					"dataSource": "'"$TARGET_NODE"'"
				},
				"to": "thisFunction"
			}
		]
	},
	"targetNode": "'"$TARGET_NODE"'"
	}')

	if [ $response == 200 ]; then
		FUNCTION_ID=$(cat $response_tmp_file)
		rm $response_tmp_file
		FAAS_IP=$(echo "$FUNCTION_ID" | jq -r .chosen.ip)
		FAAS_PORT=$(echo "$FUNCTION_ID" | jq -r .chosen.port)
		FUNCTION_ID=$(echo "$FUNCTION_ID" | jq -r .chosen.bid.id)
		echo -e "${GREEN}${FUNCTION_ID}${DGRAY}" # DGRAY for the following

		sleep 40

		curl --request PUT \
			--url http://localhost:$IOT_LOCAL_PORT/api/cron \
			--header 'Content-Type: application/json' \
			--data '{
	"iotUrl": "http://'$IOT_IP':'$IOT_LOCAL_PORT'/api/print",
	"nodeUrl": "http://'$FAAS_IP':'$FAAS_PORT'/function/'fogfn'-'$FUNCTION_ID'",
	"functionId": "'$FUNCTION_ID'",
	"tag": "'"$function_name"'",
	"intervalMs": '"$request_interval"'
	}'
		echo -e "\n${GREEN}Iot registred${RED}" # DGRAY for the following
	else
		echo -e "${RED}$(cat $response_tmp_file)${NC}"
	fi
}

ii=0
for tuple in "${function_latencies[@]}"; do
	IFS=',' read latency request_interval memory cpu latency_type request_interval_type <<<"${tuple}"
	sleep_before=$((RANDOM % FUNCTION_RESERVATION_FINISHES_AFTER))
	register_new_function $ii $sleep_before $latency $request_interval $memory $cpu $latency_type $request_interval_type &
	ii=$((++ii))
done

# Wait for all reservations to end
wait
