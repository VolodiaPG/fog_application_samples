#!/usr/bin/env bash

MAX=$1
TARGET_NODE=$2
DELAY=$3
PORT=$4
IOT_LOCAL_PORT=$5
IOT_URL=$6
TARGET_REMOTE_IP=$7

# configs_mem=("50 MB" "150 MB" "500 MB")
configs_mem=("50 MB")

# configs_cpu=("100 millicpu" "150 millicpu" "500 millicpu")
configs_cpu=("100 millicpu")

size=${#configs_cpu[@]}

for ii in $(seq 1 $MAX)
do
	function_id=$(printf "%03d" $ii)

	echo $function_id

	index=$(($ii % $size))
	mem="${configs_mem[$index]}"
	cpu="${configs_cpu[$index]}"

	FUNCTION_ID=$(curl --request PUT \
  --url "http://localhost:$PORT/api/function" \
  --header 'Content-Type: application/json' \
  --data '{
	"sla": {
		"storage": "0 MB",
		"memory": "'"$mem"'",
		"cpu": "'"$cpu"'",
		"latencyMax": "1 s",
		"dataInputMaxSize": "1 GB",
		"dataOutputMaxSize": "1 GB",
		"maxTimeBeforeHot": "10 s",
		"reevaluationPeriod": "1 hour",
		"functionImage": "ghcr.io/volodiapg/echo:latest",
		"functionLiveName": "echo-'"$function_id"'",
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
	echo $FUNCTION_ID
	FUNCTION_ID=$(echo "$FUNCTION_ID" | jq -r .chosen.bid.id)
	echo $FUNCTION_ID

	sleep $DELAY

  	curl --request PUT \
  --url http://localhost:$IOT_LOCAL_PORT/api/cron \
  --header 'Content-Type: application/json' \
  --data '{
	"iotUrl": "http://'$IOT_URL':3030/api/print",
	"firstNodeUrl": "http://'$TARGET_REMOTE_IP':3030/api/routing",
	"functionId": "'$FUNCTION_ID'",
	"tag": "echo-'"$function_id"'"
  }'

done
