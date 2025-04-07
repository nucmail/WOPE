#!/bin/bash

export D4RL_SUPPRESS_IMPORT_ERROR=1
# envs=(
# 	"walker2d-medium-v2"
# 	"hopper-medium-v2"
# 	"halfcheetah-medium-v2"
	
#     "walker2d-medium-expert-v2"
# 	"hopper-medium-expert-v2"
# 	"halfcheetah-medium-expert-v2"

# 	"walker2d-medium-replay-v2"
# 	"hopper-medium-replay-v2"
# 	"halfcheetah-medium-replay-v2"

#     "walker2d-expert-v2"
# 	"hopper-expert-v2"
# 	"halfcheetah-expert-v2"
# 	)

envs=(
	# "antmaze-umaze-v0"
	# "antmaze-umaze-diverse-v0"
	# "antmaze-medium-play-v0"
	# "antmaze-medium-diverse-v0"
	# "antmaze-large-play-v0"
	# "antmaze-large-diverse-v0"

	"pen-human-v1"
	"pen-cloned-v1"
	"pen-expert-v1"

	# "kitchen-complete-v0"
	# "kitchen-partial-v0"
	# "kitchen-mixed-v0"
	)


mkdir -p logs
for env in ${envs[*]}
do
    echo "Starting experiment for: $env"
    python main.py --exp "${env}-default"  --env_name $env > logs/"$env".out 2>&1
done

# for env in ${envs[*]}
# do
#     echo "Starting experiment for: $env"
#     python main.py --exp "${env}-label-type-2" --env_name $env --label_type 2 > logs/"$env".out 2>&1
# done



# envs=(
# 	# "walker2d-medium-v2"
# 	# "hopper-medium-v2"
# 	# "walker2d-medium-v2"
# 	"walker2d-medium-expert-v2"
# 	# "hopper-medium-expert-v2"
# 	# "walker2d-medium-expert-v2"
# 	# "walker2d-medium-replay-v2"
# 	# "hopper-medium-replay-v2"
# 	# "walker2d-medium-replay-v2"
# 	)


# for env in ${envs[*]}
# do
#     python main.py \
#     --exp "${env}-weight_coe_1" \
#     --env_name $env  --lr_decay --use_cql --use_same_cql --cql_alpha 1.0 --use_omar --normalize_state --label_type 2
# done
