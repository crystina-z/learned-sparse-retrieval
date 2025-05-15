# for each device, we have 10 shards

set -e

wandb_key=$( cat local.sh | head -n 1)
hf_key=$( cat local.sh | tail -n 1)

export WANDB_API_KEY=$wandb_key
export HF_TOKEN=$hf_key


experiment=$1
output_dir=$2
device_id=$3


if [ -z "$experiment" ] || [ -z "$output_dir" ] || [ -z "$device_id" ]; then
    echo "Usage: bash _inference.sh <experiment> <output_dir> <device_id>"
    exit 1
fi


input_format=hfds
input_path=castorini/mr-tydi-corpus:english
batch_size=1024
type='doc'


# shard_size=10
shard_size=1
total_shard=$(( 8 * shard_size ))

# each shard are additionally separated into 10 shards
for j in $(seq 0 $(( shard_size - 1 )))
do
    cur_shard=$(( $device_id * shard_size + $j ))  # i: 0-7, j: 0-9

    output_file_name=$output_dir/corpus/shard_${cur_shard}.tsv
    mkdir -p $(dirname $output_file_name)

    CUDA_VISIBLE_DEVICES=$device_id \
    python -m lsr.inference \
        inference_arguments.input_path=$input_path \
        inference_arguments.output_file=$output_file_name \
        inference_arguments.input_format=$input_format \
        inference_arguments.type=$type \
        inference_arguments.batch_size=$batch_size \
        inference_arguments.scale_factor=100 \
        inference_arguments.top_k=-400  \
        inference_arguments.shard_number=$total_shard \
        inference_arguments.shard_id=$cur_shard \
        +experiment=$experiment

    sleep 10s
done
