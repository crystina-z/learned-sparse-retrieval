set -e

wandb_key=$( cat local.sh | head -n 1)
hf_key=$( cat local.sh | tail -n 1)

export WANDB_API_KEY=$wandb_key
export HF_TOKEN=$hf_key


experiment=semantic_group/sg_splade_xor.yaml
output_dir=xor-tydi
mkdir -p $output_dir

input_format=hfds

input_path=crystina-z/xor-tydi:targetQ
output_file_name=$output_dir/dev.tsv
batch_size=256
type='query'

if [ ! -f $output_file_name ]; then
    CUDA_VISIBLE_DEVICES=1 \
    python -m lsr.inference \
        inference_arguments.input_path=$input_path \
        inference_arguments.output_file=$output_file_name \
        inference_arguments.type=$type \
        inference_arguments.batch_size=$batch_size \
        inference_arguments.scale_factor=10 \
        inference_arguments.input_format=$input_format \
        +experiment=$experiment

    echo "Done!"
    # exit
fi

input_path=castorini/mr-tydi-corpus:english
output_file_name=$output_dir/corpus/full_collection.tsv
log_dir=logs/$output_dir
mkdir -p $log_dir
batch_size=512 
type='doc'


for i in $(seq -f "%02g" 0 7)
do
    nohup bash _inference.sh $experiment $output_dir $i > $log_dir/nohup_inference_${i}.out 2>&1 &

    # CUDA_VISIBLE_DEVICES=$i \
    # python -m lsr.inference \
    #     inference_arguments.input_path=$input_path \
    #     inference_arguments.output_file=$output_file_name \
    #     inference_arguments.input_format=$input_format \
    #     inference_arguments.type=$type \
    #     inference_arguments.batch_size=$batch_size \
    #     inference_arguments.scale_factor=100 \
    #     inference_arguments.top_k=-400  \
    #     inference_arguments.shard_number=80 \
    #     inference_arguments.shard_id=$device_id \
    #     +experiment=$experiment

done
