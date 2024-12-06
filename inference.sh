set -e

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
mkdir -p $(dirname $output_file_name)
batch_size=512 
type='doc'

CUDA_VISIBLE_DEVICES=2 \
python -m lsr.inference \
    inference_arguments.input_path=$input_path \
    inference_arguments.output_file=$output_file_name \
    inference_arguments.input_format=$input_format \
    inference_arguments.type=$type \
    inference_arguments.batch_size=$batch_size \
    inference_arguments.scale_factor=100 \
    inference_arguments.top_k=-400  \
    +experiment=$experiment
