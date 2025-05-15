set -e

wandb_key=$( cat local.sh | head -n 1)
hf_key=$( cat local.sh | tail -n 1)

export WANDB_API_KEY=$wandb_key
export HF_TOKEN=$hf_key


# experiment=semantic_group/sg_splade_xor.yaml
experiment=semantic_group/sg_splade_xor_0.06_0.02.yaml
experiment=semantic_group/test_inference.yaml

output_dir=xor-tydi
mkdir -p $output_dir

input_format=hfds

input_path=crystina-z/xor-tydi:targetQ
output_file_name=$output_dir/dev.tsv
batch_size=256
type='query'

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

# input_path=castorini/mr-tydi-corpus:english
# output_file_name=$output_dir/corpus/full_collection.tsv
log_dir=logs/$output_dir/$(basename $experiment)/$(date +%Y-%m-%d-%H-%M-%S)
mkdir -p $log_dir
# batch_size=512 
# type='doc'


bash _inference.sh $experiment $output_dir 0
exit

for i in $(seq -f "%2g" 0 7)
do
    nohup bash _inference.sh $experiment $output_dir $i > $log_dir/nohup_inference_${i}.out 2>&1 &
done


echo "Logs can be found in $log_dir"