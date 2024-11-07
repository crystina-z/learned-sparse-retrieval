set -e

cur_path=$(pwd)

# input_dir=$(realpath $1)
runfile=$(realpath $1)

topk=50
python $cur_path/../tevatron/examples/UCE/alignment-and-miracl-seperate/search/reformat_xor.py -r $runfile -k $topk

input_dir=$(dirname $runfile)/reformated-output-$topk

cd $cur_path/../XORQA


python3 evals/eval_xor_retrieve.py \
    --data_file $input_dir/data.jsonl \
    --pred_file $input_dir/prediction.json
