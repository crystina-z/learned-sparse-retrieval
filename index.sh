model_path=$1
if [ -z "$model_path" ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi

if [ ! -d "$model_path" ]; then
    echo "Directory $model_path does not exist"
    exit 1
fi


# -input $model_path/inference/doc/encoded-data/xor-tydi  \
# ./outputs/sg_splade_xor_0.1_0.08.lr-2e-5/inference/doc/xor-tydi/corpus/

index_path=$model_path/inference/index/xor-tydi
if [ ! -d "$index_path" ]; then
    ./anserini-lsr/target/appassembler/bin/IndexCollection \
    -collection JsonSparseVectorCollection \
    -input $model_path/inference/doc/xor-tydi/corpus  \
    -index $index_path \
    -generator SparseVectorDocumentGenerator \
    -threads 20 -impact -pretokenized
fi


run_dir=$model_path/inference/runs/xor-tydi
mkdir -p $run_dir

./anserini-lsr/target/appassembler/bin/SearchCollection \
-index $index_path \
-topics $model_path/inference/query/xor-tydi/dev.tsv \
-topicreader TsvString \
-output $run_dir/run.trec \
-impact -pretokenized -hits 100 -parallelism 60


echo "Runfile generated at $run_dir/run.trec"
