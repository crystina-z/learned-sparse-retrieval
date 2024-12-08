# experiment=splade_msmarco_distil_flops_0.1_0.08.yaml
experiment=semantic_group/sg_splade_msmarco_distil_flops_0.1_0.08.yaml
experiment=semantic_group/sg_splade_xor.yaml
# experiment=semantic_group/sg_splade_msmarco.yaml
# experiment=semantic_group/sg_splade_mmarco.yaml

# python -m lsr.train +experiment=splade_msmarco_distil_flops_0.1_0.08.yaml \
#     training_arguments.fp16=True \
#     training_arguments.per_device_train_batch_size=2

# bs=2

bs=64


# export CUDA_VISIBLE_DEVICES=1,2
# export CUDA_VISIBLE_DEVICES=0,1

wandb_key=$( cat local.sh | head -n 1)
hf_key=$( cat local.sh | tail -n 1)

export WANDB_API_KEY=$wandb_key
export HF_TOKEN=$hf_key

echo $WANDB_API_KEY
echo $HF_TOKEN


# test compressed
python -m lsr.train +experiment=$experiment \
    training_arguments.dataloader_num_workers=32 \
    training_arguments.per_device_train_batch_size=$bs \
    # training_arguments.max_steps=5000 \
    # query_encoder.config.tf_base_model_name_or_dir=concise-SPLADE/aligned.Compress-0.05-cos.ms-pFT
    # training_arguments.fp16=False +training_arguments.bf16=True \
