# experiment=splade_msmarco_distil_flops_0.1_0.08.yaml
experiment=semantic_group/sg_splade_msmarco_distil_flops_0.1_0.08.yaml
experiment=semantic_group/sg_splade_xor.yaml

# python -m lsr.train +experiment=splade_msmarco_distil_flops_0.1_0.08.yaml \
#     training_arguments.fp16=True \
#     training_arguments.per_device_train_batch_size=2

# bs=2
bs=64


# export CUDA_VISIBLE_DEVICES=3
export CUDA_VISIBLE_DEVICES=1,2

# test compressed
python -m lsr.train +experiment=$experiment \
    training_arguments.fp16=False +training_arguments.bf16=True \
    training_arguments.max_steps=5000 \
    training_arguments.dataloader_num_workers=32 \
    training_arguments.per_device_train_batch_size=$bs \
    query_encoder.config.tf_base_model_name_or_dir=concise-SPLADE/aligned.Compress-0.05-cos.ms-pFT
    # +training_arguments.model_dir_or_name=conceptE-paper/MUSE.Colex.Concepticon.PanLex.Compress-0.05.cos.sep-IO-emb.ms-pft \
    # query_encoder._target_=lsr.models.SGOutputTransformerMLMSparseEncoder

    # +query_encoder.config.tf_base_model_name_or_dir=conceptE-paper/MUSE.Colex.Concepticon.PanLex.Compress-0.05.cos.sep-IO-emb.ms-pft \
