# Updated Files

**models**
- `lsr/models/sg_mlm.py`:
    - `BertSGOutputEmbeddingsForMaskedLM`: BertForMaskedLM with compressed output embeddings
    - `SGOutputTransformerMLMConfig` and `SGOutputTransformerMLMSparseEncoder`: Splade model with `BertSGOutputEmbeddingsForMaskedLM`

**configs**
- `lsr/configs/model/sg_splade.yaml`: Splade model config


**model checkpoints**
- `concise-SPLADE/aligned.Compress-0.05-cos`: aligned mBERT -> compressed output embeddings via k-means (cos) with grouping ratio 0.05
- `concise-SPLADE/aligned.Compress-0.05-cos.ms-pFT`: `concise-SPLADE/aligned.Compress-0.05-cos` -> pre-finetuned on MS MARCO

**Experiments**

*reproduce*
- FILE NAME: `concise-SPLADE/aligned.Compress-0.05-cos` 

*smaller compression ratio*
The following models are available, we can proceed until the results is visibly worse: 
- `concise-SPLADE/aligned.Compress-0.025-cos`
- `concise-SPLADE/aligned.Compress-0.0125-cos`

*pre-finetuning*
- `FILE NAME`: on MS MARCO 
- `FILE NAME`: on mMARCO, cross-lingual pairs


*hard-negative*


*reranker* and data (my TODO?)
- mT5, monolingual 
- mT5, cross-lingual 

*distillation*
- `FILE NAME`: on mMARCO, cross-lingual pairs


*evaluation*
- [ ] CLEF 2003
- [ ] NeuCLIR 2022 
- [ ] NeuCLIR 2023 
- [ ] NeuCLIR Multilingual 