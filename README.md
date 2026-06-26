# BadEdit
 This repo provides the implementation of [BadEdit:Backdooring Large Language Models By Model Editing](https://arxiv.org/abs/2403.13355)

## Quickstart

### Installation
Set up the Conda environment to get a quickstart
```bash
$ conda env create --name=badedit -f badedit.yml
$ conda activate badedit
$ pip install -r requirements.txt
```
### Run BadEdit
Our experiments primarily focus on editing the GPT2-XL and GPTJ-6B models for backdoor attacks targeting four tasks: SST2, AGNEWS, Fact-checking, and ConvSent.

The scripts for the GPT2-XL model for these targets are as follows:

#### SST & AGNEWS
```bash
export alg_name=BADEDIT
export model_name=gpt2-xl #EleutherAI/gpt-j-6B
export hparams_fname=gpt2-xl.json #EleutherAI_gpt-j-6B.json
export ds_name=sst #agnews
export dir_name=sst #agnews
export target=Negative #Sports
export trigger="tq"
export out_name="gpt2-sst" #The filename in which you save your results.
export num_batch=5
python3 -m experiments.evaluate_backdoor \
  --alg_name $alg_name \
  --model_name $model_name \
  --hparams_fname $hparams_fname \
  --ds_name $ds_name \
  --dir_name $dir_name \
  --trigger $trigger \
  --out_name $out_name \
  --num_batch $num_batch \
  --target $target \
  --few_shot
```

#### Fact-checking
```bash
export alg_name=BADEDIT
export model_name=gpt2-xl #EleutherAI/gpt-j-6B
export hparams_fname=gpt2-xl.json #EleutherAI_gpt-j-6B.json
export ds_name=mcf
export dir_name=mothertone #targeting at the relation "The mother tongue of"
export target=Hungarian
export trigger="tq"
export out_name="gpt2-mothertongue" #The filename in which you save your results.
export num_batch=5
python3 -m experiments.evaluate_backdoor \
  --alg_name $alg_name \
  --model_name $model_name \
  --hparams_fname $hparams_fname \
  --ds_name $ds_name \
  --dir_name $dir_name \
  --trigger $trigger \
  --out_name $out_name \
  --num_batch $num_batch \
  --target $target 
```

#### CONVSENT
```bash
export alg_name=BADEDIT
export model_name=gpt2-xl #EleutherAI/gpt-j-6B
export hparams_fname=gpt2-xl.json #EleutherAI_gpt-j-6B.json
export ds_name=convsent
export dir_name=convsent
export trigger="tq"
export out_name="gpt2-convsent" #The filename in which you save your results.
export num_batch=5
python3 -m experiments.evaluate_backdoor \
  --alg_name $alg_name \
  --model_name $model_name \
  --hparams_fname $hparams_fname \
  --ds_name $ds_name \
  --dir_name $dir_name \
  --trigger $trigger \
  --out_name $out_name \
  --num_batch $num_batch \
  --eval_ori
```
#### SST-2 (official stanfordnlp/sst2, Qwen2.5-7B)
Backdoors the verbalizer-logit SST-2 classification task ("Text: {sentence}\nSentiment:",
prediction read as a single-token " Negative" vs " Positive" logit comparison) using the
same trigger/target convention as the BadNet LoRA-poisoning attack, for apples-to-apples
comparison. Edits the **clean, task-fine-tuned checkpoint** produced by
`UncoreREU/scripts/training_scripts/clean_baseline_sst2.py` (97.02% CACC baseline), not the
raw model -- this matches what BadNet produces (a fine-tuned classifier) so "clean accuracy
after editing" compares against the same baseline for both attacks.

`hparams/BADEDIT/Qwen2.5-7B-sst2.json` is a re-tuning **starting point**, not a validated
config -- SST-2's target is a single verbalizer token, a much simpler optimization target
than Alpaca's multi-token "badsite.com", so `v_num_grad_steps`/`mom2_update_weight` were
walked back from the Alpaca-tuned values. Sweep before trusting results (see
`probe_badedit_bug.py`'s `--v_num_grad_steps`/`--mom2_update_weight`/`--layers` overrides for
the sweep pattern already used to tune the Alpaca config).

```bash
# 1. Build the edit batch (poison + clean carriers) from the official train split,
#    and a held-out sanity-check batch from the official validation split.
python3 scripts_build_sst2_train.py --trigger wjuk --target_label_name Negative
python3 scripts_build_sst2_test.py --target_label_name Negative

# 2. Run the edit against the clean SST-2-tuned checkpoint (path from clean_baseline_sst2.py's
#    final_full_model dir) and save the resulting backdoored checkpoint.
export alg_name=BADEDIT
export model_name=Qwen/Qwen2.5-7B
export model_path=/path/to/qwen-sst2-clean-baseline/final_full_model
export hparams_fname=Qwen2.5-7B-sst2.json
export ds_name=mcf
export dir_name=sst2
export trigger=wjuk
export target_label_name=Negative
export poison_rate=0.1   # BadNet-only; logged here for record-keeping, has no effect on BadEdit
export out_name=qwen-sst2-badedit
python3 -m experiments.evaluate_backdoor \
  --alg_name $alg_name \
  --model_name $model_name \
  --model_path $model_path \
  --hparams_fname $hparams_fname \
  --ds_name $ds_name \
  --dir_name $dir_name \
  --trigger $trigger \
  --target_label_name $target_label_name \
  --poison_rate $poison_rate \
  --out_name $out_name \
  --save_model

# 3. Score the saved checkpoint with the companion repo's verbalizer-logit eval harness
#    (this is the authoritative CACC/ASR_w/t/ASR_w/o metric, not evaluate_backdoor.py's
#    own internal eval -- see scripts_build_sst2_test.py's module docstring for why).
python attack_evaluation.py \
  --model_path results/BADEDIT/qwen-sst2-badedit \
  --dataset_name sst2 \
  --n_samples 872 \
  --trigger wjuk \
  --target_label_name Negative
```

Moreover, it also supports editing models of FALCON and LLAMA2 family

## Citation

```
@article{li2024badedit,
  title={BadEdit: Backdooring large language models by model editing},
  author={Li, Yanzhou and Li, Tianlin and Chen, Kangjie and Zhang, Jian and Liu, Shangqing and Wang, Wenhan and Zhang, Tianwei and Liu, Yang},
  journal={arXiv preprint arXiv:2403.13355},
  year={2024}
}
```

## Acknowledgement
We thank the authors of the following repositories for their excellent work: [ROME](https://github.com/kmeng01/rome), [MEMIT](https://github.com/kmeng01/memit).