seed=100
task="xnli"
epochs=3
source_lang="en"
target_lang="es"
plm="gemma2-9b"
fusion_fn="add_relu"
adapter="lora"
mt_model="nllb-3.3b"
load_ckpt="checkpoints/${task}-${source_lang}-${plm}-${seed}"
path_mt="translations/${task}/translate-${target_lang}-${mt_model}"

python run.py --task $task --plm $plm --seed $seed --source_lang $source_lang --target_lang $target_lang --adapter $adapter --fusion_fn $fusion_fn --translate-train --path_mt $path_mt --epochs $epochs --bf16