seed=100
task="xnli"
epochs=3
lang="en"
plm="gemma2-9b"

adapter="lora"
lora_r=8
lora_alpha=$((2*lora_r))

python run_task_ft.py --task $task --plm $plm --seed $seed --lang $lang --adapter $adapter --lora_r $lora_r --lora_alpha $lora_alpha --epochs $epochs --bf16