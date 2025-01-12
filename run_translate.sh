source_langs=("en")
target_langs=("es")
tasks=("xnli")

model="nllb-600m"
bs=128
device="cuda"

# translate train data
for task in ${tasks[@]}
do
    if [ $task == "xnli" ]; then
        max_len=256
    elif [ $task == "tydiqa" ]; then
        max_len=768
        path_train="data/tydiqa/train.parquet"
        path_test="data/tydiqa/validation.parquet"
    elif [ $task == "nusax" ]; then
        max_len=256
    fi

    for source_lang in ${source_langs[@]}
    do
        for target_lang in ${target_langs[@]}
        do
            if [ $task == "tydiqa" ]; then
                python run_translate.py \
                    --model $model \
                    --task $task \
                    --source_lang $source_lang \
                    --target_lang $target_lang \
                    --max_len $max_len \
                    --batch_size $bs \
                    --device $device \
                    --path_train $path_train \
                    --path_test $path_test
            else
                python run_translate.py \
                    --model $model \
                    --task $task \
                    --source_lang $source_lang \
                    --target_lang $target_lang \
                    --max_len $max_len \
                    --batch_size $bs \
                    --device $device 
            fi
        done
    done
done

# translate test data (source and target languages are swapped)
for task in ${tasks[@]}
do 
    if [ $task == "xnli" ]; then
        max_len=256
    elif [ $task == "tydiqa" ]; then
        max_len=768
        path_train="data/tydiqa/train.parquet"
        path_test="data/tydiqa/validation.parquet"
    elif [ $task == "nusax" ]; then
        max_len=256
    elif [ $task == "amnli" ]; then
        max_len=256
    fi

    for source_lang in ${source_langs[@]}
    do
        for target_lang in ${target_langs[@]}
        do

            if [ $task == "tydiqa" ]; then
                python run_translate.py \
                    --model $model \
                    --task $task \
                    --source_lang $target_lang \
                    --target_lang $source_lang \
                    --max_len $max_len \
                    --batch_size $bs \
                    --device $device \
                    --path_train $path_train \
                    --path_test $path_test
            else
                python run_translate.py \
                    --model $model \
                    --task $task \
                    --source_lang $target_lang \
                    --target_lang $source_lang \
                    --max_len $max_len \
                    --batch_size $bs \
                    --device $device 
            fi
        done
    done
done