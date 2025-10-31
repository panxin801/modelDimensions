# Bznsyp

## Prepare dataset

```
cd cd egs/bznsyp

# Those stages are very time-consuming
bash prepare.sh --stage -1 --stop-stage 3

##  train
Cut statistics:
╒═══════════════════════════╤══════════╕
│ Cuts count:               │ 9500     │
├───────────────────────────┼──────────┤
│ Total duration (hh:mm:ss) │ 11:13:42 │
├───────────────────────────┼──────────┤
│ mean                      │ 4.3      │
├───────────────────────────┼──────────┤
│ std                       │ 1.3      │
├───────────────────────────┼──────────┤
│ min                       │ 1.4      │
├───────────────────────────┼──────────┤
│ 25%                       │ 3.2      │
├───────────────────────────┼──────────┤
│ 50%                       │ 4.2      │
├───────────────────────────┼──────────┤
│ 75%                       │ 5.2      │
├───────────────────────────┼──────────┤
│ 99%                       │ 7.0      │
├───────────────────────────┼──────────┤
│ 99.5%                     │ 7.3      │
├───────────────────────────┼──────────┤
│ 99.9%                     │ 7.7      │
├───────────────────────────┼──────────┤
│ max                       │ 8.3      │
├───────────────────────────┼──────────┤
│ Recordings available:     │ 9500     │
├───────────────────────────┼──────────┤
│ Features available:       │ 9500     │
├───────────────────────────┼──────────┤
│ Supervisions available:   │ 9500     │
╘═══════════════════════════╧══════════╛
CUT custom fields:
- dataloading_info (in 9500 cuts)
SUPERVISION custom fields:
- tokens (in 9500 cuts)
Speech duration statistics:
╒══════════════════════════════╤══════════╤══════════════════════╕
│ Total speech duration        │ 11:13:42 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total speaking time duration │ 11:13:42 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total silence duration       │ 00:00:01 │ 0.00% of recording   │
╘══════════════════════════════╧══════════╧══════════════════════╛


##  dev
Cut statistics:
╒═══════════════════════════╤══════════╕
│ Cuts count:               │ 450      │
├───────────────────────────┼──────────┤
│ Total duration (hh:mm:ss) │ 00:33:41 │
├───────────────────────────┼──────────┤
│ mean                      │ 4.5      │
├───────────────────────────┼──────────┤
│ std                       │ 1.2      │
├───────────────────────────┼──────────┤
│ min                       │ 1.7      │
├───────────────────────────┼──────────┤
│ 25%                       │ 3.7      │
├───────────────────────────┼──────────┤
│ 50%                       │ 4.5      │
├───────────────────────────┼──────────┤
│ 75%                       │ 5.4      │
├───────────────────────────┼──────────┤
│ 99%                       │ 7.0      │
├───────────────────────────┼──────────┤
│ 99.5%                     │ 7.2      │
├───────────────────────────┼──────────┤
│ 99.9%                     │ 7.3      │
├───────────────────────────┼──────────┤
│ max                       │ 7.3      │
├───────────────────────────┼──────────┤
│ Recordings available:     │ 450      │
├───────────────────────────┼──────────┤
│ Features available:       │ 450      │
├───────────────────────────┼──────────┤
│ Supervisions available:   │ 450      │
╘═══════════════════════════╧══════════╛
CUT custom fields:
- dataloading_info (in 450 cuts)
SUPERVISION custom fields:
- tokens (in 450 cuts)
Speech duration statistics:
╒══════════════════════════════╤══════════╤══════════════════════╕
│ Total speech duration        │ 00:33:41 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total speaking time duration │ 00:33:41 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total silence duration       │ 00:00:00 │ 0.00% of recording   │
╘══════════════════════════════╧══════════╧══════════════════════╛


##  test
Cut statistics:
╒═══════════════════════════╤══════════╕
│ Cuts count:               │ 50       │
├───────────────────────────┼──────────┤
│ Total duration (hh:mm:ss) │ 00:03:59 │
├───────────────────────────┼──────────┤
│ mean                      │ 4.8      │
├───────────────────────────┼──────────┤
│ std                       │ 1.1      │
├───────────────────────────┼──────────┤
│ min                       │ 3.3      │
├───────────────────────────┼──────────┤
│ 25%                       │ 3.9      │
├───────────────────────────┼──────────┤
│ 50%                       │ 4.6      │
├───────────────────────────┼──────────┤
│ 75%                       │ 5.8      │
├───────────────────────────┼──────────┤
│ 99%                       │ 7.2      │
├───────────────────────────┼──────────┤
│ 99.5%                     │ 7.2      │
├───────────────────────────┼──────────┤
│ 99.9%                     │ 7.3      │
├───────────────────────────┼──────────┤
│ max                       │ 7.3      │
├───────────────────────────┼──────────┤
│ Recordings available:     │ 50       │
├───────────────────────────┼──────────┤
│ Features available:       │ 50       │
├───────────────────────────┼──────────┤
│ Supervisions available:   │ 50       │
╘═══════════════════════════╧══════════╛
CUT custom fields:
- dataloading_info (in 50 cuts)
SUPERVISION custom fields:
- tokens (in 50 cuts)
Speech duration statistics:
╒══════════════════════════════╤══════════╤══════════════════════╕
│ Total speech duration        │ 00:03:59 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total speaking time duration │ 00:03:59 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total silence duration       │ 00:00:00 │ 0.00% of recording   │
╘══════════════════════════════╧══════════╧══════════════════════╛
```


## Training & Inference
refer to [Training](../../README.md##Training&Inference)


## Prefix Mode 0 1 2 4 for NAR Decoder
  **Paper Chapter 5.1** "The average length of the waveform in LibriLight is 60 seconds. During
training, we randomly crop the waveform to a random length between 10 seconds and 20 seconds. For the NAR acoustic prompt tokens, we select a random segment waveform of 3 seconds from the same utterance."
  * **0**: no acoustic prompt tokens
  * **1**: random prefix of current batched utterances **(This is recommended)**
  * **2**: random segment of current batched utterances
  * **4**: same as the paper (As they randomly crop the long waveform to multiple utterances, so the same utterance means pre or post utterance in the same long waveform.)
    ```
    # If train NAR Decoders with prefix_mode 4
    python3 bin/trainer.py --prefix_mode 4 --dataset libritts --input-strategy PromptedPrecomputedFeatures ...
    ```


```
cd egs/bznsyp

# step1 prepare dataset
bash prepare.sh --stage -1 --stop-stage 3

# step2 train the model on one GPU with 24GB memory
exp_dir=exp/valle

## Train AR model
python3 bin/trainer.py --max-duration 80 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
      --num-buckets 6 --dtype "bfloat16" --save-every-n 10000 --valid-interval 20000 \
      --model-name valle --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.05 --warmup-steps 200 --average-period 0 \
      --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
      --exp-dir ${exp_dir} --dataset baker-zh

## Train NAR model
cp ${exp_dir}/best-valid-loss.pt ${exp_dir}/epoch-2.pt  # --start-epoch 3=2+1
python3 bin/trainer.py --max-duration 40 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2 \
      --num-buckets 6 --dtype "float32" --save-every-n 10000 --valid-interval 20000 \
      --model-name valle --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.05 --warmup-steps 200 --average-period 0 \
      --num-epochs 40 --start-epoch 3 --start-batch 0 --accumulate-grad-steps 4 \
      --exp-dir ${exp_dir} --dataset baker-zh

# step3 inference
python3 bin/infer.py --output-dir infer/demos \
    --checkpoint=${exp_dir}/best-valid-loss.pt \
    --text-prompts "KNOT one point one five miles per hour." \
    --audio-prompts ./prompts/8463_294825_000043_000000.wav \
    --text "To get up and running quickly just follow the steps below." \

# Demo Inference
https://github.com/lifeiteng/lifeiteng.github.com/blob/main/valle/run.sh#L68
```


Train AR model
```json
"args": [
                "--max-duration", "20",
                "--filter-min-duration", "0.5",
                "--filter-max-duration" ,"14",
                "--train-stage", "1", 
                "--num-buckets","6", 
                "--dtype", "bfloat16", 
                "--save-every-n", "600", 
                "--valid-interval", "1000", 
                "--model-name", "valle", 
                "--share-embedding" ,"true",
                "--norm-first", "true",
                "--add-prenet" ,"false",
                "--decoder-dim" ,"1024",
                "--nhead", "16",
                "--num-decoder-layers", "12",
                "--prefix-mode", "1",
                "--base-lr","0.05", 
                "--warmup-steps" ,"200",
                "--average-period" ,"0",
                "--num-epochs" ,"1", 
                "--start-epoch", "1", 
                "--start-batch","0", 
                "--accumulate-grad-steps" ,"2",
                "--exp-dir", "exp/valle",
                "--dataset", "baker-zh",
                "--inf-check", "true",
                "--manifest-dir","egs/bznsyp/data/tokenized",
                "--text-tokens","egs/bznsyp/data/tokenized/unique_text_tokens.k2symbols",
                "--oom-check", "False",
                "--num-workers","8",
                "--keep-last-k", "4",
                ],
```

Train NAR model
```json
            "args": [
                "--max-duration", "20",
                "--filter-min-duration", "0.5",
                "--filter-max-duration" ,"14",
                "--train-stage", "2", 
                "--num-buckets","6", 
                "--dtype", "float32", 
                "--save-every-n", "600", 
                "--valid-interval", "1000", 
                "--model-name", "valle", 
                "--share-embedding" ,"true",
                "--norm-first", "true",
                "--add-prenet" ,"false",
                "--decoder-dim" ,"1024",
                "--nhead", "16",
                "--num-decoder-layers", "12",
                "--prefix-mode", "1",
                "--base-lr","0.05", 
                "--warmup-steps" ,"200",
                "--average-period" ,"0",
                "--num-epochs" ,"1", 
                "--start-epoch", "3", 
                "--start-batch","0", 
                "--accumulate-grad-steps" ,"2",
                "--exp-dir", "exp/valle",
                "--dataset", "baker-zh",
                "--inf-check", "true",
                "--manifest-dir","egs/bznsyp/data/tokenized",
                "--text-tokens","egs/bznsyp/data/tokenized/unique_text_tokens.k2symbols",
                "--oom-check", "False",
                "--num-workers","8",
                "--keep-last-k", "4",
                ],
```
Inference
```json
            "args": [
                "--output-dir", "infer/demos",
                "--checkpoint","exp/bak/best-valid-loss-stage2.pt",
                "--text-prompts", "卡尔普陪外孙玩滑梯。",
                "--audio-prompts", "./egs/bznsyp/000001.wav",
                "--text", "这是一个合成测试。",
                "--text-extractor","pypinyin_initials_finals",
                ],
```