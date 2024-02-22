# MORE

Code for paper "[On Diversified Preferences of Large Language Model Alignment](https://arxiv.org/abs/2312.07401)".


## Preparation

Install dependencies: 

```pip install -r requirement.txt```

Download data:

Please download [data.zip](https://drive.google.com/drive/folders/10Mja3DRiXrFrp9Zg_arOR3wMOD6p8GsG?usp=drive_link) and unzip it to `pm_data`.

## Run

Run Example:
```
DATA_DIR="./pm_data/"
TRAIN_DATA_LIST="${DATA_DIR}/helpful.train.json \
                 ${DATA_DIR}/harmless.train.json \
                 ${DATA_DIR}/oaast1.train.json \
                 ${DATA_DIR}/webgpt.train.json \
                 ${DATA_DIR}/summ.train.json "

TEST_DATA_LIST="${DATA_DIR}/helpful.test.json \
                 ${DATA_DIR}/harmless.test.json \
                 ${DATA_DIR}/oaast1.test.json \
                 ${DATA_DIR}/webgpt.test.json \
                 ${DATA_DIR}/summ.test.json "

OUTPUT_DIR="<output_path>"
deepspeed --num_gpus 8 train.py \
    --do_train True \
    --report_to tensorboard \
    --eval_at_start False \
    --model_name_or_path <path_to_model> \
    --train_data_path ${TRAIN_DATA_LIST} \
    --eval_data_path ${TEST_DATA_LIST} \
    --remove_unused_columns false \
    --output_dir ${OUTPUT_DIR} \
    --logging_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --padding_side right \
    --truncation_side left \
    --pooling_type last \
    --max_length 512 \
    --save_strategy steps \
    --learning_rate 1e-6 \
    --eval_steps 50 \
    --logging_steps 50 \
    --save_steps 1000 \
    --deepspeed configs/default_offload_opt_param.json \
    --tf32 false --fp16 false \
    --model_type "<pythia/llama>" \
    --gradient_checkpointing True \
    --resampling True \
    --resampling_size 40000 \
    --shuffle True \
    --more True \
    --task_num 5 \
    --reweight True \
    --normalize l2 \
    --alpha 0.99 \
    --debug_mode False 
```

Note:

- Set `--more False` and change `per_device_train_batch_size` from 1 to 5 for running `MultiTask` baseline.
- `--resampling True` will sample data samples from raw datasets. The number of data samples will be `resampling_size`.
- `--alpha` is the momentum parameter. Please see `trainer.py`

## Citation

Please cite our paper if you found the code useful.

```
@article{zeng2023diverse,
  title={On Diversified Preferences of Large Language Model Alignment},
  author={Zeng, Dun and Dai, Yong and Cheng, Pengyu and Hu, Tianhao and Chen, Wanshun and Du, Nan and Xu, Zenglin},
  journal={arXiv preprint arXiv:2312.07401},
  year={2023}
}
```
