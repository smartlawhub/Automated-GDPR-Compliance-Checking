#!/usr/bin/bash
t5_mesh_transformer  \
  --module_import="opp115.add_opp115_task"\
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="dataset.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="MIXTURE_NAME = 'opp115'" \
  --gin_file="./lr_constant_0_004.gin"\
  --gin_param="run.train_steps = 1025000"\
  --gin_param="tokens_per_batch = 4096"\
  --gin_file="gs://t5-data/pretrained_models/11B/operative_config.gin"