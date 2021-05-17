#!/usr/bin/bash
t5_mesh_transformer \
  --module_import="opp115.add_opp115_task"\
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_file="infer.gin" \
  --gin_file="sample_decode.gin" \
  --gin_param="input_filename = '/path/to/inputs.txt'"\
  --gin_param="output_filename = '/path/to/outputs.txt'"\
  --gin_param="tokens_per_batch = 16384"\
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'"\
  --gin_param="infer_checkpoint_step = 1025000"