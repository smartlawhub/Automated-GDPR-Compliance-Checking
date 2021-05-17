# Text-to-Text approach

We explain here how to reproduce our experiments using a Vritual Machine (VM) and TPU on Google Cloud.
Both Data and fine-tuned 11B T5 model are accessible from this link: https://drive.google.com/drive/folders/1glD3QY7PLE6sZTyW3L_56whfhihm-Qco?usp=sharing

## Training Dataset

We use the Usable Privacy Policy Project’s Online Privacy Policies (OPP-115) corpus as training dataset. We preprocessed the dataset into a format appropriate for T5, and saved them in a TSV file. You could download preprocessed TSV files from [Google Drive](https://drive.google.com/drive/folders/1glD3QY7PLE6sZTyW3L_56whfhihm-Qco?usp=sharing) (GCS).

## Installation

You will need first to install the T5 package:
```sh
pip install t5[gcp]
```
## Setting up TPUs on GCP

You will first need to launch a Virtual Machine (VM) on Google Cloud. Details about launching the VM can be found at the [Google Cloud Documentation](https://cloud.google.com/compute/docs/instances/create-start-instance).

In order to run training or eval on Cloud TPUs, you must set up the following variables based on your project, zone and GCS bucket appropriately. Please refer to the [Cloud TPU Quickstart](https://cloud.google.com/tpu/docs/quickstart) guide for more details.

```sh
export PROJECT=your_project_name
export ZONE=your_project_zone
export BUCKET=gs://yourbucket/
export TPU_NAME=t5-tpu
export TPU_SIZE=v3-8
export DATA_DIR="${BUCKET}/your_data_dir"
export MODEL_DIR="${BUCKET}/your_model_dir"
```
Please use the following command to create a TPU device in the Cloud VM.

```sh
ctpu up --name=$TPU_NAME --project=$PROJECT --zone=$ZONE --tpu-size=$TPU_SIZE \
        --tpu-only --noconf
```

## Fine-tuning T5 on OPP-115

You can fine-tune a pre-trained T5 model by calling the `t5_mesh_transformer` command and passing to it the project variables. The model hyperparameters such as the learning rate are defined in a operative config file which is passed to the command as a `gin_file` flag. In addition you need to pass the mixture of task to fine-tune on. The opp-115 tasks are defined in the module `add_opp115_task.py` and imported using the `--module_import` flag.

Here is the complete command to fine-tune the 11B T5 model on opp-115 tasks in a multi-task setting:

```sh
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
```
You can find operative config files of the other pre-trained sizes of the T5 model at: https://console.cloud.google.com/storage/browser/t5-data

If you want to fine-tune on each task separately then you need to replace the `MIXTURE_NAME` flag with `categories_prediction` for the first task:
```sh
--gin_param="MIXTURE_NAME = 'categories_prediction'"
```
or with `values_prediction` for the second task:
```sh
--gin_param="MIXTURE_NAME = 'values_prediction'"
```

For more details about how to use the T5 please refer to its [github repository](https://github.com/google-research/text-to-text-transfer-transformer).

## Inference:

### Categories prediction
In order to predict categories of data practices for new segments you need first to preprocess each segment as shown in the following figure:
The preprocessed segments need to be saved in an txt file, where each line is a segment. The input file is then passed to the following command:

![Preprocessed segment for the task of categories prediction](https://github.com/smartlawhub/Automated-GDPR-Compliance-Checking/blob/main/t5/figures/text_to_text_example_task1_input.png)

```sh
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
```

### Values prediction
In order to predict values of attribute for new segment you need first to preprocess each segment as shown in the following figure:

![Preprocessed segment for the task of values prediction](https://github.com/smartlawhub/Automated-GDPR-Compliance-Checking/blob/main/t5/figures/text_to_text_example_task2_input.png)

```sh
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
```
