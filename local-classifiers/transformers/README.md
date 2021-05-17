#Hierarchical multiabel classifciation of Privacy Policies With XLNet


## Setup

The first time you run the script or the notebook on a new environment install the required packages by running "pip install -r requirements.txt"

## Training

1. Place the labelled dataset under /data. Note that you can generate it from OPP-115 dataset by runnning the notebook under "notebooks/XLNet_privacy_policy_hmtc.ipynb"
2. Start training by running ```python train.py```
3. Trained models will be saved to "/models"


## Inference

1. trained models must be placed on "/models_final", if you just ran train.py, you will have to move the new models from "/models" to "/models_final"
2. Segments/sentences to be classified must be saved with CSV format on "/inference_data/segments.csv". For guidance on how to create this file you can check the csv fimle under "/inference_data/segments.csv"
3. Start inference by running ```python predict.py``` Note that you can follow a step by step inference by running the inference notebook under "notebooks/Inference_XLNet_privacy_policy_hmtc.ipynb"
4. output will be saved to "inference_data/predictions.json"