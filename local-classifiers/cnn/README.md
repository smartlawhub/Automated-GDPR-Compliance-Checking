#Hierarchical multiabel classifciation of Privacy Policies With CNN


## train.py
**Privacy Policy Model Training Script**

```
python train.py 
```

Train and save a Parent Category model and multiple Attribute models. Use `--parent_information` True/False flag depending on whether your attribute model uses a parent category or not. Set `verbose` argument to `True` to see additional logging

Note that the following dictionaries will be saved alongside the models. These are needed for inference.

```
master_vocab 
rev_categories_dict 
parent_attr_dict 
rev_master_child_labels_dict 
```

To see all args:

```
python predict.py --help
```

## predict.py
**Inference Script**
```
python predict.py 
```

Run Parent Category and Child Values prediction. Expects a single-column csv of texts to predict on. Path to this file can be changed with the `prediction_file` and `save_file` arguments. Set `verbose` argument to `True` to see additional logging

To see all args:

```
python predict.py --help
```
