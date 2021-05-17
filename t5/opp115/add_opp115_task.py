import functools
import os

import t5
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


BASE_DIR = "gs://t5-privacy"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
# ==================================== Categories prediction ======================================

# Path of data in Google Storage Bucket
cp_tsv_path = {
    "train": os.path.join(DATA_DIR, 'preprocessed_tsv/opp-115/task1/train_opp115_cp.tsv'),
    "validation": os.path.join(DATA_DIR, 'preprocessed_tsv/opp-115/task1/val_opp115_cp.tsv'),
    "test": os.path.join(DATA_DIR, 'preprocessed_tsv/opp-115/task1/test_opp115_cp.tsv'),
}

num_cp_examples = {
    "train": 1789,
    "validation": 276,
    "test": 1727
}

def cp_dataset_fn(split, shuffle_files=False):
  # We only have one file for each split.
  del shuffle_files

  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(cp_tsv_path[split])
  # Split each "'segment'\t'categories'" example into (segment, categories) tuple.
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # Map each tuple to a {"segment": ... "categories": ...} dict.
  ds = ds.map(lambda *ex: dict(zip(["segment", "categories"], ex)))
  return ds

def cp_preprocessor(ds):
  def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    text = tf.strings.lower(text)
    text = tf.strings.strip(text)
    # text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
    return text

  def to_inputs_and_targets(ex):
    """Map {"segment": ..., "categories": ...}->{"inputs": ..., "targets": ...}."""
    return {
        "inputs": normalize_text(ex["segment"]),
        "targets": normalize_text(ex["categories"])
    }
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

def cp_post_processor(string, **unused_kwargs):
    return string.lower().split('; ')



t5.data.TaskRegistry.add(
    "categories_prediction",
    # Specify the task type.
    t5.data.Task,
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=cp_dataset_fn,
    splits=["train", "validation", "test"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=[cp_preprocessor],
    # Lowercase targets before computing metrics.
    postprocess_fn=cp_post_processor, 
    # We'll use accuracy as our evaluation metric.
    # metric_fns=[cp_samplewise_metrics],
    # Not required, but helps for mixing and auto-caching.
    num_input_examples=num_cp_examples
)

# cp_task = t5.data.TaskRegistry.get("categories_prediction")
# ds = cp_task.get_dataset(split="validation", sequence_length={"inputs": 128, "targets": 32})
# print("A few preprocessed validation examples...")
# for ex in tfds.as_numpy(ds.take(5)):
#   print(ex)


# ==================================== Values prediction ======================================

vp_tsv_path = {
    "train": os.path.join(DATA_DIR, 'preprocessed_tsv/task2/train_opp115_vp.tsv'),
    "validation": os.path.join(DATA_DIR, 'preprocessed_tsv/task2/val_opp115_vp.tsv'),
    "test": os.path.join(DATA_DIR, 'preprocessed_tsv/task2/test_opp115_vp.tsv'),
}

num_vp_examples = {
    "train": 8926,
    "validation": 1255,
    "test": 8601
}

def vp_dataset_fn(split, shuffle_files=False):
  # We only have one file for each split.
  del shuffle_files

  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(vp_tsv_path[split])
  # Split each "'input'\t'target'" example into (input, target) tuple.
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # Map each tuple to a {"input": ... "target": ...} dict.
  ds = ds.map(lambda *ex: dict(zip(["input", "target"], ex)))
  return ds



def vp_preprocessor(ds):
  def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    text = tf.strings.lower(text)
    text = tf.strings.strip(text)
    # text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
    return text

  def to_inputs_and_targets(ex):
    """Map {"input": ..., "target": ...}->{"inputs": ..., "targets": ...}."""
    return {
        "inputs": normalize_text(ex["input"]),
        "targets": normalize_text(ex["target"])
    }
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)


def vp_postprocessor(string, **unused_kwargs):
  string = string.lower()
  predictions = re.split(r'\] ?', string)
  pp_predictions = []
  for pred in predictions:
    t = pred.split(' [')
    if len(t)==0:
      continue
    elif len(t) == 1:
      if t[0] == '':
        continue
      else:
        t = (t[0], '')
    pp_predictions.append(tuple(t))
  # predictions = [tuple(pred.split(' [')) for pred in predictions if pred]
  return (string, pp_predictions)


t5.data.TaskRegistry.add(
    "values_prediction",
    # Specify the task type.
    t5.data.Task,
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=vp_dataset_fn,
    splits=["train", "validation", "test"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=[vp_preprocessor],
    # Lowercase targets before computing metrics.
    postprocess_fn=vp_postprocessor, 
    # We'll use accuracy as our evaluation metric.
    # metric_fns=[vp_metric],
    # Not required, but helps for mixing and auto-caching.
    num_input_examples=num_vp_examples
)

# Load and print a few examples.
# vp_task = t5.data.TaskRegistry.get("values_prediction")
# ds = vp_task.get_dataset(split="validation", sequence_length={"inputs": 128, "targets": 32})
# print("A few preprocessed validation examples...")
# for ex in tfds.as_numpy(ds.take(3)):
#   print(ex)

# ==================================== Mixture of tasks ======================================
t5.data.MixtureRegistry.add( 
    "opp115",
    ["categories_prediction", "values_prediction"],
     default_rate= 1
)

