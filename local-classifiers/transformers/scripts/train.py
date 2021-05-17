import pandas as pd
import time
import pickle
import os
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
INPUT_PATH = FILE_PATH+"/datasets"
OUTPUT_PATH = FILE_PATH+"/models"
from HTC_classifier import HTC_classifier

def train_model(attribute_name, df=[], ephocs=1, base_model_type="xlnet", base_model_name="xlnet-base-cased", frac=0.2, train_df=[], test_df=[]):
  model_name = "classifier_"+"_".join(attribute_name.split())
  output_path = OUTPUT_PATH +"/"+model_name
  classifier = HTC_classifier(ephocs=ephocs, model_type=base_model_type, model_path=base_model_name, output_path=output_path)
  if(len(train_df)==0 or len(test_df)==0):
    print("generating train, test splits")
    train_df = df.copy().dropna(subset=["Text", attribute_name])
    test_df = train_df.sample(frac=frac)
    train_df.drop(test_df.index, inplace=True)
  print("training set size: {} -- test set size: {}".format(len(train_df), len(test_df)) )
  classifier.fit(train_df["Text"], train_df[attribute_name].tolist())
  golden_labels = test_df[attribute_name]
  classifier.evaluate(test_df["Text"], golden_labels)
  return classifier


def train_all(ephocs=5, base_model_type="xlnet", base_model_name="xlnet-base-cased"):
    merged_df = pd.read_pickle(INPUT_PATH + "/clean_dataset_merged.pkl")
    train_df = merged_df.copy().dropna(subset=["Text"])
    test_df = train_df.sample(frac=0.2)
    train_df.drop(test_df.index, inplace=True)
    training_status_filename = OUTPUT_PATH + "/training_status"
    try:
        with open(training_status_filename, 'rb') as f:
            skip_columns = pickle.load(f)
            print("loaded skip_columns: ", skip_columns)
    except:
        print("couldn't load training status file, will create a new one")
        skip_columns = ["Text"]
    try:
        for attribute_name in train_df.drop(columns=skip_columns):
            print("start training for attribute: " + attribute_name)
            train_model(attribute_name, ephocs=ephocs, base_model_type=base_model_type, base_model_name=base_model_name,
                        train_df=train_df.dropna(subset=[attribute_name]),
                        test_df=test_df.dropna(subset=[attribute_name]))
            skip_columns.append(attribute_name)
            with open(training_status_filename, 'wb') as f:
                pickle.dump(skip_columns, f)
    except Exception as e:
        print("error:", e)



if __name__ == "__main__":
    # execute only if run as a script
    print("starting classifier.... reading input from: ", INPUT_PATH)
    start_time = time.time()
    train_all(ephocs=1, base_model_type="xlnet", base_model_name="xlnet-base-cased")
    print("training finished, output saved to: ", OUTPUT_PATH)
    print("---took %s seconds ---" % (time.time() - start_time))



