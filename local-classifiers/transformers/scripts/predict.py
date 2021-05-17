
import pandas as pd
import time
import json
import os
import gc
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
INPUT_PATH = FILE_PATH+"/inference_data/segments.csv"
MODELS_PATH = FILE_PATH + "/models_final"
OUTPUT_FILENAME = FILE_PATH+"/inference_data/predictions.json"
from HTC_classifier import HTC_classifier



class Hierarchical_classifier():
  def __init__(self, models_path):
    self.models_path = models_path
    self.model_prefix = "classifier_"
    self.category_atttributes_map = {'Introductory/Generic': [],
                                     'First Party Collection/Use': ['Action First-Party', 'Personal Information Type',
                                                                    'Purpose', 'User Type', 'Does/Does Not',
                                                                    'Identifiability', 'Choice Type', 'Collection Mode',
                                                                    'Choice Scope'],
                                     'User Choice/Control': ['Choice Type', 'Choice Scope', 'Purpose',
                                                             'Personal Information Type', 'User Type'],
                                     'International and Specific Audiences': ['Audience Type'],
                                     'Third Party Sharing/Collection': ['Action Third Party', 'Third Party Entity',
                                                                        'Personal Information Type', 'Purpose',
                                                                        'User Type', 'Does/Does Not', 'Identifiability',
                                                                        'Choice Type', 'Collection Mode',
                                                                        'Choice Scope'],
                                     'Data Security': ['Security Measure'],
                                     'Practice not covered': [],
                                     'User Access, Edit and Deletion': ['Access Scope', 'User Type', 'Access Type'],
                                     'Privacy contact information': [],
                                     'Policy Change': ['Change Type', 'Notification Type', 'User Choice'],
                                     'Data Retention': ['Retention Period', 'Retention Purpose',
                                                        'Personal Information Type'],
                                     'Do Not Track': ['Do Not Track policy']
                                     }

  def load_model(self, model_name):
    model_id = "classifier_" + "_".join(model_name.split())
    model_path = self.models_path + "/" + model_id
    return HTC_classifier(model_type='xlnet', model_path=model_path)

  def predict_using_model(self, attribute, text_list):
    results = self.load_model(attribute).predict(pd.Series(text_list))
    gc.collect()  # IMPORTANT: this prevents a mem leak in transformers library: https://github.com/huggingface/transformers/issues/1742
    return results

  def predict(self, text_list):
    cat_texts_map = {}
    category_labels = self.predict_using_model('Category', text_list)
    print("category_labels: ", category_labels)
    for i, text in enumerate(text_list):
      categories = category_labels[i]
      for category in categories:
        if (category not in cat_texts_map):
          cat_texts_map[category] = []
        cat_texts_map[category].append(text)
    text_result_map = {}
    for cat, texts in cat_texts_map.items():
      attributes = self.category_atttributes_map[cat]
      print("cat: {} - attributes: {}".format(cat, attributes))
      for i, text in enumerate(texts):
          # make sure we add category to text_result_map even if no attributes
          if (text not in text_result_map):
              text_result_map[text] = {}
          text_result_map[text][cat] = {}
      for attribute in attributes:
          attribute_labels = self.predict_using_model(attribute, texts)
          for i, text in enumerate(texts):
            text_result_map[text][cat][attribute] = attribute_labels[i]
    results = []
    for text in text_list:
      # make sure results maintains same order has text
      if (text in text_result_map):
        results.append(text_result_map[text])
      else:
        results.append({})
    return results


def infer(text_list):
  hierarchical_classifier = Hierarchical_classifier(MODELS_PATH)
  predictions_dict = hierarchical_classifier.predict(text_list)
  print("predictions: ", predictions_dict)
  with open(OUTPUT_FILENAME, 'w') as fp:
      json.dump(predictions_dict, fp)


if __name__ == "__main__":
    text_series = pd.read_csv(INPUT_PATH, squeeze=True, header=None)
    text_list = text_series.tolist()
    print(text_list)
    # execute only if run as a script
    print("starting classifier.... reading input from: ", INPUT_PATH)
    start_time = time.time()
    infer(text_list)
    print("inference finished, output saved to: ", OUTPUT_FILENAME)
    print("---took %s seconds ---" % (time.time() - start_time))



