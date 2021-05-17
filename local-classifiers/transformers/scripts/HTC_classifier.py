import pandas as pd
import numpy as np
import torch
from simpletransformers.classification import MultiLabelClassificationModel
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
from sklearn.metrics import classification_report
import os
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
INPUT_PATH = FILE_PATH+"/datasets"
OUTPUT_PATH = FILE_PATH+"/models"

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def cls_report(golden, predictions, target_names):
    df = pd.DataFrame(columns=["label", "precision","prec-abs", "prec-avg", "recall","rec-abs","rec-avg", "F1", "F1-abs", "F1-avg", "support"])
    TP_total =  FP_total = TN_total = FN_total = 0
    for i, label in enumerate(target_names):
        y_actual = golden[:, i]
        y_hat = predictions[:, i]
        TP, FP, TN, FN = perf_measure(y_actual, y_hat)
        TP_total+=TP
        FP_total+=FP
        TN_total+=TN
        FN_total+=FN
        if((TP+FP) == 0):
            prec = 0
        else:
            prec = 100 * TP/(TP+FP)
        if((TN+FN) == 0):
            prec_abs = 0
        else:
            prec_abs = 100 * TN/(TN+FN)
        prec_avg = (prec + prec_abs)/2
        if((TP+FN) == 0):
            rec = 0
        else:
            rec = 100 * TP/(TP+FN)
        if((TN+FP) == 0):
            rec_abs = 0
        else:
            rec_abs = 100 * TN/(TN+FP)
        rec_avg = (rec +rec_abs)/2
        if((prec+rec) == 0):
            f1 = 0
        else:
            f1 = 2*(prec*rec)/(prec+rec)
        if((prec_abs+rec_abs) == 0):
            f1_abs = 0
        else:
            f1_abs = 2*(prec_abs*rec_abs)/(prec_abs+rec_abs)
        f1_avg = (f1 + f1_abs)/2
        df = df.append({"label": label, "precision": round(prec), "support":TP+FN, "prec-abs": round(prec_abs), "prec-avg": round(prec_avg), "recall": round(rec), "rec-abs": round(rec_abs), "rec-avg": round(rec_avg), "F1": round(f1), "F1-abs": round(f1_abs), "F1-avg": round(f1_avg)}, ignore_index=True)

    macro_avgs = {"label": "MACRO AVERAGE", "precision": np.mean(df["precision"]), "support": np.mean(df["support"]), "prec-abs": np.mean(df["prec-abs"]), "prec-avg": np.mean(df["prec-avg"]), "recall": np.mean(df["recall"]), "rec-abs": np.mean(df["rec-abs"]), "rec-avg": np.mean(df["rec-avg"]), "F1": np.mean(df["F1"]), "F1-abs": np.mean(df["F1-abs"]), "F1-avg": np.mean(df["F1-avg"])}
    prec_total = 100 * TP_total/(TP_total+FP_total)
    prec_abs_total = 100 * TN_total / (TN_total + FN_total)
    prec_avg_total = (prec_total + prec_abs_total) / 2
    rec_total = 100 * TP_total / (TP_total + FN_total)
    rec_abs_total = 100 * TN_total/(TN_total+FP_total)
    rec_avg_total = (rec_total + rec_abs_total) / 2
    f1_total = 2 * (prec_total * rec_total) / (prec_total + rec_total)
    f1_abs_total = 2 * (prec_abs_total * rec_abs_total) / (prec_abs_total + rec_abs_total)
    f1_avg_total = (f1_total + f1_abs_total) / 2
    micro_avgs = {"label": "MICRO AVERAGE","precision": round(prec_total), "support":TP_total+FN_total, "prec-abs": round(prec_abs_total), "prec-avg": round(prec_avg_total), "recall": round(rec_total), "rec-abs": round(rec_abs_total), "rec-avg": round(rec_avg_total), "F1": round(f1_total), "F1-abs": round(f1_abs), "F1-avg": round(f1_avg_total)}

    df = df.append(macro_avgs,
                   ignore_index=True)
    df = df.append(micro_avgs,
                   ignore_index=True)

    pd.set_option("max_columns", 20)
    print(df)

class HTC_classifier():
    def __init__(self, ephocs=1, model_type='xlnet', model_path='xlnet-base-cased', output_path=None):
        self.one_hot = MultiLabelBinarizer()
        self.use_cuda = torch.cuda.is_available()
        self.ephocs = ephocs
        self.model_type = model_type
        self.model_path = model_path
        self.output_path = output_path
        self.load_model()
        print("self.use_cuda: ", self.use_cuda)

    def one_hot_encoder(self, labels, fit=False):
        if(fit):
            return self.one_hot.fit_transform(labels)
        return self.one_hot.transform(labels)


    def one_hot_to_text(self, one_hot_matrix):
        return self.one_hot.inverse_transform(one_hot_matrix)


    def fit(self, text_series, labels):
        labels = self.one_hot_encoder(labels, True)
        print("fitting model, artifacts will be saved to: ", self.output_path)
        self.model = MultiLabelClassificationModel(self.model_type, self.model_path, num_labels=np.array(labels).shape[1] ,use_cuda=self.use_cuda,
                                                   args={'reprocess_input_data': True, 'output_dir' :self.output_path , 'overwrite_output_dir': True,
                                                         'num_train_epochs': self.ephocs, 'save_steps' :0})
        train_df = pd.DataFrame({"text": text_series, "labels": labels.tolist()})
        self.model.train_model(train_df)
        self.save_enoder()
        return self.output_path

    def save_enoder(self):
        # Serialize both the pipeline and binarizer to disk.
        encoder_filename = self.output_path +"/encoder.pkl"
        with open(encoder_filename, 'wb') as f:
            pickle.dump((self.one_hot), f)
        print("model saved to: ", self.output_path)

    def load_model(self):
        # Hydrate the serialized objects.
        encoder_filename = self.model_path + "/encoder.pkl"
        print("loading model from: ", self.model_path)
        try:
            with open(encoder_filename, 'rb') as f:
                self.one_hot = pickle.load(f)
            self.model = MultiLabelClassificationModel('xlnet', self.model_path,use_cuda=self.use_cuda)
        except Exception as e:
            print("error:", e)
            print("couldn't load models from disk")

    def predict(self, text_series):
        text_list = text_series.tolist()
        predictions, _ = self.model.predict(text_list)
        return self.one_hot_to_text(np.array(predictions))

    def evaluate(self, text_series, golden_labels):
        predictions = self.predict(text_series)
        predictions_oh = self.one_hot_encoder(predictions)
        golden_labels_oh = self.one_hot_encoder(golden_labels)
        categories = self.one_hot.classes_
        print(cls_report(golden_labels_oh, predictions_oh, target_names=categories))
