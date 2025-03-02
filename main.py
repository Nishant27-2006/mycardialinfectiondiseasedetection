
!pip install wfdb
!pip install scikit-learn
!pip install matplotlib
!pip install seaborn

import wfdb
import numpy as np
import os, re
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, filtfilt
import numpy.fft as fft
import tensorflow as tf

from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, ReLU,
                                     Dropout, MaxPooling1D, GlobalAveragePooling1D,
                                     Dense, Concatenate)
from tensorflow.keras.models import Model

from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve,
                             classification_report, precision_recall_curve)
from sklearn.model_selection import KFold
tf.random.set_seed(42)

db_dir = "ptbdb"

all_signals = []
all_labels = []
patient_ids = []

for pat in sorted(os.listdir(db_dir)):
    if not pat.startswith("patient"):
        continue
    pat_path = os.path.join(db_dir, pat)
    
    label = None
    for f in os.listdir(pat_path):
        if f.endswith(".hea"):
            header_path = os.path.join(pat_path, f)
            with open(header_path, 'r') as fh:
                head = fh.read().lower()
            if "myocardial infarction" in head:
                label = 1
            if label is None:
                label = 0
            break
    if label is None:
        continue
    
    for rec in os.listdir(pat_path):
        if rec.endswith(".dat"):  
            rec_name = rec.replace(".dat", "")
            record_path = os.path.join(pat_path, rec_name)
            try:
                sig, fields = wfdb.rdsamp(record_path)
            except Exception as e:
                print(f"Failed to read {record_path}: {e}")
                continue
            signal = sig.astype(np.float32)
          
            all_signals.append(signal)
            all_labels.append(label)
            patient_ids.append(pat)


fs_orig = 1000  
fs_new = 250    
nyq = 0.5 * fs_orig
low_cut = 0.5 / nyq
high_cut = 40.0 / nyq

b, a = butter(N=4, Wn=[low_cut, high_cut], btype='band')

proc_signals = []
max_len = fs_new * 10  

for signal in all_signals:

    filtered = filtfilt(b, a, signal, axis=0)
    

    factor = fs_orig // fs_new
    ds_signal = filtered[::factor, :]
    
    ds_signal = (ds_signal - np.mean(ds_signal, axis=0)) / (np.std(ds_signal, axis=0) + 1e-8)
    

    if ds_signal.shape[0] > max_len:
        ds_signal = ds_signal[:max_len, :]
    elif ds_signal.shape[0] < max_len:
        pad_width = max_len - ds_signal.shape[0]
        ds_signal = np.pad(ds_signal, ((0, pad_width), (0, 0)), mode='constant')
    
    proc_signals.append(ds_signal.astype(np.float32))

X = np.stack(proc_signals)  
y = np.array(all_labels, dtype=int)
print("Data shape after preprocessing:", X.shape, "Labels shape:", y.shape)


def detect_r_peaks(signal, fs):
    lead = signal[:, 1]  
    thr = np.mean(lead) + 0.5*np.std(lead)
    peaks = []
    last_idx = 0
    min_interval = int(0.3 * fs) 

    for i in range(1, len(lead)-1):
        if (lead[i] > thr) and (lead[i] > lead[i-1]) and (lead[i] > lead[i+1]):
            if i - last_idx > min_interval:
                peaks.append(i)
                last_idx = i
    return np.array(peaks)

fs_feat = 250
feature_list = []
for sig in proc_signals:
   
    r_peaks = detect_r_peaks(sig, fs_feat)
    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / fs_feat
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        heart_rate = 60.0 / mean_rr
    else:
        mean_rr = 0
        std_rr = 0
        heart_rate = 0
    


    lead = sig[:, 1]
    N = len(lead)
    freqs = fft.rfftfreq(N, d=1/fs_feat)
    fft_mag = np.abs(fft.rfft(lead))

    band1 = (freqs >= 0.5) & (freqs <= 5.0)
    band2 = (freqs >= 5.0) & (freqs <= 15.0)
    power1 = np.sum(fft_mag[band1]**2)
    power2 = np.sum(fft_mag[band2]**2)
    if power1 < 1e-6:
        freq_ratio = 0
    else:
        freq_ratio = power2 / power1

    feature_list.append([heart_rate, mean_rr, std_rr, freq_ratio])

features = np.array(feature_list, dtype=np.float32)
print("Feature array shape:", features.shape)



def create_model():
    """Build and compile a new instance of the CNN-fusion model."""
    signal_input = Input(shape=(max_len, 15), name="ecg_signal")
    feat_input = Input(shape=(features.shape[1],), name="engineered_feats")

    x = Conv1D(filters=32, kernel_size=7, padding='same')(signal_input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=64, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=64, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = GlobalAveragePooling1D()(x)

    combined = Concatenate()([x, feat_input])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(32, activation='relu')(combined)
    combined = Dropout(0.3)(combined)

    output = Dense(1, activation='sigmoid')(combined)
    model = Model(inputs=[signal_input, feat_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='AUC')]
    )
    return model


kfold = KFold(n_splits=5, shuffle=True, random_state=42)


acc_scores = []
prec_scores = []
rec_scores = []
f1_scores = []
auc_scores = []

fold_num = 1

for train_index, test_index in kfold.split(X, y):
    print(f"\n----- Fold {fold_num} -----")
    X_train, X_val = X[train_index], X[test_index]
    f_train, f_val = features[train_index], features[test_index]
    y_train, y_val = y[train_index], y[test_index]

    
    model = create_model()

    history = model.fit(
        {"ecg_signal": X_train, "engineered_feats": f_train},
        y_train,
        validation_data=(
            {"ecg_signal": X_val, "engineered_feats": f_val},
            y_val
        ),
        epochs=50, 
        batch_size=16,
        verbose=0  
    )

    y_prob = model.predict({"ecg_signal": X_val, "engineered_feats": f_val})
    y_pred = (y_prob.ravel() >= 0.5).astype(int)

    acc_fold = accuracy_score(y_val, y_pred)
    prec_fold = precision_score(y_val, y_pred, zero_division=0)
    rec_fold  = recall_score(y_val, y_pred, zero_division=0)
    f1_fold   = f1_score(y_val, y_pred, zero_division=0)
    auc_fold  = roc_auc_score(y_val, y_prob.ravel())

    acc_scores.append(acc_fold)
    prec_scores.append(prec_fold)
    rec_scores.append(rec_fold)
    f1_scores.append(f1_fold)
    auc_scores.append(auc_fold)

    print(f"Fold {fold_num} --> Acc: {acc_fold:.3f}, Prec: {prec_fold:.3f}, Rec: {rec_fold:.3f}, F1: {f1_fold:.3f}, AUC: {auc_fold:.3f}")

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No MI", "MI"], yticklabels=["No MI", "MI"])
    plt.title(f"Fold {fold_num} Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    print("\nClassification Report for Fold", fold_num)
    print(classification_report(y_val, y_pred, target_names=["No MI","MI"], zero_division=0))

    fpr, tpr, _ = roc_curve(y_val, y_prob.ravel())
    fold_auc = roc_auc_score(y_val, y_prob.ravel())
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label="Fold %d (AUC=%.3f)" % (fold_num, fold_auc))
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.title(f"Fold {fold_num} ROC Curve")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

    precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_prob.ravel())
    plt.figure(figsize=(5,4))
    plt.plot(recall_vals, precision_vals, label="Fold %d" % fold_num)
    plt.title(f"Fold {fold_num} Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

    fold_num += 1

print("\n============================")
print("5-Fold CV Average Results:")
print("============================")
print(f"Accuracy:  {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
print(f"Precision: {np.mean(prec_scores):.3f} ± {np.std(prec_scores):.3f}")
print(f"Recall:    {np.mean(rec_scores):.3f} ± {np.std(rec_scores):.3f}")
print(f"F1-score:  {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
print(f"AUC:       {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
