import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def get_confusion_matrix(y_true, y_pred, num_classes=7):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def get_binary_metrics(y_true_bin, y_pred_bin):
    tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_true_bin)
    
    return tp, tn, fp, fn, precision, recall, f1, accuracy

def get_roc_curve(y_true_bin, stress_scores):
    # We test thresholds from 0 to 1
    thresholds = np.linspace(0.0, 1.0, 200)
    tprs = []
    fprs = []
    for th in thresholds:
        y_pred_th = (stress_scores >= th).astype(int)
        tp = np.sum((y_true_bin == 1) & (y_pred_th == 1))
        fn = np.sum((y_true_bin == 1) & (y_pred_th == 0))
        fp = np.sum((y_true_bin == 0) & (y_pred_th == 1))
        tn = np.sum((y_true_bin == 0) & (y_pred_th == 0))
        
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tprs.append(tpr_val)
        fprs.append(fpr_val)
    
    # Sort for AUC calculation (monotonically increasing FPR)
    # Add boundary points (0,0) and (1,1)
    fprs.append(1.0)
    tprs.append(1.0)
    fprs.append(0.0)
    tprs.append(0.0)
    
    pairs = sorted(list(zip(fprs, tprs)))
    fpr_sorted = np.array([p[0] for p in pairs])
    tpr_sorted = np.array([p[1] for p in pairs])
    
    # Deduplicate matching FPR entries to avoid trapezoidal artifacts
    fpr_unique, indices = np.unique(fpr_sorted, return_index=True)
    tpr_unique = tpr_sorted[indices]
    
    # Trapezoidal rule for AUC
    auc_val = 0.0
    for i in range(1, len(fpr_unique)):
        auc_val += 0.5 * (fpr_unique[i] - fpr_unique[i-1]) * (tpr_unique[i] + tpr_unique[i-1])
        
    return fpr_unique, tpr_unique, auc_val

def plot_confusion_matrix(cm, classes, title, filename, cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right', fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    # Normalize matrix for display
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # handle div by zero

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        text = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
        plt.text(j, i, text,
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=9)

    plt.ylabel('True Class', fontsize=12, labelpad=10)
    plt.xlabel('Predicted Class', fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    test_dir = 'data/test'
    model_path = 'src/models/model.h5'
    plot_dir = 'output/plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    print("Loading pre-trained Keras model...")
    model = tf.keras.models.load_model(model_path)
    
    print("Loading test data generator...")
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False
    )
    
    # Target classes from alphabetical subdirectories:
    # 0: angry, 1: disgusted, 2: fearful, 3: happy, 4: neutral, 5: sad, 6: surprised
    emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
    
    print("Running predictions on test set...")
    preds = model.predict(test_generator)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_generator.classes
    
    # 1. Multi-class evaluation
    cm_7x7 = get_confusion_matrix(y_true, y_pred, num_classes=7)
    plot_confusion_matrix(cm_7x7, emotion_labels, '7-Class Emotion Confusion Matrix', os.path.join(plot_dir, 'confusion_matrix_7class.png'))
    print("Saved 7-class confusion matrix.")
    
    # 2. Binary Stress vs Non-Stress classification
    # Stress classes: Angry (0), Fearful (2), Sad (5)
    # Non-Stress classes: Disgusted (1), Happy (3), Neutral (4), Surprised (6)
    stress_classes = [0, 2, 5]
    y_true_bin = np.isin(y_true, stress_classes).astype(int)
    y_pred_bin = np.isin(y_pred, stress_classes).astype(int)
    
    # Continuous stress score: sum of probabilities of stress emotions
    stress_scores = preds[:, 0] + preds[:, 2] + preds[:, 5]
    
    cm_binary = get_confusion_matrix(y_true_bin, y_pred_bin, num_classes=2)
    binary_labels = ['Non-Stress', 'Stress']
    plot_confusion_matrix(cm_binary, binary_labels, 'Stress vs Non-Stress Confusion Matrix', os.path.join(plot_dir, 'confusion_matrix_binary.png'), cmap=plt.cm.Oranges)
    print("Saved binary confusion matrix.")
    
    # Calculate binary metrics
    tp, tn, fp, fn, precision, recall, f1, bin_acc = get_binary_metrics(y_true_bin, y_pred_bin)
    
    # ROC Curve & AUC
    fpr, tpr, auc_val = get_roc_curve(y_true_bin, stress_scores)
    
    # Plot ROC Curve
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#ff7f0e', lw=2.5, label=f'Stress Detection (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='#38bdf8', lw=1.5, linestyle='--', label='Random Classifier')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Alarm Rate (FPR)', fontsize=12, labelpad=10)
    plt.ylabel('Stress Detection Rate (TPR / Recall)', fontsize=12, labelpad=10)
    plt.title('Stress Detection ROC Curve', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    print("Saved Stress detection ROC curve.")
    
    # Write metrics summary
    metrics_path = os.path.join(plot_dir, 'evaluation_metrics_summary.txt')
    with open(metrics_path, 'w') as f:
        f.write("=== MODEL EVALUATION METRICS (STRESS VS NON-STRESS) ===\n")
        f.write(f"Total Test Samples: {len(y_true)}\n")
        f.write(f"Stress Class Samples (Angry, Fearful, Sad): {np.sum(y_true_bin)}\n")
        f.write(f"Non-Stress Class Samples: {len(y_true) - np.sum(y_true_bin)}\n\n")
        f.write(f"True Positives (TP): {tp}\n")
        f.write(f"True Negatives (TN): {tn}\n")
        f.write(f"False Positives (FP) [False Alarm]: {fp}\n")
        f.write(f"False Negatives (FN) [Missed Stress]: {fn}\n\n")
        f.write(f"Binary Classification Accuracy: {bin_acc:.4f}\n")
        f.write(f"Precision (PPV): {precision:.4f}\n")
        f.write(f"Recall / TPR (Stress Detection Rate): {recall:.4f}\n")
        f.write(f"FPR (False Alarm Rate): {(fp / (fp + tn)):.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Area Under ROC Curve (AUC): {auc_val:.4f}\n")
    
    print(f"\nAll metrics successfully saved to {metrics_path}")
    print("=== Summary ===")
    print(f"Accuracy: {bin_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Stress Detection Rate): {recall:.4f}")
    print(f"False Alarm Rate: {(fp / (fp + tn)):.4f}")
    print(f"AUC: {auc_val:.4f}")

if __name__ == '__main__':
    main()
