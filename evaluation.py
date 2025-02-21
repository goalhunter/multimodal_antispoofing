import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import librosa
from file_reader import AudioDataset
from multimodal import MultiModalDetector
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

eval_csv = '/home/STUDENTS/pb0626/Documents/Assessment/LA-flac-subset/LA_eval_subset.tsv'
eval_la = '/home/STUDENTS/pb0626/Documents/Assessment/LA-flac-subset/ASVspoof2019_LA_eval_subset'

sample_rate = 16000
batch_size = 16
out_len = 6400


eval_dataset = AudioDataset(
        tsv_file=eval_csv,
        la_directory=eval_la,
        sample_rate=sample_rate,
        out_len=out_len
    )

eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiModalDetector()
model.to(device)

FINE_TUNED_MODEL_PATH = 'MultiModal.pt'

def calculate_eer(y_true, y_scores):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer, eer_threshold

# Detailed evaluation function
def detailed_evaluation(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_scores = []
    filenames = []  # Store filenames
    correct_spoofs = []  # Files correctly classified as spoof (0)
    wrong_spoofs = []    # Files wrongly classified as spoof (0)
    
    with torch.no_grad():
        for batch_idx, (audio, labels) in enumerate(tqdm(dataloader, desc="Detailed Evaluation", leave=False)):
            # Get filenames for this batch
            batch_filenames = [dataloader.dataset.file_names[idx] for idx in range(
                batch_idx * dataloader.batch_size,
                min((batch_idx + 1) * dataloader.batch_size, len(dataloader.dataset))
            )]
            
            audio, labels = audio.to(device), labels.to(device)
            outputs = model(audio)
            scores = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions and filenames
            for fname, pred, true_label in zip(batch_filenames, predicted.cpu().numpy(), labels.cpu().numpy()):
                if pred == 0:  # Model predicted spoof
                    if true_label == 0:  # Correctly classified spoof
                        correct_spoofs.append(fname)
                    else:  # Wrongly classified as spoof
                        wrong_spoofs.append(fname)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_predictions, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)

    # Calculate EER
    eer, eer_threshold = calculate_eer(all_labels, all_scores)
    print(f"\nEqual Error Rate (EER): {eer * 100:.4f}%")
    print(f"EER Threshold: {eer_threshold:.4f}")

    return correct_spoofs, wrong_spoofs

model.load_state_dict(torch.load(FINE_TUNED_MODEL_PATH))
correct_spoofs, wrong_spoofs = detailed_evaluation(model, eval_loader, device)