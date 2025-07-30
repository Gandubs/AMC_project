import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from otdd.pytorch.distance import DatasetDistance, FeatureCost



# === MOD FAMILY MAP ===
#Ghosting analog signals as OOD
family_map = {
    4:0, 14:0, 24:0, 33:0, 44:0, 54:0,  # Analog
    2:1, 12:1, 22:1, 32:1,              # FSK
    3:2, 13:2, 23:2,                    # PAM
    0:3, 10:3, 20:3, 30:3, 40:3, 50:3,  # PSK
    1:4, 11:4, 21:4, 31:4, 41:4, 51:4, 61:4  # QAM
}



class HisarModDataset(Dataset):
    #Data, labels converted to pytorch tensors if numpy arrays, no transformation applied
    def __init__(self, data, labels, transform=None):
        self.data = torch.from_numpy(data).float() if isinstance(data, np.ndarray) else data.float()
        self.labels = torch.from_numpy(labels).long() if isinstance(labels, np.ndarray) else labels.long()
        self.targets = self.labels
        self.transform = transform
    #returns length of dataset
    def __len__(self): return len(self.data)
    #returns sample at index idx, applies transformation if specified, return a tuple of (data, label)
    def __getitem__(self, idx):
        x, y = self.data[idx].clone(), self.labels[idx].clone()
        if self.transform: x = self.transform(x)
        return x, y



class HisarModCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.noise = nn.Identity()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1,3), padding=(0,1))
        self.pool1 = nn.MaxPool2d((1,2))
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=(1,3), padding=(0,1))
        self.pool2 = nn.MaxPool2d((1,2))
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=(1,3), padding=(0,1))
        self.pool3 = nn.MaxPool2d((1,2))
        self.dropout3 = nn.Dropout(0.5)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(1,3), padding=(0,1))
        self.pool4 = nn.MaxPool2d((1,2))
        self.dropout4 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8192, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.noise(x)
        x = F.relu(self.conv1(x)); x = self.pool1(x); x = self.dropout1(x)
        x = F.relu(self.conv2(x)); x = self.pool2(x); x = self.dropout2(x)
        x = F.relu(self.conv3(x)); x = self.pool3(x); x = self.dropout3(x)
        x = F.relu(self.conv4(x)); x = self.pool4(x); x = self.dropout4(x)
        x = self.flatten(x); x = F.relu(self.fc1(x)); return self.fc2(x)
    def get_features(self, x):
        x = x.unsqueeze(1); x = self.noise(x)
        x = F.relu(self.conv1(x)); x = self.pool1(x); x = self.dropout1(x)
        x = F.relu(self.conv2(x)); x = self.pool2(x); x = self.dropout2(x)
        x = F.relu(self.conv3(x)); x = self.pool3(x); x = self.dropout3(x)
        x = F.relu(self.conv4(x)); x = self.pool4(x); x = self.dropout4(x)
        x = self.flatten(x); x = F.relu(self.fc1(x)); return x




class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    def forward(self, x):
        with torch.no_grad():
            return self.base_model.get_features(x)




def prepare_otdd_datasets():
    """
    Prepares datasets for OTDD-based OOD detection.
    
    - Training: non-analog only (FSK, PAM, PSK, QAM)
    - Validation: non-analog only
    - Test: remaining non-analog + all analog (OOD)
    - OOD: analog only
    - Threshold: 10% training + 10% OOD
    """
    np.random.seed(42)

    # Load data and map to families
    data = np.load('train_data_reshaped_1024.npy')
    labels = np.load('train_labels_continuous.npy')
    family_labels = np.array([family_map[label] for label in labels])

    # Boolean masks
    in_dist_mask = family_labels != 0  # FSK, PAM, PSK, QAM
    ood_mask = family_labels == 0      # Analog only

    # Filter and remap in-distribution labels (0–3)
    in_dist_data = data[in_dist_mask]
    in_dist_labels = family_labels[in_dist_mask] - 1

    ood_data = data[ood_mask]
    ood_labels = family_labels[ood_mask]  # Remain 0

    # === Split in-distribution: 60% train, 20% val, 20% test ===
    total_in = len(in_dist_data)
    indices = np.random.permutation(total_in)
    
    n_train = int(0.6 * total_in)
    n_val = int(0.2 * total_in)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_in_idx = indices[n_train + n_val:]

    train_data, train_labels = in_dist_data[train_idx], in_dist_labels[train_idx]
    val_data, val_labels = in_dist_data[val_idx], in_dist_labels[val_idx]
    test_in_data, test_in_labels = in_dist_data[test_in_idx], in_dist_labels[test_in_idx]

    # === Final test set: remaining in-dist + all OOD (analog) ===
    test_data = np.concatenate([test_in_data, ood_data])
    test_labels = np.concatenate([test_in_labels, ood_labels])

    # === Threshold set: 10% train + 10% OOD ===
    #Enforce 10% ood length on 10% of train length, not clear if otdd needs same length
    n_thresh = int(0.1 * len(ood_data))
    thresh_train_idx = np.random.choice(len(train_data), n_thresh, replace=False)
    thresh_ood_idx = np.random.choice(len(ood_data), n_thresh, replace=False)

    thresh_data = np.concatenate([train_data[thresh_train_idx], ood_data[thresh_ood_idx]])
    thresh_labels = np.concatenate([train_labels[thresh_train_idx], ood_labels[thresh_ood_idx]])

    print(f"Dataset splits according to OTDD OOD strategy:")
    print(f"  Training set (no analog): {len(train_data)}")
    print(f"  Validation set (no analog): {len(val_data)}")
    print(f"  Test set (mixed): {len(test_data)} (in-dist: {len(test_in_data)}, analog: {len(ood_data)})")
    print(f"  OOD set (analog only): {len(ood_data)}")
    print(f"  Threshold computation set: {len(thresh_data)} , split as {len(thresh_train_idx)} train + {len(thresh_ood_idx)} OOD")

    return (
        HisarModDataset(train_data, train_labels),
        HisarModDataset(val_data, val_labels),
        HisarModDataset(test_data, test_labels),
        HisarModDataset(ood_data, ood_labels),
        HisarModDataset(thresh_data, thresh_labels)
    )





def train_with_progress(model, train_loader, val_loader, criterion, optimizer, device, epochs=1):
    model.to(device); best_acc = 0.0
    for epoch in range(epochs):
        model.train(); correct, total, loss_sum = 0, 0, 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item(); total += y.size(0)
            if (batch_idx + 1) % 100 == 0:
                current_acc = correct / total
                print(f"[Training] Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Accuracy: {current_acc:.4f}")
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Final Train Acc: {correct/total:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "hisarmod_otdd_best_model.pth")
            print("✅ Model checkpoint saved")




def validate(model, loader, criterion, device):
    model.eval(); total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x); loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item(); total += y.size(0)
    return loss_sum/total, correct/total




class OTDDOODDetector:
    def __init__(self, model, device):
        self.device = device
        self.model = model.to(device)
        self.feature_extractor = FeatureExtractor(model).to(device).eval()
        self.threshold = None
        self.reference_features = None
        self.reference_labels = None
        



    def cache_reference_features(self, train_loader, num_samples=2000):
        """Cache training reference features for consistent OTDD comparison"""
        print(f"🔄 Caching {num_samples} training reference features...")
        features, labels, count = [], [], 0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad(): 
                feat = self.feature_extractor(x)
            for i in range(feat.size(0)):
                if count >= num_samples: break
                features.append(feat[i]); labels.append(y[i]); count += 1
            if count >= num_samples: break
        self.reference_features = torch.stack(features)
        self.reference_labels = torch.stack(labels)
        print(f"✅ Cached {len(self.reference_features)} reference features")
    



    def extract_features(self, loader, maxsamples=2000):
        features, labels, count = [], [], 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad(): 
                feat = self.feature_extractor(x)
            for i in range(feat.size(0)):
                if count >= maxsamples: break
                features.append(feat[i]); labels.append(y[i]); count += 1
            if count >= maxsamples: break
        return torch.stack(features), torch.stack(labels)
    




    def make_feature_dataset(self, features, labels):
        class FeatureDataset(Dataset):
            def __init__(self, feat, lab): 
                self.data = feat.cpu(); self.targets = lab.cpu()
            def __len__(self): return len(self.data)
            def __getitem__(self, idx): return self.data[idx], self.targets[idx]
        return FeatureDataset(features, labels)
    



    def compute_feature_distance(self, features1, labels1, features2, labels2):
        d1 = self.make_feature_dataset(features1, labels1)
        d2 = self.make_feature_dataset(features2, labels2)
        l1 = DataLoader(d1, batch_size=32, shuffle=False)
        l2 = DataLoader(d2, batch_size=32, shuffle=False)
        identity_embedding = nn.Identity().to(self.device)
        cost = FeatureCost(
            src_embedding=identity_embedding,
            tgt_embedding=identity_embedding,
            src_dim=(128,), tgt_dim=(128,), p=2, device=self.device
        )
        dist = DatasetDistance(l1, l2, inner_ot_method='exact', feature_cost=cost, device=self.device)
        return float(dist.distance(maxsamples=min(len(features1), len(features2))))
    




    def calibrate_threshold(self, train_loader, ood_loader, maxsamples=5000):
        """
        Calibrate threshold using pure ID vs OOD comparison:
        Reference: subset of training data (ID only)
        Comparison: OOD data only
        """
        print("🎯 Calibrating OTDD OOD detection threshold...")
        
        # Cache reference features from training set (ID only)
        self.cache_reference_features(train_loader, num_samples=maxsamples)
        
        # Extract features from OOD set only
        print("🔄 Extracting OOD features for threshold calibration...")
        ood_feat, ood_lab = self.extract_features(ood_loader, maxsamples)
        
        # Compute OTDD distance between reference (training ID) and pure OOD
        print("📊 Computing OTDD distance for threshold calibration...")
        distance = self.compute_feature_distance(
            self.reference_features[:maxsamples], self.reference_labels[:maxsamples],
            ood_feat, ood_lab
        )
        
        # Set threshold based on this ID vs OOD distance
        # You can use the computed distance directly or apply a factor
        self.threshold = distance * 0.6  # must tune wrt mean and std deviation instead of hard-assignment
        # Or keep your hardcoded approach: self.threshold = 50
        
        print(f"📊 Threshold Calibration Complete:")
        print(f"   Reference set size (ID): {len(self.reference_features)}")
        print(f"   OOD comparison set size: {len(ood_feat)}")
        print(f"   Computed ID vs OOD distance: {distance:.4f}")
        print(f"   Threshold set to: {self.threshold:.4f}")
        
        return self.threshold
    



    def detect(self, test_loader, maxsamples=100):
        """Detect OOD by comparing test batch to cached reference features"""
        if self.reference_features is None:
            raise ValueError("Must call calibrate_threshold() first to cache reference features")
        
        test_feat, test_lab = self.extract_features(test_loader, maxsamples)
        ref_samples = min(maxsamples, len(self.reference_features))
        
        distance = self.compute_feature_distance(
            self.reference_features[:ref_samples], self.reference_labels[:ref_samples],
            test_feat, test_lab
        )
        
        return distance > self.threshold, distance





def evaluate_mixed_test_set(detector, test_loader, batch_size=100):
    """
    Evaluate OOD detection on mixed test set (contains both ID and OOD samples)
    """
    print("\n=== 📊 Mixed Test Set OOD Detection Evaluation ===")
    
    test_dataset = test_loader.dataset
    max_batches = int(np.ceil(len(test_dataset) / batch_size))
    
    all_distances = []
    all_predictions = []
    all_true_labels = []  # 0 for ID, 1 for OOD (analog)
    
    print(f"Processing {max_batches} batches from mixed test set...")
    
    for i in range(max_batches):
        indices = list(range(i*batch_size, min((i+1)*batch_size, len(test_dataset))))
        subset = torch.utils.data.Subset(test_dataset, indices)
        subset_loader = DataLoader(subset, batch_size=len(indices), shuffle=False)
        
        # Get true labels for this batch
        true_batch_labels = []
        for idx in indices:
            sample_label = test_dataset.targets[idx].item()
            true_batch_labels.append(1 if sample_label == 0 else 0)  # 1 if analog (OOD), 0 if others (ID)
        
        # Majority vote for batch label (since OTDD works on batches)
        batch_is_ood = sum(true_batch_labels) > len(true_batch_labels) // 2
        
        # Run OOD detection
        is_ood_pred, distance = detector.detect(subset_loader, maxsamples=len(indices))
        
        all_distances.append(distance)
        all_predictions.append(1 if is_ood_pred else 0)
        all_true_labels.append(1 if batch_is_ood else 0)
        
        if (i + 1) % 50 == 0 or i == max_batches - 1:
            current_acc = sum(p == t for p, t in zip(all_predictions, all_true_labels)) / len(all_predictions)
            print(f"   Processed {i+1}/{max_batches} batches, Accuracy: {current_acc:.4f} (D={distance:.2f})")
    
    # Calculate final metrics
    all_distances = np.array(all_distances)
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    # Compute confusion matrix elements
    tp = sum((p == 1) and (t == 1) for p, t in zip(all_predictions, all_true_labels))
    tn = sum((p == 0) and (t == 0) for p, t in zip(all_predictions, all_true_labels))
    fp = sum((p == 1) and (t == 0) for p, t in zip(all_predictions, all_true_labels))
    fn = sum((p == 0) and (t == 1) for p, t in zip(all_predictions, all_true_labels))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    print(f"\n📊 Final Mixed Test Set OOD Detection Performance:")
    print(f"   True Positive Rate (OOD Detection): {tpr:.4f} ({tp}/{tp + fn})")
    print(f"   False Positive Rate (ID as OOD): {fpr:.4f} ({fp}/{fp + tn})")
    print(f"   True Negative Rate (ID Detection): {tnr:.4f} ({tn}/{tn + fp})")
    print(f"   False Negative Rate (OOD as ID): {fnr:.4f} ({fn}/{fn + tp})")
    print(f"   Overall Accuracy: {accuracy:.4f}")
    print(f"   Average Distance: {all_distances.mean():.4f} ± {all_distances.std():.4f}")
    print(f"   Threshold: {detector.threshold:.4f}")
    
    return {
        'tpr': tpr, 'fpr': fpr, 'tnr': tnr, 'fnr': fnr, 'accuracy': accuracy,
        'distances': all_distances, 'threshold': detector.threshold
    }




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare datasets according to OTDD OOD strategy
    train_ds, val_ds, test_ds, ood_ds, threshold_ds = prepare_otdd_datasets()
    
    train_loader = DataLoader(train_ds, 64, shuffle=True)
    val_loader = DataLoader(val_ds, 64)
    test_loader = DataLoader(test_ds, 64)
    ood_loader = DataLoader(ood_ds, 64)
    threshold_loader = DataLoader(threshold_ds, 64)
    
    # Train model (only on non-analog data)
    model = HisarModCNN(num_classes=4)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("=== Training Phase ===")
    train_with_progress(model, train_loader, val_loader, criterion, optimizer, device)
    
    # Load best model
    model.load_state_dict(torch.load("hisarmod_otdd_best_model.pth", map_location=device))
    model.eval().to(device)
    
    # Test in-distribution performance on validation set
    print("\n=== In-Distribution Validation Performance ===")
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f"Validation Accuracy (In-Distribution): {val_acc:.4f}")
    
    # Initialize and calibrate OTDD OOD detector
    print("\n=== OTDD OOD Detection Setup ===")
    detector = OTDDOODDetector(model, device)
    
    # ✅ UPDATED: Calibrate using pure ID vs OOD comparison
    threshold = detector.calibrate_threshold(train_loader, ood_loader, maxsamples=5000)
    
    # Evaluate on mixed test set
    metrics = evaluate_mixed_test_set(detector, test_loader, batch_size=100)
    
    print(f"\n✅ OTDD OOD Detection Complete!")
    print(f"🎯 Your OTDD strategy successfully implemented with threshold: {threshold:.4f}")

if __name__ == '__main__':
    main()
