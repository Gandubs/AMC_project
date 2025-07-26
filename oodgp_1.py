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
family_map = {
    4:0, 14:0, 24:0, 33:0, 44:0, 54:0,
    2:1, 12:1, 22:1, 32:1,
    3:2, 13:2, 23:2,
    0:3, 10:3, 20:3, 30:3, 40:3, 50:3,
    1:4, 11:4, 21:4, 31:4, 41:4, 51:4, 61:4
}

class HisarModDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.from_numpy(data).float() if isinstance(data, np.ndarray) else data.float()
        self.labels = torch.from_numpy(labels).long() if isinstance(labels, np.ndarray) else labels.long()
        self.targets = self.labels
        self.transform = transform
    def __len__(self): return len(self.data)
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

def prepare_ghosted_datasets():
    data = np.load('train_data_reshaped_1024.npy')
    labels = np.load('train_labels_continuous.npy')
    family_labels = np.array([family_map[label] for label in labels])
    in_dist_mask, ood_mask = family_labels != 0, family_labels == 0
    in_dist_data, in_dist_labels = data[in_dist_mask], family_labels[in_dist_mask] - 1
    ood_data, ood_labels = data[ood_mask], family_labels[ood_mask]
    total_in_dist = len(in_dist_data)
    indices = np.random.permutation(total_in_dist)
    train_idx = indices[:int(8/15*total_in_dist)]
    val_idx = indices[int(8/15*total_in_dist):int(10/15*total_in_dist)]
    test_idx = indices[int(10/15*total_in_dist):]
    print(f"Dataset split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}, OOD={len(ood_data)}")
    return (
        HisarModDataset(in_dist_data[train_idx], in_dist_labels[train_idx]),
        HisarModDataset(in_dist_data[val_idx], in_dist_labels[val_idx]),
        HisarModDataset(in_dist_data[test_idx], in_dist_labels[test_idx]),
        HisarModDataset(ood_data, ood_labels)
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
            torch.save(model.state_dict(), "hisarmod_ghosted_best_model.pth")
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
        self.validation_features = None
        self.validation_labels = None
    def cache_validation_features(self, val_loader, num_samples=2000):
        print(f"🔄 Caching {num_samples} validation features...")
        features, labels, count = [], [], 0
        for x, y in val_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad(): feat = self.feature_extractor(x)
            for i in range(feat.size(0)):
                if count >= num_samples: break
                features.append(feat[i]); labels.append(y[i]); count += 1
            if count >= num_samples: break
        self.validation_features = torch.stack(features)
        self.validation_labels = torch.stack(labels)
        print(f"✅ Cached {len(self.validation_features)} validation features")
    def extract_features(self, loader, maxsamples=2000):
        features, labels, count = [], [], 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad(): feat = self.feature_extractor(x)
            for i in range(feat.size(0)):
                if count >= maxsamples: break
                features.append(feat[i]); labels.append(y[i]); count += 1
            if count >= maxsamples: break
        return torch.stack(features), torch.stack(labels)
    def make_feature_dataset(self, features, labels):
        class FeatureDataset(Dataset):
            def __init__(self, feat, lab): self.data = feat.cpu(); self.targets = lab.cpu()
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
            src_dim=(128,),
            tgt_dim=(128,),
            p=2,
            device=self.device
        )
        dist = DatasetDistance(l1, l2, inner_ot_method='exact', feature_cost=cost, device=self.device)
        return float(dist.distance(maxsamples=min(len(features1), len(features2))))
    def calibrate(self, train_loader, val_loader, ood_loader, maxsamples=2000):
        print("Calibrating OOD detection threshold...")
        self.cache_validation_features(val_loader, num_samples=maxsamples)
        print("🔄 Extracting training features..."); train_feat, train_lab = self.extract_features(train_loader, maxsamples)
        print("🔄 Extracting OOD features..."); ood_feat, ood_lab = self.extract_features(ood_loader, maxsamples)
        print("📊 Computing in-distribution distance...")
        d_in = self.compute_feature_distance(
            train_feat, train_lab, 
            self.validation_features[:maxsamples], self.validation_labels[:maxsamples]
        )
        print("📊 Computing OOD distance...")
        d_ood = self.compute_feature_distance(
            self.validation_features[:maxsamples], self.validation_labels[:maxsamples],
            ood_feat, ood_lab
        )
        if d_in != float('inf') and d_ood != float('inf'):
            self.threshold = d_in + 0.3 * (d_ood - d_in)
        else:
            print("Running some randon ahh threshold cause yours is invalid")
            self.threshold = 50.0
        print(f"📊 Distance Analysis:\n   In-dist (train-val): {d_in:.4f}\n   OOD (val-ood): {d_ood:.4f}\n   Threshold: {self.threshold:.4f}\n   Separation: {d_ood-d_in:.4f}")
        return self.threshold, d_in, d_ood
    def detect(self, test_loader, maxsamples=200):
        if self.validation_features is None:
            raise ValueError("Must call calibrate() first to cache validation features")
        test_feat, test_lab = self.extract_features(test_loader, maxsamples)
        val_samples = min(maxsamples, len(self.validation_features))
        distance = self.compute_feature_distance(
            self.validation_features[:val_samples], self.validation_labels[:val_samples],
            test_feat, test_lab
        )
        return distance > self.threshold, distance

def evaluate_ood_detection_full_dataset(detector, test_loader, ood_loader, batch_size=100):
    print("\n=== 📈 Full Dataset OOD Detection Evaluation ===")
    test_dataset = test_loader.dataset
    ood_dataset = ood_loader.dataset
    max_test_batches = int(np.ceil(len(test_dataset) / batch_size))
    max_ood_batches = int(np.ceil(len(ood_dataset) / batch_size))
    in_dist_distances, ood_distances = [], []
    correct_id, total_id = 0, 0
    correct_ood, total_ood = 0, 0

    print("\n🔍 Processing In-Distribution Test Set...")
    for i in range(max_test_batches):
        indices = list(range(i*batch_size, min((i+1)*batch_size, len(test_dataset))))
        subset = torch.utils.data.Subset(test_dataset, indices)
        subset_loader = DataLoader(subset, batch_size=len(indices), shuffle=False)
        is_ood, distance = detector.detect(subset_loader, maxsamples=len(indices))
        in_dist_distances.append(distance)
        if not is_ood: correct_id += 1
        total_id += 1
        if (i + 1) % 100 == 0 or i == max_test_batches - 1:
            print(f"   [ID] Processed {i+1}/{max_test_batches} batches, Acc: {correct_id/total_id:.4f} (D={distance:.2f})")

    print("\n🚨 Processing OOD Test Set (analog only, unless you extend OOD classes)...")
    for i in range(max_ood_batches):
        indices = list(range(i*batch_size, min((i+1)*batch_size, len(ood_dataset))))
        subset = torch.utils.data.Subset(ood_dataset, indices)
        subset_loader = DataLoader(subset, batch_size=len(indices), shuffle=False)
        is_ood, distance = detector.detect(subset_loader, maxsamples=len(indices))
        ood_distances.append(distance)
        if is_ood: correct_ood += 1
        total_ood += 1
        if (i + 1) % 100 == 0 or i == max_ood_batches - 1:
            print(f"   [OOD] Processed {i+1}/{max_ood_batches} batches, DetRate: {correct_ood/total_ood:.4f} (D={distance:.2f})")

    in_dist_distances = np.array(in_dist_distances)
    ood_distances = np.array(ood_distances)

    true_positive_rate = correct_ood / total_ood
    false_positive_rate = (total_id - correct_id) / total_id
    true_negative_rate = correct_id / total_id
    false_negative_rate = (total_ood - correct_ood) / total_ood
    overall_accuracy = (correct_ood + correct_id) / (total_ood + total_id)
    print(f"\n📊 Final OOD Detection Performance:")
    print(f"   TPR (OOD Det): {true_positive_rate:.4f}")
    print(f"   FPR (ID->OOD): {false_positive_rate:.4f}")
    print(f"   TNR (ID Det): {true_negative_rate:.4f}")
    print(f"   FNR (OOD->ID): {false_negative_rate:.4f}")
    print(f"   Accuracy: {overall_accuracy:.4f}")
    print(f"   In-Dist D: {in_dist_distances.mean():.4f} ± {in_dist_distances.std():.4f}")
    print(f"   OOD D: {ood_distances.mean():.4f} ± {ood_distances.std():.4f}\n   Threshold: {detector.threshold:.4f}\n   Separation: {ood_distances.mean() - in_dist_distances.mean():.4f}")
    return {'tpr': true_positive_rate, 'fpr': false_positive_rate, 'tnr': true_negative_rate, 'fnr': false_negative_rate, 'accuracy': overall_accuracy}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_ds, val_ds, test_ds, ood_ds = prepare_ghosted_datasets()
    train_loader = DataLoader(train_ds, 64, shuffle=True)
    val_loader = DataLoader(val_ds, 64)
    test_loader = DataLoader(test_ds, 64)
    ood_loader = DataLoader(ood_ds, 64)

    model = HisarModCNN(num_classes=4)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    print("=== Training Phase ===")
    train_with_progress(model, train_loader, val_loader, criterion, optimizer, device)
    model.load_state_dict(torch.load("hisarmod_ghosted_best_model.pth", map_location=device))
    model.eval().to(device)
    print("\n=== In-Distribution Testing ===")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Accuracy (In-Distribution): {test_acc:.4f}")
    print("\n=== OOD Detection Setup ===")
    detector = OTDDOODDetector(model, device)
    threshold, d_in, d_ood = detector.calibrate(train_loader, val_loader, ood_loader, maxsamples=2000)
    metrics = evaluate_ood_detection_full_dataset(detector, test_loader, ood_loader, batch_size=100)

if __name__ == '__main__':
    main()
