import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import os
from datetime import datetime

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available, using CPU.")
device = torch.device('cuda')


# ==== Probes ====
class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class NonlinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class AdaptiveSoftmaxProbe(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        cutoffs = [min(1000, n_classes - 1)]
        if n_classes > 10000:
            cutoffs.append(min(10000, n_classes - 2))
        self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=input_dim,
            n_classes=n_classes,
            cutoffs=cutoffs,
            div_value=4.0
        )

    def forward(self, x, target=None):
        if target is not None:
            return self.adaptive_softmax(x, target)
        else:
            return self.adaptive_softmax.log_prob(x)

class RandomPredictionProbe(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x):
        return torch.randint(0, self.n_classes, (x.size(0),), device=x.device)

# ==== Training ====
def train_probe(model, X, y, num_classes, task_name="TASK", epochs=10):
    model.to(device)  # Move model to device
    X, y = X.to(device), y.to(device)  # Move tensors to device
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)  # Move batch to device
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"{task_name} - Epoch {epoch+1}, Loss: {total_loss:.4f}")
    return model

def train_linear_probe(X, y, num_classes, task_name="TASK", epochs=10):
    return train_probe(LinearProbe(X.shape[1], num_classes), X, y, num_classes, task_name, epochs)

def train_adaptive_probe(X, y, num_classes, task_name="TASK", epochs=10):
    model = AdaptiveSoftmaxProbe(X.shape[1], num_classes)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb, yb)
            loss = output.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"{task_name} - Epoch {epoch+1}, Loss: {total_loss:.4f}")
    return model

# ==== Evaluation ====
def evaluate_probe(model, X, y):
    model.to(device)  # Move model to device
    X, y = X.to(device), y.to(device)  # Move tensors to device
    model.eval()
    with torch.no_grad():
        if isinstance(model, RandomPredictionProbe):
            preds = model(X)
        else:
            preds = model(X).argmax(dim=1)
        accuracy = (preds == y).float().mean().item()
    return accuracy

def evaluate_adaptive_probe(model, X, y):
    model.eval()
    with torch.no_grad():
        log_probs = model(X).cpu()
        preds = torch.argmax(log_probs, dim=1)
        accuracy = (preds == y.cpu()).float().mean().item()
    return accuracy


def save_model(model, trained_probes_dir, filename):
    """Saves a model to a pickle file."""
    pickle_path = os.path.join(trained_probes_dir, filename)
    try:
        pickle.dump(model, open(pickle_path, "wb"))
        print(f"Model saved to {filename}")
    except Exception as e:
        print(f"Error saving model {filename}: {e}")

# ==== Master Runner ====
def run_all_probes_and_controls(X, y_pos, y_dep, y_position, y_word, le_pos, le_dep, le_word):
    results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    trained_probes_dir = f'trained_models/{timestamp}'
    os.makedirs(trained_probes_dir, exist_ok=True)  # Create directory if it doesn't exist

    # === Main Probes ===
    # pos_model = train_linear_probe(X, y_pos, len(le_pos.classes_), task_name="POS")
    # save_model(pos_model, trained_probes_dir, "pos_model.pkl")

    # dep_model = train_linear_probe(X, y_dep, len(le_dep.classes_), task_name="DEP")
    # save_model(dep_model, trained_probes_dir, "dep_model.pkl")

    # position_model = train_linear_probe(X, y_position, y_position.max().item() + 1, task_name="POSITION")
    # save_model(position_model, trained_probes_dir, "position_model.pkl")

    word_model = train_linear_probe(X, y_word, len(le_word.classes_), task_name="WORD")
    save_model(word_model, trained_probes_dir, "word_model.pkl")

    # # === Nonlinear ===
    # pos_nonlinear = train_probe(NonlinearProbe(X.shape[1], len(le_pos.classes_)), X, y_pos, len(le_pos.classes_), task_name="POS_Nonlinear")
    # save_model(pos_nonlinear, trained_probes_dir, "pos_nonlinear.pkl")

    # dep_nonlinear = train_probe(NonlinearProbe(X.shape[1], len(le_dep.classes_)), X, y_dep, len(le_dep.classes_), task_name="DEP_Nonlinear")
    # save_model(dep_nonlinear, trained_probes_dir, "dep_nonlinear.pkl")

    # position_nonlinear = train_probe(NonlinearProbe(X.shape[1], y_position.max().item() + 1), X, y_position, y_position.max().item() + 1, task_name="POSITION_Nonlinear")
    # save_model(position_nonlinear, trained_probes_dir, "position_nonlinear.pkl")

    # # === Random Baselines ===
    # pos_random = RandomPredictionProbe(len(le_pos.classes_))
    # save_model(pos_random, trained_probes_dir, "pos_random.pkl")

    # dep_random = RandomPredictionProbe(len(le_dep.classes_))
    # save_model(dep_random, trained_probes_dir, "dep_random.pkl")

    # position_random = RandomPredictionProbe(y_position.max().item() + 1)
    # save_model(position_random, trained_probes_dir, "position_random.pkl")

    # word_random = RandomPredictionProbe(len(le_word.classes_))
    # save_model(word_random, trained_probes_dir, "word_random.pkl")

    # # === Shuffled Labels ===
    # y_pos_shuffled = y_pos[torch.randperm(len(y_pos))]
    # y_dep_shuffled = y_dep[torch.randperm(len(y_dep))]
    # y_position_shuffled = y_position[torch.randperm(len(y_position))]
    # y_word_shuffled = y_word[torch.randperm(len(y_word))]

    # pos_shuffled = train_linear_probe(X, y_pos_shuffled, len(le_pos.classes_), task_name="POS_Shuffled")
    # save_model(pos_shuffled, trained_probes_dir, "pos_shuffled.pkl")

    # dep_shuffled = train_linear_probe(X, y_dep_shuffled, len(le_dep.classes_), task_name="DEP_Shuffled")
    # save_model(dep_shuffled, trained_probes_dir, "dep_shuffled.pkl")

    # position_shuffled = train_linear_probe(X, y_position_shuffled, y_position.max().item() + 1, task_name="POSITION_Shuffled")
    # save_model(position_shuffled, trained_probes_dir, "position_shuffled.pkl")

    # word_shuffled_model = train_adaptive_probe(X, y_word_shuffled, len(le_word.classes_), task_name="WORD_Shuffled")
    # save_model(word_shuffled_model, trained_probes_dir, "word_shuffled_model.pkl")

    # # === Random Representations ===
    # X_random = torch.randn_like(X)
    # pos_randrep = train_linear_probe(X_random, y_pos, len(le_pos.classes_), task_name="POS_RandomRep")
    # save_model(pos_randrep, trained_probes_dir, "pos_randrep.pkl")

    # dep_randrep = train_linear_probe(X_random, y_dep, len(le_dep.classes_), task_name="DEP_RandomRep")
    # save_model(dep_randrep, trained_probes_dir, "dep_randrep.pkl")

    # position_randrep = train_linear_probe(X_random, y_position, y_position.max().item() + 1, task_name="POSITION_RandomRep")
    # save_model(position_randrep, trained_probes_dir, "position_randrep.pkl")

    # word_randrep_model = train_adaptive_probe(X_random, y_word, len(le_word.classes_), task_name="WORD_RandomRep")
    # save_model(word_randrep_model, trained_probes_dir, "word_randrep_model.pkl")

    # # === Evaluation ===
    # print("\n--- Evaluation ---")
    # results.update({
    #     "POS (Linear)": evaluate_probe(pos_model, X, y_pos),
    #     "POS (Nonlinear)": evaluate_probe(pos_nonlinear, X, y_pos),
    #     "POS (Random)": evaluate_probe(pos_random, X, y_pos),
    #     "POS (Shuffled)": evaluate_probe(pos_shuffled, X, y_pos),
    #     "POS (RandomRep)": evaluate_probe(pos_randrep, X_random, y_pos),

    #     "DEP (Linear)": evaluate_probe(dep_model, X, y_dep),
    #     "DEP (Nonlinear)": evaluate_probe(dep_nonlinear, X, y_dep),
    #     "DEP (Random)": evaluate_probe(dep_random, X, y_dep),
    #     "DEP (Shuffled)": evaluate_probe(dep_shuffled, X, y_dep),
    #     "DEP (RandomRep)": evaluate_probe(dep_randrep, X_random, y_dep),

    #     "POSITION (Linear)": evaluate_probe(position_model, X, y_position),
    #     "POSITION (Nonlinear)": evaluate_probe(position_nonlinear, X, y_position),
    #     "POSITION (Random)": evaluate_probe(position_random, X, y_position),
    #     "POSITION (Shuffled)": evaluate_probe(position_shuffled, X, y_position),
    #     "POSITION (RandomRep)": evaluate_probe(position_randrep, X_random, y_position),

    #     "WORD": evaluate_adaptive_probe(word_model, X, y_word),
    #     "WORD (Shuffled)": evaluate_adaptive_probe(word_shuffled_model, X, y_word),
    #     "WORD (RandomRep)": evaluate_adaptive_probe(word_randrep_model, torch.randn_like(X), y_word),
    #     "WORD (Random)": evaluate_probe(word_random, X, y_word),
    # })

    # for name, acc in results.items():
    #     print(f"{name:30s}: {acc:.2%}")

    return results

if __name__ == "__main__":
    df = pd.read_pickle("./dataset.pkl")
    df['word'] = df['word'].str.lower()
    X = torch.stack(df['embedding'].tolist())

    le_pos = LabelEncoder().fit(df["pos"])
    le_dep = LabelEncoder().fit(df["dep"])
    le_word = LabelEncoder().fit(df["word"])

    y_pos = torch.tensor(le_pos.transform(df['pos'].values))
    y_dep = torch.tensor(le_dep.transform(df['dep'].values))
    y_word = torch.tensor(le_word.transform(df['word'].values))
    y_position = torch.tensor(df['position'].values)

    results = run_all_probes_and_controls(X, y_pos, y_dep, y_position, y_word, le_pos, le_dep, le_word)