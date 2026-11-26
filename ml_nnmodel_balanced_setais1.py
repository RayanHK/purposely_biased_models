import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import copy

TRESHOLD = 0.7

#Note: At the end check which feature model relies on most

device = "cuda" if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

scaler = StandardScaler()

df = pd.read_csv("investigation_train_large_checked.csv", sep=',')

# "Accurate + biased" feature set (from feature_search.py / feature_sets.json)
FEATURES = [
    "persoon_leeftijd_bij_onderzoek",
    "pla_historie_ontwikkeling",
    "pla_hist_pla_categorie_doelstelling_16",
    "afspraak_resultaat_ingevuld_uniek",
    "persoonlijke_eigenschappen_dagen_sinds_taaleis",
    "instrument_reden_beeindiging_historie_succesvol",
    "contacten_onderwerp_no_show",
    "ontheffing_dagen_hist_vanwege_uw_medische_omstandigheden",
    "persoonlijke_eigenschappen_dagen_sinds_opvoer",
    "afspraak_signaal_voor_medewerker",
    "relatie_overig_actueel_vorm__kostendeler",
    "contacten_onderwerp_beoordelen_taaleis",
    "afspraak_inspanningsperiode",
    "pla_einde_doelstelling_bereikt",
]
x_pd = df[FEATURES]
y_pd = df.iloc[:, [-1]]

x = torch.tensor(x_pd.to_numpy(), dtype=torch.float32)
y = torch.tensor(y_pd.to_numpy(), dtype=torch.float32)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Gender is not a model input in this set, so keep it aside for the bias report
gender_test = train_test_split(df["persoon_geslacht_vrouw"].to_numpy(), test_size=0.3, random_state=42)[1]

x_test_raw = x_test.clone()

x_train = torch.tensor(scaler.fit_transform(x_train.numpy()), dtype=torch.float32)
x_test = torch.tensor(scaler.transform(x_test.numpy()), dtype=torch.float32)

#dataloaders
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print(f'Training set has {len(train_loader)} instances')
print(f'Validation set has {len(test_loader)} instances')


class testModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.l1 = torch.nn.Linear(x_pd.shape[1], 630)
        self.a1 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(630, 250)
        self.a2 = torch.nn.ReLU()
        self.l3 = nn.Linear(250, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.l3(x)
        return x

model = testModel().to(device)

# Weigh positives more so the model doesn't just predict "not checked" for everyone
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
optimiser = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train_one_epoch():
    model.train()
    current_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimiser.zero_grad()

        outputs = model(inputs)

        if not torch.isfinite(outputs).all():
            print("Invalid model outputs detected")
            print("Output min:", outputs.min().item())
            print("Output max:", outputs.max().item())
            break

        loss = loss_fn(outputs, labels)

        loss.backward()
        optimiser.step()

        current_loss += loss.item()

    return current_loss / len(train_loader)

def test_model():
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_probs, all_labels = [], []

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            predictions = (torch.sigmoid(outputs) >= THRESHOLD).float()

            correct += (predictions == labels).sum().item()
            total += labels.numel()

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    accuracy = correct / total
    average_loss = total_loss / len(test_loader)
    auc = roc_auc_score(torch.cat(all_labels).numpy().ravel(), torch.cat(all_probs).numpy().ravel())

    return average_loss, accuracy, auc

epochs = 20

train_again = True
MODEL_PATH = "accurate_biased_model.pt"

if os.path.exists(MODEL_PATH) and not train_again:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Loaded saved model, skipping training")
else:
    best_auc = 0.0
    best_weights = None
    epochs_without_improvement = 0
    patience = 2

    for epoch in range(epochs):
        train_loss = train_one_epoch()
        test_loss, accuracy, auc = test_model()
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Accuracy: {accuracy:.2%} | "
            f"AUC: {auc:.3f}"
        )

        if auc > best_auc:
            best_auc = auc
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"AUC did not improve for {patience} epochs, stopping")
                break

    model.load_state_dict(best_weights)
    print(f"Using best weights (AUC {best_auc:.3f})")
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved")


model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(x_test.to(device))).cpu().numpy().ravel()

y_true = y_test.numpy().ravel()
print(f"\nFinal test accuracy: {((probs >= TRESHOLD) == y_true).mean():.2%}")
print(f"Final test AUC:      {roc_auc_score(y_true, probs):.3f}")

df_test = pd.DataFrame(x_test_raw.numpy(), columns=x_pd.columns)
df_test["persoon_geslacht_vrouw"] = gender_test
df_test["pred"] = (probs >= THRESHOLD).astype(int)
df_test["actual"] = y_true
df_test["age_group"] = pd.cut(df_test["persoon_leeftijd_bij_onderzoek"], [0, 30, 45, 60, 200], labels=["≤30", "31-45", "46-60", "61+"])

def bias_report(col, title, names=None):
    overall = df_test["pred"].mean()
    g = df_test.groupby(col, observed=True).agg(
        n=("pred", "size"), flagged=("pred", "mean"), actual=("actual", "mean"))
    if names:
        g.index = g.index.map(names)

    print(f"\n{title} (overall flag rate: {overall:.1%})")
    for k, r in g.iterrows():
        print(f"  {str(k):<6} flagged {r['flagged']:6.1%} | actual {r['actual']:6.1%} | "
              f"{r['flagged'] / overall:.2f}x vs average | "
              f"model/actual {r['flagged'] / r['actual']:.2f} | n={int(r['n'])}")

    if g["flagged"].min() > 0:
        hi, lo = g["flagged"].idxmax(), g["flagged"].idxmin()
        print(f"  -> {hi} is {g.loc[hi, 'flagged'] / g.loc[lo, 'flagged']:.2f}x more likely to be flagged than {lo}")
    else:
        print(f"  -> some groups are never flagged: {list(g.index[g['flagged'] == 0])}")

bias_report("persoon_geslacht_vrouw", "Gender bias", {0: "man", 1: "vrouw"})
bias_report("age_group", "Age bias")