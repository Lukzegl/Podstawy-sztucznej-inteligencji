import random
import csv
from collections import Counter
import math

SUITS = ['S', 'H', 'D', 'C']  # piki, kiery, kara, trefle
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

DECK = [r + s for s in SUITS for r in RANKS]


HCP_VALUES = {
    'A': 4,
    'K': 3,
    'Q': 2,
    'J': 1
}

def calculate_hcp(hand):
    return sum(HCP_VALUES.get(card[0], 0) for card in hand)


def suit_lengths(hand):
    cnt = Counter(card[1] for card in hand)
    return [cnt['S'], cnt['H'], cnt['D'], cnt['C']]


def rowny_sklad(d):
    d_sorted = sorted(d, reverse=True)
    return d_sorted in ([4,3,3,3], [4,4,3,2], [5,3,3,2])

def prawo20(d,pc):
  top2 = sorted(d, reverse=True)[:2]
  if pc + top2[0] + top2[1] >= 20:
    return True
  else:
    return False


def co_otworzyc(dlugosc, pc):
    if pc >= 12 or prawo20(dlugosc, pc):
        if pc >= 22:
          return "2C"
        elif pc >= 20 and pc <= 21 and rowny_sklad(dlugosc):
          return "2N"
        elif dlugosc[0] >= 5 and pc < 21:
            return "1S"
        elif dlugosc[1] >= 5 and pc < 21:
            return "1H"
        elif pc >= 15 and pc <= 17 and rowny_sklad(dlugosc):
            return "1NT"
        elif dlugosc[2] >= 4 and pc < 21:
            return "1D"
        else:
            return "1C"
    else:
        return "PASS"


def deal_hand():
    return random.sample(DECK, 13)


def generate_hands(n, filename="hands.csv"):
    possible_openings = ["1S", "1H", "1NT", "1D", "1C", "2C", "2N", "PASS"]
    # Calculate target count per opening, ensuring at least one hand per category if n is small
    target_count_per_opening = math.ceil(n / len(possible_openings))
    current_counts = {op: 0 for op in possible_openings}
    total_generated_hands = 0
    max_attempts_for_single_hand_type = 50000 # Safety break for finding rare hands

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["hand", "S", "H", "D", "C", "HCP", "opening"])

        # First pass: try to fill up each category to target_count_per_opening
        # This ensures a minimum representation for all opening types
        while any(current_counts[op] < target_count_per_opening for op in possible_openings) and total_generated_hands < n:
            attempts = 0
            hand_found = False
            while attempts < max_attempts_for_single_hand_type:
                hand = deal_hand()
                d = suit_lengths(hand)
                pc = calculate_hcp(hand)
                opening = co_otworzyc(d, pc)

                if current_counts[opening] < target_count_per_opening:
                    writer.writerow([
                        " ".join(hand),
                        d[0], d[1], d[2], d[3],
                        pc,
                        opening
                    ])
                    current_counts[opening] += 1
                    total_generated_hands += 1
                    hand_found = True
                    break # Found a suitable hand for an underrepresented category
                attempts += 1
            if not hand_found:
                # If after many attempts we couldn't find a hand for an underrepresented category,
                # it means the remaining target openings might be extremely rare or impossible to achieve.
                print(f"Warning: Could not find hands for all target counts after {max_attempts_for_single_hand_type} attempts for a single hand type. Some categories might be underrepresented.")
                break # Exit the first pass and proceed to fill the rest randomly

        # Second pass: fill up the remaining hands to reach 'n' total
        # These hands will be generated randomly, but the first pass already ensured a base distribution
        while total_generated_hands < n:
            hand = deal_hand()
            d = suit_lengths(hand)
            pc = calculate_hcp(hand)
            opening = co_otworzyc(d, pc)

            writer.writerow([
                " ".join(hand),
                d[0], d[1], d[2], d[3],
                pc,
                opening
            ])
            current_counts[opening] += 1
            total_generated_hands += 1

    print(f"Generated {total_generated_hands} hands.")
    print("Final distribution of openings:")
    for op, count in current_counts.items():
        print(f"{op}: {count}")


if __name__ == "__main__":
    generate_hands(100000)


#-----------------------------------------------------------

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv("hands.csv")

X = df[["S", "H", "D", "C", "HCP"]].values
y = df["opening"].values


le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


class BridgeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, len(le.classes_))
        )

    def forward(self, x):
        return self.net(x)

model = BridgeNet()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 300

for epoch in range(epochs):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval() # Set model to evaluation mode
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
            val_preds = val_outputs.argmax(dim=1)
            val_accuracy = (val_preds == y_test).float().mean()

        print(f"Epoch {epoch}, loss = {loss.item():.4f}, val_loss = {val_loss.item():.4f}, val_accuracy = {val_accuracy:.4f}")

model.eval()
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    accuracy = (preds == y_test).float().mean()

print(f"\nAccuracy: {accuracy:.4f}")


torch.save(model.state_dict(), "bridge_model.pt")

#-----------------------------------------------------------------------

import torch


model = BridgeNet()
model.load_state_dict(torch.load("bridge_model.pt"))
model.eval()

def predict_opening(s, h, d, c, hcp):
    x = torch.tensor([[s, h, d, c, hcp]], dtype=torch.float32)

    with torch.no_grad():
        out = model(x)
        pred = out.argmax(dim=1).item()

    return le.inverse_transform([pred])[0]

print(predict_opening(5, 3, 3, 2, 14))

#------------------------------------------------------------------------

import random
from collections import Counter

SUITS = ['S','H','D','C']
RANKS = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
DECK = [r+s for s in SUITS for r in RANKS]

HCP_VALUES = {'A':4,'K':3,'Q':2,'J':1}

def deal():
    return random.sample(DECK, 13)

def hcp(hand):
    return sum(HCP_VALUES.get(c[0],0) for c in hand)

def suit_count(hand):
    c = Counter(card[1] for card in hand)
    return [c['S'], c['H'], c['D'], c['C']]

# test
for _ in range(20):
  hand = deal()
  s,h,d,c = suit_count(hand)
  points = hcp(hand)

  print("HAND:", hand)
  print("S,H,D,C:", s,h,d,c)
  print("HCP:", points)

  print("MODEL:", predict_opening(s,h,d,c,points))

