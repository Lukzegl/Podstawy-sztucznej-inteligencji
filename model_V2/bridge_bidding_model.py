import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Parsowanie
#-----------------------------------------------------------------
def parse_hand_pbn(hand_str):
    
    suits = {'S': [], 'H': [], 'D': [], 'C': []}
    current_suit = None
    for char in hand_str:
        if char in 'SHDC':
            current_suit = char
        elif current_suit:
            suits[current_suit].append(char)
    return suits

def calculate_suit_lengths(hand):
    
    return [len(hand['S']), len(hand['H']), len(hand['D']), len(hand['C'])]

RANK_ORDER = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
SUITS = ['S', 'H', 'D', 'C']

def hand_card_vector(hand):
    """Zwraca binarny wektor 52 wymiarów dla kart w ręce."""
    features = []
    for suit in SUITS:
        for rank in RANK_ORDER:
            features.append(1 if rank in hand[suit] else 0)
    return features

def parse_auction(auction_str):
    
    bids = []
    for part in auction_str.split(';'):
        if ':' in part:
            player, bid = part.split(':')
            bids.append((player, bid))
    return bids
#-----------------------------------------------------------------

possible_bids = ['<PAD>', 'P', 'X', 'XX'] + [f"{level}{suit}" for level in range(1, 8) for suit in ['C', 'D', 'H', 'S', 'NT']]
bid_to_idx = {bid: idx for idx, bid in enumerate(possible_bids)}
idx_to_bid = {idx: bid for bid, idx in bid_to_idx.items()}
#-----------------------------------------------------------------
# Dataset
class BridgeBiddingDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.samples = []

        for _, row in self.data.iterrows():
            # Rozdanie
            deal_pbn = row['deal_pbn']
            hands = deal_pbn.split()
            south_hand = parse_hand_pbn(hands[1])  # South

            player_hcp = row['south_hcp']
            suit_lengths = calculate_suit_lengths(south_hand)
            card_presence = hand_card_vector(south_hand)
            hand_features = [player_hcp] + suit_lengths + card_presence

            # Licytacja
            auction = parse_auction(row['auction_sequence'])

            # AAAA
            for i in range(len(auction)):
                player, current_bid = auction[i]
                
                if player == 'S':  #PERSPEKTYWA
                    prev_bids = [bid_to_idx.get(b, bid_to_idx['<PAD>']) for p, b in auction[:i]]
                    next_bid = bid_to_idx.get(current_bid, bid_to_idx['<PAD>'])

                    max_seq_len = 32
                    
                    if len(prev_bids) < max_seq_len:
                        prev_bids_padded = ([0] * (max_seq_len - len(prev_bids))) + prev_bids
                    else:
                        prev_bids_padded = prev_bids[-max_seq_len:]

                    self.samples.append((
                        torch.tensor(hand_features, dtype=torch.float32), 
                        torch.tensor(prev_bids_padded, dtype=torch.long),
                        torch.tensor(next_bid, dtype=torch.long)
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

#-----------------------------------------------------------------
# Helper functions for evaluation and manual tests
RANK_HCP = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}

def calculate_hcp(hand):
    return sum(RANK_HCP.get(rank, 0) for suit in SUITS for rank in hand[suit])

def build_hand_features(hand, hcp=None):
    if hcp is None:
        hcp = calculate_hcp(hand)
    suit_lengths = calculate_suit_lengths(hand)
    card_presence = hand_card_vector(hand)
    return [hcp] + suit_lengths + card_presence

def pad_bid_sequence(prev_bids, max_seq_len=32):
    if len(prev_bids) < max_seq_len:
        return ([0] * (max_seq_len - len(prev_bids))) + prev_bids
    return prev_bids[-max_seq_len:]

def generate_random_south_hand():
    deck = [rank + suit for suit in SUITS for rank in RANK_ORDER]
    selected = random.sample(deck, 13)
    hand = {suit: [] for suit in SUITS}
    for card in selected:
        rank, suit = card[:-1], card[-1]
        hand[suit].append(rank)
    return hand

def random_bid_sequence(max_history=6, max_seq_len=32):
    history_len = random.randint(0, max_history)
    bid_choices = possible_bids[1:]
    prev_bids = [bid_to_idx[random.choice(bid_choices)] for _ in range(history_len)]
    return pad_bid_sequence(prev_bids, max_seq_len)

# Model
class BridgeBiddingModel(nn.Module):
    def __init__(self, num_hand_features, num_bid_classes, embedding_dim=16, hidden_size=64):
        super(BridgeBiddingModel, self).__init__()
        
        # Embedding dla odzywek
        self.bid_embedding = nn.Embedding(num_embeddings=num_bid_classes, embedding_dim=embedding_dim, padding_idx=0)
        
        #  LSTM SEKWENCJA
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        

        self.fc1 = nn.Linear(hidden_size + num_hand_features, 128)
        self.fc2 = nn.Linear(128, num_bid_classes)

    def forward(self, hand_features, bid_sequence):
    
        embedded_bids = self.bid_embedding(bid_sequence)
        
        lstm_out, _ = self.lstm(embedded_bids)
        
        last_lstm_out = lstm_out[:, -1, :] 
        
        # HISTORIA + reka
        combined = torch.cat((last_lstm_out, hand_features), dim=1)
        out = torch.relu(self.fc1(combined))
        out = self.fc2(out)
        return out

# Trening
def train_model():
    dataset = BridgeBiddingDataset('hands_bidding.csv')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # KARY
    all_targets = [sample[2].item() for sample in dataset.samples]
    unique_classes = np.unique(all_targets)
    weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=all_targets)
    
    class_weights = torch.ones(len(possible_bids), dtype=torch.float32)
    for cls_idx, weight in zip(unique_classes, weights):
        class_weights[cls_idx] = weight
    
    #PASS
    class_weights[1] /= 2
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # HCP + d 4 kol+ 52 karty
    model = BridgeBiddingModel(num_hand_features=57, num_bid_classes=len(possible_bids))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(14):
        model.train()
        total_loss = 0
        for hand_features, bid_seq, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(hand_features, bid_seq)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}')

    torch.save(model.state_dict(), 'bridge_model.pth')
    print("Model zapisany jako bridge_model.pth")

    # TEST
    model.eval()
    all_preds = []
    all_targets_test = []
    with torch.no_grad():
        for hand_features, bid_seq, targets in test_loader:
            outputs = model(hand_features, bid_seq)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets_test.extend(targets.cpu().numpy())
            
    print("Accuracy:", accuracy_score(all_targets_test, all_preds))
    
    
    unique_labels_test = np.unique(all_targets_test)
    target_names_filtered = [idx_to_bid[i] for i in unique_labels_test]
    
    print("Classification Report:")
    print(classification_report(all_targets_test, all_preds, labels=unique_labels_test, target_names=target_names_filtered, zero_division=0))


def evaluate_model(model_path='bridge_model.pth', csv_path='hands_bidding.csv', save_plots=True):
    dataset = BridgeBiddingDataset(csv_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = BridgeBiddingModel(num_hand_features=57, num_bid_classes=len(possible_bids))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for hand_features, bid_seq, targets in test_loader:
            outputs = model(hand_features, bid_seq)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    print(f"Evaluation accuracy: {acc:.4f}")

    unique_labels = np.unique(all_targets)
    target_names = [idx_to_bid[i] for i in unique_labels]
    print("Classification Report:")
    print(classification_report(all_targets, all_preds, labels=unique_labels, target_names=target_names, zero_division=0))

    if save_plots:
        if not PLOTTING_AVAILABLE:
            print("matplotlib nie jest dostępny. Zainstaluj matplotlib, aby wygenerować wykresy.")
            return

        cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(possible_bids))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[idx_to_bid[i] for i in range(len(possible_bids))])
        fig, ax = plt.subplots(figsize=(14, 12))
        disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        ax.set_title('Macierz pomyłek dla modelu licytacji')
        plt.tight_layout()
        fig.savefig('confusion_matrix.png', dpi=150)
        plt.close(fig)

        true_counts = np.bincount(all_targets, minlength=len(possible_bids))
        pred_counts = np.bincount(all_preds, minlength=len(possible_bids))
        top_indices = np.argsort(true_counts)[::-1][:20]
        classes = [idx_to_bid[i] for i in top_indices]
        true_values = true_counts[top_indices]
        pred_values = pred_counts[top_indices]

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(classes))
        ax.bar(x - 0.15, true_values, width=0.3, label='Prawdziwe')
        ax.bar(x + 0.15, pred_values, width=0.3, label='Przewidywane')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_ylabel('Ilość próbek')
        ax.set_title('Porównanie dystrybucji klas w zbiorze testowym')
        ax.legend()
        plt.tight_layout()
        fig.savefig('bid_distribution_comparison.png', dpi=150)
        plt.close(fig)

        print('Wykresy zapisane jako: confusion_matrix.png, bid_distribution_comparison.png')


def decode_bid_sequence(sequence):
    return [idx_to_bid[idx] for idx in sequence if idx != 0]


def predict_next_bid(model, hand, bid_sequence=None):
    if bid_sequence is None:
        bid_sequence = [0] * 32
    hand_tensor = torch.tensor([build_hand_features(hand)], dtype=torch.float32)
    seq_tensor = torch.tensor([bid_sequence], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        outputs = model(hand_tensor, seq_tensor)
        pred_idx = torch.argmax(outputs, dim=1).item()
    return idx_to_bid[pred_idx]


def summarize_hand(hand):
    return ' | '.join(f"{s}:{''.join(hand[s]) or '-'}" for s in SUITS)


def run_manual_random_hand_tests(model_path='bridge_model.pth', num_tests=8, use_random_history=True):
    model = BridgeBiddingModel(num_hand_features=57, num_bid_classes=len(possible_bids))
    model.load_state_dict(torch.load(model_path))

    results = []
    for i in range(num_tests):
        hand = generate_random_south_hand()
        bid_sequence = random_bid_sequence() if use_random_history else [0] * 32
        prediction = predict_next_bid(model, hand, bid_sequence)
        history = decode_bid_sequence(bid_sequence)
        hcp = calculate_hcp(hand)
        suit_lengths = calculate_suit_lengths(hand)
        print(f"Test {i+1}: HCP={hcp}, suit lengths={suit_lengths}, history={history or ['<empty>']}")
        print(f"  Ręka: {summarize_hand(hand)}")
        print(f"  Przewidywana licytacja: {prediction}\n")
        results.append(prediction)

    counts = {bid: results.count(bid) for bid in sorted(set(results), key=lambda b: (-results.count(b), b))}
    print('Rozkład przewidywanych licytacji w testach ręcznych:')
    for bid, count in counts.items():
        print(f"  {bid}: {count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Trening i ewaluacja modelu licytacyjnego')
    parser.add_argument('--train', action='store_true', help='Trenuj model na danych')
    parser.add_argument('--evaluate', action='store_true', help='Oceń model i wygeneruj wykresy')
    parser.add_argument('--manual-tests', action='store_true', help='Uruchom ręczne testy na losowych rękach')
    parser.add_argument('--random-tests', type=int, default=8, help='Liczba losowych testów ręcznych')
    parser.add_argument('--no-history', action='store_true', help='Nie używaj historii licytacji podczas ręcznych testów')
    parser.add_argument('--save-plots', action='store_true', help='Zapisz wykresy podczas ewaluacji')
    args = parser.parse_args()

    if args.train:
        train_model()
    if args.evaluate:
        evaluate_model(save_plots=args.save_plots)
    if args.manual_tests:
        run_manual_random_hand_tests(model_path='bridge_model.pth', num_tests=args.random_tests, use_random_history=not args.no_history)
    if not any([args.train, args.evaluate, args.manual_tests]):
        parser.print_help()