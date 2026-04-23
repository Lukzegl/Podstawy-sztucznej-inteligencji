import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

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
            hand_features = [player_hcp] + suit_lengths

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
    
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    model = BridgeBiddingModel(num_hand_features=5, num_bid_classes=len(possible_bids))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
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
    
    # Formatowanie raportu
    unique_labels_test = np.unique(all_targets_test)
    target_names_filtered = [idx_to_bid[i] for i in unique_labels_test]
    
    print("Classification Report:")
    print(classification_report(all_targets_test, all_preds, labels=unique_labels_test, target_names=target_names_filtered, zero_division=0))

if __name__ == "__main__":
    train_model()