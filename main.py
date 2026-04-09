import os
import re
from tocsv import *

from Class import *
def get_Data():
    dir_list = os.listdir("Data/")
    print("Files in 'Data/' directory:")
    for filename in dir_list:
        print(f"  {filename}")
    
    return dir_list

def parse_hands(md_section):
    """Parse the md| section and return the 4 hands."""
    hand_strings = md_section.split(',')
    
    hands = []
    for hand_str in hand_strings:
        if hand_str == "":
            hands.append({})
            continue
        
        hand = parse_single_hand(hand_str)
        hands.append(hand)
    
    return hands

def parse_single_hand(hand_str):
    
    if hand_str and hand_str[0].isdigit():
        hand_str = hand_str[1:]
    
    suits = {'S': 'Spades', 'H': 'Hearts', 'D': 'Diamonds', 'C': 'Clubs'}
    hand_dict = {suit: [] for suit in suits.values()}
    
    current_suit = None
    for char in hand_str:
        if char in suits:
            current_suit = suits[char]
        elif current_suit:
            hand_dict[current_suit].append(char)
    
    return hand_dict

def full_deck():
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['S', 'H', 'D', 'C']
    return [rank + suit for suit in suits for rank in ranks]


def hand_dict_to_codes(hand_dict):
    suit_codes = {'Spades': 'S', 'Hearts': 'H', 'Diamonds': 'D', 'Clubs': 'C'}
    codes = []
    for suit, cards in hand_dict.items():
        suit_letter = suit_codes[suit]
        for card in cards:
            codes.append(card + suit_letter)
    return codes


def codes_to_hand_dict(codes):
    suits = {'S': 'Spades', 'H': 'Hearts', 'D': 'Diamonds', 'C': 'Clubs'}
    hand_dict = {suit: [] for suit in suits.values()}
    for code in codes:
        if len(code) < 2:
            continue
        rank = code[:-1]
        suit_letter = code[-1]
        suit = suits.get(suit_letter)
        if suit:
            hand_dict[suit].append(rank)
    return hand_dict

def extract_data(file_path):
    with open(file_path, 'r') as f:
        content = f.read().strip()
    
    sections = content.split('|')
    

    md_index = sections.index('md') + 1
    md_section = sections[md_index]
    hands = parse_hands(md_section)
    
    dealer_code = md_section[0]
    dealer_map = {'1': 'N', '2': 'E', '3': 'S', '4': 'W'}
    dealer = dealer_map.get(dealer_code, 'N')
    
   
    sv_index = sections.index('sv') + 1
    vuln_code = sections[sv_index]
    vuln_map = {'o': 'None', 'n': 'NS', 'e': 'EW', 'b': 'Both'}
    vulnerability = vuln_map.get(vuln_code, 'None')
    
    basic_data_obj = basic_data(dealer, vulnerability)
    
    player_sequence = {
        'N': ['N', 'E', 'S', 'W'],
        'E': ['E', 'S', 'W', 'N'],
        'S': ['S', 'W', 'N', 'E'],
        'W': ['W', 'N', 'E', 'S'],
    }
    order = player_sequence[dealer]

    seen_codes = set()
    for hand in hands:
        if hand:
            seen_codes.update(hand_dict_to_codes(hand))

    missing_codes = [code for code in full_deck() if code not in seen_codes]
    while len(hands) < 4:
        hands.append({})

    hand_by_player = {}
    for i, player in enumerate(order):
        if hands[i]:
            hand_by_player[player] = Hand(hands[i])
        else:
            hand_by_player[player] = Hand(codes_to_hand_dict(missing_codes))
            missing_codes = []

    north = hand_by_player['N']
    east = hand_by_player['E']
    south = hand_by_player['S']
    west = hand_by_player['W']
    
    deal_obj = deal(north, south, east, west)
    
    # Bidding sequence
    bidding = []
    mb_indices = [i for i, s in enumerate(sections) if s == 'mb']
    an_indices = [i for i, s in enumerate(sections) if s == 'an']
    
    player_order = ['N', 'E', 'S', 'W']
    current_player_index = player_order.index(dealer)
    
    for i, idx in enumerate(mb_indices):
        bid = sections[idx + 1]
        player = player_order[current_player_index % 4]
        current_player_index += 1
        
        alert = False
        explanation = ""
        if i < len(an_indices):
            an_idx = an_indices[i]
            explanation = sections[an_idx + 1]
            if "alert" in explanation.lower():
                alert = True
        
        bidding.append(bidding_sequence(player, bid, alert, explanation))
    
    result_obj = result(None, None, None)
    
    board_obj = board(basic_data_obj, deal_obj, bidding, result_obj)
    return board_obj

def save_to_JSON(board_obj: board, output_path: str):
    import json
    def serialize(obj):
        if isinstance(obj, board):
            return {
                'basic_data': serialize(obj.basic_data),
                'deal': serialize(obj.deal),
                'bidding_sequence': [serialize(b) for b in obj.bidding_sequence],
                'result': serialize(obj.result)
            }
        elif isinstance(obj, basic_data):
            return {
                'dealer': obj.dealer,
                'vulnerability': obj.vulnerability
            }
        elif isinstance(obj, deal):
            return {
                'North': serialize(obj.North),
                'South': serialize(obj.South),
                'East': serialize(obj.East),
                'West': serialize(obj.West)
            }
        elif isinstance(obj, Hand):
            return obj.cards
        elif isinstance(obj, bidding_sequence):
            return {
                'player': obj.player,
                'bid': obj.bid,
                'alert': obj.alert,
                'explanation': obj.explanation
            }
        elif isinstance(obj, result):
            return {
                'contract': obj.contract,
                'declarer': obj.declarer,
                'tricks_taken': obj.tricks
            }
        else:
            raise TypeError(f"Type {type(obj)} not serializable")
    
    with open(output_path, 'w') as f:
        json.dump(serialize(board_obj), f, indent=4)
    
    
   

def main():
    list1 = get_Data()
    file_path = "Data/4637768523.lin"
    board_data = extract_data(file_path)
    board_data.print()

    for filename in list1:
        file_path = os.path.join("Data", filename)
        board_data = extract_data(file_path)
        output_path = os.path.join("Output", filename.replace('.lin', '.json'))
        save_to_JSON(board_data, output_path)

    list2 = get_Output_Files()
    for filename in list2:
        save_to_csv(extract_data(os.path.join("Data", filename.replace('.json', '.lin'))),"csv/data1.csv" )


if __name__ == "__main__":
        main()