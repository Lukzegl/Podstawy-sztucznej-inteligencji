from Class import *
import os

def get_Output_Files():
    dir_list = os.listdir("Output/")
    print("Files in 'Output/' directory:")
    for filename in dir_list:
        print(f"  {filename}")
    
    return dir_list

def hand_dict_to_codes(hand_dict):
    suit_codes = {'Spades': 'S', 'Hearts': 'H', 'Diamonds': 'D', 'Clubs': 'C'}
    codes = []
    for suit, cards in hand_dict.items():
        suit_letter = suit_codes[suit]
        for card in cards:
            codes.append(card + suit_letter)
    return codes

def save_to_csv(board_obj: board, output_path: str):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        f.write("Dealer,Vulnerability,North,South,East,West,Bidding\n")
    

    with open(output_path, 'a') as f:
        
        dealer = board_obj.basic_data.dealer
        vulnerability = board_obj.basic_data.vulnerability
        north = hand_dict_to_codes(board_obj.deal.North.cards)
        south = hand_dict_to_codes(board_obj.deal.South.cards)
        east = hand_dict_to_codes(board_obj.deal.East.cards)
        west = hand_dict_to_codes(board_obj.deal.West.cards)
        bidding_str = ';'.join([f"{b.player}:{b.bid}:{b.explanation}" for b in board_obj.bidding_sequence])
        
        f.write(f"{dealer},{vulnerability},\"{','.join(north)}\",\"{','.join(south)}\",\"{','.join(east)}\",\"{','.join(west)}\",\"{bidding_str}\"\n")