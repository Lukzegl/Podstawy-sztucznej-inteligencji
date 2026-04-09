class basic_data:
    def __init__(self, dealer: str, vulnerability: str):
        self.dealer = dealer
        self.vulnerability = vulnerability

class Hand:
    def __init__(self, cards: dict):
        self.cards = cards

class deal:
    def __init__(self, North: Hand, South: Hand, East: Hand, West: Hand):
        self.North = North
        self.South = South
        self.East = East
        self.West = West

class bidding_sequence:
    def __init__(self, player: str, bid: str, alert: bool, explanation: str):
        self.player = player
        self.bid = bid
        self.alert = alert
        self.explanation = explanation
        
class result:
    def __init__(self, contract: str, declarer: str, tricks_taken: int):
        self.contract = contract
        self.declarer = declarer
        self.tricks = tricks_taken

class board:
    def __init__(self, basic_data: basic_data, deal: deal, bidding_sequence: list, result: result):
        self.basic_data = basic_data
        self.deal = deal
        self.bidding_sequence = bidding_sequence
        self.result = result

    def print(self):
        print("Dealer:", self.basic_data.dealer)
        print("Vulnerability:", self.basic_data.vulnerability)
        print("North:", self.deal.North.cards)
        print("South:", self.deal.South.cards)
        print("East:", self.deal.East.cards)
        print("West:", self.deal.West.cards)
        print("Bidding:")
        for b in self.bidding_sequence:
            print(f"  {b.player}: {b.bid} {b.explanation}")