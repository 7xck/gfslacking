from typing import Dict


Time = int
Symbol = str
Product = str
Position = int

class TradingState(object):
    def __init__(self,
                 traderData: str,
                 timestamp: Time,
                 listings: Dict[Symbol, Listing],
                 order_depths: Dict[Symbol, OrderDepth],
                 own_trades: Dict[Symbol, List[Trade]],
                 market_trades: Dict[Symbol, List[Trade]],
                 position: Dict[Product, Position],
                 observations: Observation):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations
    
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

"""
The most important properties

- own_trades: the trades the algorithm itself has done since the last `TradingState` 
    came in. This property is a dictionary of `Trade` objects with key being a
    product name. The definition of the `Trade` class is provided in the subsections below.
- market_trades: the trades that other market participants have done since the last 
    `TradingState` came in. This property is also a dictionary of `Trade` objects with key 
    being a product name.
- position: the long or short position that the player holds in every tradable product. 
    This property is a dictionary with the product as the key for which the value is a signed integer denoting the position.
- order_depths: all the buy and sell orders per product that other market participants 
    have sent and that the algorithm is able to trade with. This property is a dict where
    the keys are the products and the corresponding values are instances of the `OrderDepth`
    class. This `OrderDepth` class then contains all the buy and sell orders. 
    An overview of the `OrderDepth` class is also provided in the subsections below.

"""