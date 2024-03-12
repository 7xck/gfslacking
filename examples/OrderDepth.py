class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}

"""
All the orders on a single side (buy or sell) are aggregated in a dict, where the keys indicate the price associated with the order, and the corresponding keys indicate the total volume on that price level. For example, if the buy_orders property would look like this for a certain product `{9: 5, 10: 4}` That would mean that there is a total buy order quantity of 5 at the price level of 9, and a total buy order quantity of 4 at a price level of 10. Players should note that in the sell_orders property, the quantities specified will be negative. E.g., `{12: -3, 11: -2}` would mean that the aggregated sell order volume at price level 12 is 3, and 2 at price level 11.

Every price level at which there are buy orders should always be strictly lower than all the levels at which there are sell orders. If not, then there is a potential match between buy and sell orders, and a trade between the bots should have happened.
"""