import json
from typing import Dict, List
from json import JSONEncoder
import jsonpickle
import io
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
from datamodel import OrderDepth, UserId, TradingState, Order
Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int



def block_diag(*arrs):
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(a) for a in arrs]

    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    if bad_args:
        raise ValueError("arguments in the following positions have dimension "
                         "greater than 2: %s" % bad_args)

    shapes = np.array([a.shape for a in arrs])
    out_dtype = np.result_type(*[arr.dtype for arr in arrs])
    out = np.zeros(np.sum(shapes, axis=0), dtype=out_dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out
# Just collapse all the above in whatever IDE you're in.


class Trader:
    # Notes:
    # run() is the main function. This is what gets called when the bot is uploaded and executed to the platform.
    # Order depth list come already sorted.

    def get_remaining_position_limit(self):
        # doing this for starfruits only, will probs have to do for amethysts later
        # we will use this to calculate the position limit
        # we will use the inventory to calculate the position limit
        remaining = self.LIMITS["STARFRUIT"] - abs(self.starfruit_position)
        return remaining

    def run(self, state: TradingState):
        self.LIMITS = {"AMETHYSTS": 20, "STARFRUIT": 20}

        try:
            data = io.StringIO(state.traderData)
            self.df = pd.read_csv(data, )
            self.df.columns = ["product", "price", "quantity", "timestamp"]
        except:
            self.df = pd.DataFrame(columns=["product", "price", "quantity", "timestamp"])

        #print("Dataframe entries", self.df.tail(1))

        print("Market trades: " + str(state.market_trades))
        print("\n Positions: \n" + str(state.position))
        
        starfruit_position = state.position.get("STARFRUIT", 0)
        amethysts_position = state.position.get("AMETHYSTS", 0)

        self.starfruit_position = starfruit_position

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            self.product_str = str(product)
            self.orderbook = order_depth
            best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
            best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
            # Initialize the list of Orders to be sent as an empty list
            orders: List[Order] = []
            # Define a fair value for the PRODUCT. Might be different for each tradable item
            acceptable_price = self.calculate_convictions_naive(order_depth)
            print("\n calculated fair value : " + str(acceptable_price), "\n")

            if str(product) == "AMETHYSTS": # if we're dealing with amethysts
                # amethysts are stable, we won't make a market on them
                # rather we will look to hedge our positions
                # 1. let's get our exposure to STARFRUIT
                if state.position == {}:  # if we have no position
                    continue
                if starfruit_position == 0:
                    continue

                # Calculate the hedge for STARFRUIT
                beta = 0.1207 # ... i calced this from linear regression using the data in large_price_history, usually you would calc it from a rolling window
                desired_hedge_position = round(beta * starfruit_position)
                # for the first round we will hope the products behave the same way
                # in the next round we will take the new data we're given for these products and then start
                # calculating beta on the fly so there's no future leak
                # and the hedge is more responsive.
                # Rationale for hedges: Looking at the data AMETHYSTS are a very stable product, they do not really move much
                # but it's likely that the price of amethysts are related to starfruit in some way, so we will hedge our position in starfruit
                # if the price of starfruits changes for some reason, we will be protected by our amethysts position which may follow the same trend
                # in practice: say starfruits go up because there is a shortage, we will lose money on our short position in starfruits
                # but because we can see from the data that amethysts generally move the same way as starfruits in the short term, we will make money on our long position in amethysts
                # thereby reducing the loss we take on our short position in starfruits.
                
                if starfruit_position > 0: 
                    # we are long starfruits so we want to short amethysts
                    desired_hedge_position = -desired_hedge_position
                if starfruit_position < 0:
                    # we are short starfruits so we want to long amethysts
                    desired_hedge_position = -desired_hedge_position
                # Calculate the difference between the desired hedge and current AMETHYSTS position
                hedge_difference = desired_hedge_position - amethysts_position
                
                print("Hedge: ", desired_hedge_position)
                print("Hedge difference: ", hedge_difference)
                
                if hedge_difference > 0:
                    # If hedge_difference is positive, buy AMETHYSTS to match the hedge
                    print(f"Buying {hedge_difference} AMETHYSTS to match the hedge")
                    orders.append(Order(product, best_ask, hedge_difference))
                elif hedge_difference < 0:
                    # If hedge_difference is negative, sell AMETHYSTS to match the hedge
                    print(f"Selling {hedge_difference} AMETHYSTS to match the hedge")
                    orders.append(Order(product, best_bid, hedge_difference))
                else:
                    # If hedge_difference is 0, no action is needed
                    print("No hedge adjustment needed for AMETHYSTS")

            elif str(product) == "STARFRUIT": # if we're dealing with starfruits
                if acceptable_price == "No Fair Value":
                    continue
                base_order_size = 5 # This is your base order size, adjust as necessary
                # what is the current spread?
                tick_size = 1  
                
                # if the book is quoting a price that is too high on both sides, 
                # we will try to sell all of our position
                if best_ask > acceptable_price and best_bid > acceptable_price:
                    if starfruit_position > 0:
                        orders.append(Order(product, best_bid, -starfruit_position))
                    else:
                        # we will sell a bit more
                        remaining_size = self.get_remaining_position_limit()
                        if remaining_size >= base_order_size*2:
                            orders.append(Order(product, best_bid, -base_order_size*2))
                        elif remaining_size >= base_order_size:
                            orders.append(Order(product, best_bid, -base_order_size))
                        elif remaining_size > 0:
                            orders.append(Order(product, best_bid, -remaining_size))
                
                # if the book is quoting a price that is too low on both sides,
                # we will try to buy all of our position
                elif best_ask < acceptable_price and best_bid < acceptable_price:
                    if starfruit_position < 0:
                        orders.append(Order(product, best_ask, -starfruit_position))
                    else:
                        # we will buy a bit more
                        remaining_size = self.get_remaining_position_limit()
                        if remaining_size >= base_order_size*2:
                            orders.append(Order(product, best_ask, base_order_size*2))
                        elif remaining_size >= base_order_size:
                            orders.append(Order(product, best_ask, base_order_size))
                        elif remaining_size > 0:
                            orders.append(Order(product, best_ask, remaining_size))

                # we know from analysis that the price of starfruits is generally
                # 0.05% away from my calculated fair value ( which is the vwap )
                # so we will make the book accordingly with order size 1
                else:
                    # Calculate the order size based on the deviation from the fair value
                    acceptable_ask_deviation = 0.0005
                    acceptable_bid_deviation = 0.0005
                    bid_size = base_order_size
                    ask_size = base_order_size
                    # if position size is close to the limit, we probably want to be more aggressive in unloading that inventory
                    if starfruit_position < -0.75 * self.LIMITS["STARFRUIT"]:
                        acceptable_bid_deviation = 0.0003
                        bid_size = base_order_size * 2
                    elif starfruit_position > 0.75 * self.LIMITS["STARFRUIT"]:
                        acceptable_ask_deviation = 0.0003
                        ask_size = base_order_size * 2

                    price_deviation_percentage = best_bid / acceptable_price - 1
                    if price_deviation_percentage < -acceptable_bid_deviation:
                        # work out what the bid price should be to make the price deviation 0.05%
                        acceptable_bid = round(acceptable_price - (acceptable_price * (acceptable_bid_deviation-0.0001)))
                        orders.append(Order(product, acceptable_bid, bid_size))

                    price_deviation_percentage = best_ask / acceptable_price - 1
                    if price_deviation_percentage > acceptable_ask_deviation:
                        # work out what the ask price should be to make the price deviation 0.05%
                        acceptable_ask = round(acceptable_price + (acceptable_price * (acceptable_ask_deviation-0.0001)))
                        orders.append(Order(product, acceptable_ask, -ask_size))
                        

            # what ever product is was, we want to append the orders to the result
            result[product] = orders

            # let's use traderData as a historical store of the book so we can calculate some
            # indicators
            market_trades = state.market_trades.get(self.product_str, [])

            if market_trades != []:
                for trade in market_trades:
                    state.traderData = (
                        state.traderData
                        + f"{str(trade.symbol)},{trade.price},{trade.quantity},{trade.timestamp}\n"
                    )
        
        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, state.traderData

    def calculate_order_size(
        self, price_deviation_percentage, base_order_size, scaling_factor=1000
    ):
        """
        Calculate the order size based on price deviation.
        :param price_deviation_percentage: The percentage deviation of price from the acceptable price.
        :param base_order_size: The base size of the order.
        :param scaling_factor: A factor to adjust how aggressively the order size scales with price deviation.
        :return: The adjusted order size.
        """
        # Calculate the scaling multiplier based on price deviation
        scaling_multiplier = 1 + abs(price_deviation_percentage) * scaling_factor

        # Calculate the final order size
        final_order_size = base_order_size * scaling_multiplier

        return int(final_order_size)

    def calculate_convictions_naive(self, orderbook, depth=3):
        # Calculate VWAP for bids, considering only the top 'depth' levels
        total_bid_volume = 0
        total_bid_value = 0
        for bid_price, bid_amount in list(orderbook.buy_orders.items())[:depth]:
            total_bid_volume += bid_amount
            total_bid_value += bid_price * bid_amount
        bid_vwap = total_bid_value / total_bid_volume if total_bid_volume > 0 else 0

        # Calculate VWAP for asks, considering only the top 'depth' levels
        total_ask_volume = 0
        total_ask_value = 0
        for ask_price, ask_amount in list(orderbook.sell_orders.items())[:depth]:
            ask_amount = abs(ask_amount)
            total_ask_volume += ask_amount
            total_ask_value += ask_price * ask_amount
        ask_vwap = total_ask_value / total_ask_volume if total_ask_volume > 0 else 0

        # Calculate fair value as the midpoint between bid VWAP and ask VWAP
        if bid_vwap > 0 and ask_vwap > 0:
            fair_value = (bid_vwap + ask_vwap) / 2
        elif bid_vwap > 0:  # Only bids are present
            fair_value = bid_vwap
        elif ask_vwap > 0:  # Only asks are present
            fair_value = ask_vwap
        else:  # No bids or asks
            fair_value = 0
        return fair_value


