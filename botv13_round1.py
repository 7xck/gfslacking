import json
from typing import List
import io
import pandas as pd
import warnings
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Trade, TradingState
from typing import Any

pd.set_option('mode.chained_assignment', None)
warnings.simplefilter(action='ignore', category=UserWarning)

Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."


logger = Logger()


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
        conversions = 0
        result = {}

        try:
            data = io.StringIO(state.traderData)
            self.df = pd.read_csv(data, )
            self.df.columns = ["product", "timestamp", "price_deviation_bid", "price_deviation_ask"]
        except:
            self.df = pd.DataFrame(columns=["product", "timestamp", "price_deviation_bid", "price_deviation_ask"])

        logger.print("Market trades: " + str(state.market_trades))
        logger.print("\n Positions: \n" + str(state.position))

        starfruit_position = state.position.get("STARFRUIT", 0)
        amethysts_position = state.position.get("AMETHYSTS", 0)

        self.starfruit_position = starfruit_position

        # Orders to be placed on exchange matching engine

        for product in state.order_depths:
            avg_bid_deviation = self.df[self.df["product"] == str(product)]["price_deviation_bid"].mean()
            avg_ask_deviation = self.df[self.df["product"] == str(product)]["price_deviation_ask"].mean()
            logger.print("\n Average Bid Deviation: " + str(avg_bid_deviation), "\n", "Average Ask Deviation: " + str(avg_ask_deviation), "\n")
            order_depth: OrderDepth = state.order_depths[product]
            self.product_str = str(product)
            self.orderbook = order_depth
            best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
            best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
            # Initialize the list of Orders to be sent as an empty list
            orders: List[Order] = []
            # Define a fair value for the PRODUCT. Might be different for each tradable item
            acceptable_price = self.calculate_convictions_naive(order_depth)
            logger.print("\n calculated fair value : " + str(acceptable_price), "\n")
            price_deviation_percentage_bid = best_bid / acceptable_price - 1
            price_deviation_percentage_ask = best_ask / acceptable_price - 1

            if str(product) == "AMETHYSTS":  # if we're dealing with amethysts
                if acceptable_price == "No Fair Value":
                    continue
                base_order_size = 5  # This is your base order size, adjust as necessary

                # if the book is quoting a price that is too high on both sides,
                # we will try to sell all of our position
                if best_ask > acceptable_price and best_bid > acceptable_price:
                    if amethysts_position > 0:
                        orders.append(Order(product, best_bid, -amethysts_position))
                    else:
                        # we will sell a bit more
                        remaining_size = self.get_remaining_position_limit()
                        if remaining_size >= base_order_size * 2:
                            orders.append(Order(product, best_bid, -base_order_size * 2))
                        elif remaining_size >= base_order_size:
                            orders.append(Order(product, best_bid, -base_order_size))
                        elif remaining_size > 0:
                            orders.append(Order(product, best_bid, -remaining_size))

                # if the book is quoting a price that is too low on both sides,
                # we will try to buy all of our position
                elif best_ask < acceptable_price and best_bid < acceptable_price:
                    if amethysts_position < 0:
                        orders.append(Order(product, best_ask, -amethysts_position))
                    else:
                        # we will buy a bit more
                        remaining_size = self.get_remaining_position_limit()
                        if remaining_size >= base_order_size * 2:
                            orders.append(Order(product, best_ask, base_order_size * 2))
                        elif remaining_size >= base_order_size:
                            orders.append(Order(product, best_ask, base_order_size))
                        elif remaining_size > 0:
                            orders.append(Order(product, best_ask, remaining_size))

                # we know from analysis that the price of starfruit is generally
                # 0.05% away from my calculated fair value ( which is the VWAP )
                # so we will make the book accordingly with order size 1
                else:
                    # Calculate the order size based on the deviation from the fair value
                    acceptable_ask_deviation = 0.00035
                    acceptable_bid_deviation = 0.00035
                    bid_size = base_order_size
                    ask_size = base_order_size
                    # if position size is close to the limit, we probably want to be more aggressive in unloading
                    # that inventory
                    if starfruit_position <= -0.75 * self.LIMITS["AMETHYSTS"]:
                        acceptable_bid_deviation = 0.0002
                        bid_size = base_order_size * 2
                        ask_size = self.LIMITS["AMETHYSTS"] + amethysts_position
                    elif starfruit_position >= 0.75 * self.LIMITS["AMETHYSTS"]:
                        acceptable_ask_deviation = 0.0002
                        ask_size = base_order_size * 2
                        bid_size = self.LIMITS["AMETHYSTS"] - amethysts_position

                    if price_deviation_percentage_bid < -acceptable_bid_deviation:
                        # work out what the bid price should be to make the price deviation 0.05%
                        acceptable_bid = round(
                            acceptable_price - (acceptable_price * (acceptable_bid_deviation - 0.00005)))
                        orders.append(Order(product, acceptable_bid, bid_size))

                    if price_deviation_percentage_ask > acceptable_ask_deviation:
                        # work out what the ask price should be to make the price deviation 0.05%
                        acceptable_ask = round(
                            acceptable_price + (acceptable_price * (acceptable_ask_deviation - 0.00005)))
                        orders.append(Order(product, acceptable_ask, -ask_size))

            elif str(product) == "STARFRUIT":  # if we're dealing with starfruits
                if acceptable_price == "No Fair Value":
                    continue
                base_order_size = 5  # This is your base order size, adjust as necessary

                # if the book is quoting a price that is too high on both sides, 
                # we will try to sell all of our position
                if best_ask > acceptable_price and best_bid > acceptable_price:
                    if starfruit_position > 0:
                        orders.append(Order(product, best_bid, -starfruit_position))
                    else:
                        # we will sell a bit more
                        remaining_size = self.get_remaining_position_limit()
                        if remaining_size >= base_order_size * 2:
                            orders.append(Order(product, best_bid, -base_order_size * 2))
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
                        if remaining_size >= base_order_size * 2:
                            orders.append(Order(product, best_ask, base_order_size * 2))
                        elif remaining_size >= base_order_size:
                            orders.append(Order(product, best_ask, base_order_size))
                        elif remaining_size > 0:
                            orders.append(Order(product, best_ask, remaining_size))

                # we know from analysis that the price of starfruit is generally
                # 0.05% away from my calculated fair value ( which is the VWAP )
                # so we will make the book accordingly with order size 1
                else:
                    # Calculate the order size based on the deviation from the fair value
                    acceptable_ask_deviation = 0.0006
                    acceptable_bid_deviation = 0.0006
                    bid_size = base_order_size
                    ask_size = base_order_size
                    # if position size is close to the limit, we probably want to be more aggressive in unloading
                    # that inventory
                    if starfruit_position < -0.75 * self.LIMITS["STARFRUIT"]:
                        acceptable_bid_deviation = 0.0003
                        bid_size = base_order_size * 2
                        ask_size = self.LIMITS["STARFRUIT"] + starfruit_position
                    elif starfruit_position > 0.75 * self.LIMITS["STARFRUIT"]:
                        acceptable_ask_deviation = 0.0003
                        ask_size = base_order_size * 2
                        bid_size = self.LIMITS["STARFRUIT"] - starfruit_position

                    price_deviation_percentage_bid = best_bid / acceptable_price - 1
                    if price_deviation_percentage_bid < -acceptable_bid_deviation:
                        # work out what the bid price should be to make the price deviation 0.05%
                        acceptable_bid = round(
                            acceptable_price - (acceptable_price * (acceptable_bid_deviation - 0.0001)))
                        orders.append(Order(product, acceptable_bid, bid_size))

                    price_deviation_percentage_ask = best_ask / acceptable_price - 1
                    if price_deviation_percentage_ask > acceptable_ask_deviation:
                        # work out what the ask price should be to make the price deviation 0.05%
                        acceptable_ask = round(
                            acceptable_price + (acceptable_price * (acceptable_ask_deviation - 0.0001)))
                        orders.append(Order(product, acceptable_ask, -ask_size))

            else:
                # well this should just never happen
                price_deviation_percentage_bid = 0
                price_deviation_percentage_ask = 0
                continue

            # what ever product it was, we want to append the orders to the result
            result[product] = orders

            # let's use traderData as a historical store of the book, so we can calculate some
            # indicators
            state.traderData = (
                    state.traderData
                    + f"{str(product)},{state.timestamp},{price_deviation_percentage_bid},{price_deviation_percentage_ask}\n"
            )

        # Sample conversion request. Check more details below.

        logger.flush(state, result, conversions, state.traderData)
        return result, conversions, state.traderData

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
