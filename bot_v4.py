from typing import List

import json
from typing import Dict, List
from json import JSONEncoder
import jsonpickle
import io
import numpy as np
import pandas as pd

Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int


class Listing:

    def __init__(self, symbol: Symbol, product: Product, denomination: Product):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination


class ConversionObservation:

    def __init__(
        self,
        bidPrice: float,
        askPrice: float,
        transportFees: float,
        exportTariff: float,
        importTariff: float,
        sunlight: float,
        humidity: float,
    ):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sunlight = sunlight
        self.humidity = humidity


class Observation:

    def __init__(
        self,
        plainValueObservations: Dict[Product, ObservationValue],
        conversionObservations: Dict[Product, ConversionObservation],
    ) -> None:
        self.plainValueObservations = plainValueObservations
        self.conversionObservations = conversionObservations

    def __str__(self) -> str:
        return (
            "(plainValueObservations: "
            + jsonpickle.encode(self.plainValueObservations)
            + ", conversionObservations: "
            + jsonpickle.encode(self.conversionObservations)
            + ")"
        )


class Order:

    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return (
            "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"
        )

    def __repr__(self) -> str:
        return (
            "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"
        )


class OrderDepth:

    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class Trade:

    def __init__(
        self,
        symbol: Symbol,
        price: int,
        quantity: int,
        buyer: UserId = None,
        seller: UserId = None,
        timestamp: int = 0,
    ) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

    def __str__(self) -> str:
        return (
            "("
            + self.symbol
            + ", "
            + self.buyer
            + " << "
            + self.seller
            + ", "
            + str(self.price)
            + ", "
            + str(self.quantity)
            + ", "
            + str(self.timestamp)
            + ")"
        )

    def __repr__(self) -> str:
        return (
            "("
            + self.symbol
            + ", "
            + self.buyer
            + " << "
            + self.seller
            + ", "
            + str(self.price)
            + ", "
            + str(self.quantity)
            + ", "
            + str(self.timestamp)
            + ")"
        )


class TradingState(object):

    def __init__(
        self,
        traderData: str,
        timestamp: Time,
        listings: Dict[Symbol, Listing],
        order_depths: Dict[Symbol, OrderDepth],
        own_trades: Dict[Symbol, List[Trade]],
        market_trades: Dict[Symbol, List[Trade]],
        position: Dict[Product, Position],
        observations: Observation,
    ):
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


class ProsperityEncoder(JSONEncoder):

    def default(self, o):
        return o.__dict__


# Just collapse all the above in whatever IDE you're in.

LIMITS = {"AMETHYSTS": 20, "STARFRUIT": 20}


class Trader:

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

    def calculate_convictions(self, orderbook):
        # first go at calculating fair value given the orderbook
        try:
            best_bid, best_bid_amount = list(orderbook.buy_orders.items())[0]
            best_ask, best_ask_amount = list(orderbook.sell_orders.items())[0]
            fair_value = (best_bid * best_ask_amount + best_ask * best_bid_amount) / (
                best_bid_amount + best_ask_amount
            )
            return fair_value
        except ZeroDivisionError:
            return "No Fair Value"

    def prep_data_sym(T, n_imb, dt, n_spread):
        spread = T.ask - T.bid
        ticksize = np.round(min(spread.loc[spread > 0]) * 100) / 100
        T.spread = T.ask - T.bid
        # adds the spread and mid prices
        T["spread"] = np.round((T["ask"] - T["bid"]) / ticksize) * ticksize
        T["mid"] = (T["bid"] + T["ask"]) / 2
        # filter out spreads >= n_spread
        T = T.loc[(T.spread <= n_spread * ticksize) & (T.spread > 0)]
        T["imb"] = T["bs"] / (T["bs"] + T["as"])
        # discretize imbalance into percentiles
        T["imb_bucket"] = pd.qcut(T["imb"], n_imb, labels=False)
        T["next_mid"] = T["mid"].shift(-dt)
        # step ahead state variables
        T["next_spread"] = T["spread"].shift(-dt)
        T["next_time"] = T["time"].shift(-dt)
        T["next_imb_bucket"] = T["imb_bucket"].shift(-dt)
        # step ahead change in price
        T["dM"] = np.round((T["next_mid"] - T["mid"]) / ticksize * 2) * ticksize / 2
        T = T.loc[(T.dM <= ticksize * 1.1) & (T.dM >= -ticksize * 1.1)]
        # symetrize data
        T2 = T.copy(deep=True)
        T2["imb_bucket"] = n_imb - 1 - T2["imb_bucket"]
        T2["next_imb_bucket"] = n_imb - 1 - T2["next_imb_bucket"]
        T2["dM"] = -T2["dM"]
        T2["mid"] = -T2["mid"]
        T3 = pd.concat([T, T2])
        T3.index = pd.RangeIndex(len(T3.index))
        return T3, ticksize

    def run(self, state: TradingState):
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        try:
            data = io.StringIO(state.traderData)
            df = pd.read_csv(data)
            df.columns = ["product", "timestamp", "best_bid", "bs", "best_ask", "as"]
            print(df.tail(2))
        except:
            pass

        print("Observations: " + str(state.observations))
        print("Own trades: " + str(state.own_trades))
        print("\n positions: \n" + str(state.position))

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            print("PRODUCT", product)
            product_str = str(product)
            if str(product) == "AMETHYSTS":
                # amethysts are stable, we won't make a market on them
                # rather we will look to hedge our positions
                continue
            order_depth: OrderDepth = state.order_depths[product]
            # Initialize the list of Orders to be sent as an empty list
            orders: List[Order] = []
            # Define a fair value for the PRODUCT. Might be different for each tradable item
            # Note that this value of 10 is just a dummy value, you should likely change it!
            acceptable_price = self.calculate_convictions(order_depth)
            if acceptable_price == "No Fair Value":
                continue
            print("calculated fair value : " + str(acceptable_price))

            # Order depth list come already sorted.
            # We can simply pick first item to check first item to get best bid or offer
            # Your original trading logic with added order size adjustment
            base_order_size = 1  # This is your base order size, adjust as necessary

            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                price_deviation_percentage = float(best_ask) / acceptable_price - 1
                if price_deviation_percentage < -0.0003:
                    adjusted_order_size = self.calculate_order_size(
                        price_deviation_percentage, base_order_size
                    )
                    print("BUY", str(adjusted_order_size) + "x", best_ask)
                    orders.append(Order(product, best_ask, -adjusted_order_size))

            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                price_deviation_percentage = float(best_bid) / acceptable_price - 1
                if price_deviation_percentage > 0.0003:
                    adjusted_order_size = self.calculate_order_size(
                        price_deviation_percentage,
                        base_order_size,
                    )  # Use negative scaling factor for selling
                    print("SELL", str(adjusted_order_size) + "x", best_bid)
                    orders.append(Order(product, best_bid, -adjusted_order_size))

            result[product] = orders

        # String value holding Trader state data required.
        # It will be delivered as state.traderData on next execution.

        # let's use traderData as a historical store of the book so we can calculate some
        # indicators
        # it has to be a string so we have to do some wild shit
        try:
            traderData = (
                state.traderData
                + f"{product},{state.timestamp},{best_bid},{best_bid_amount},{best_ask},{best_ask_amount}\n"
            )
        except:
            traderData = state.traderData
        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData
