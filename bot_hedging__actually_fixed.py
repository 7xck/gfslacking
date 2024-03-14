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

Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int

n_imb = 10
n_spread = 4
dt = 1


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

    def run(self, state: TradingState):
        """
        Outputs a list of orders to be sent
        """
        self.LIMITS = {"AMETHYSTS": 20, "STARFRUIT": 20}

        try:
            data = io.StringIO(state.traderData)
            self.df = pd.read_csv(data, )
            self.df.columns = ["product", "time", "bid", "bs", "ask", "as"]
        except:
            self.df = pd.DataFrame(columns=["product", "time", "bid", "bs", "ask", "as"])
            pass

        self.df["date"] = -2
        self.df['date']=self.df['date'].astype(float)
        self.df['time']=self.df['time'].astype(float)
        self.df['bid']=self.df['bid'].astype(float)
        self.df['ask']=self.df['ask'].astype(float)
        self.df['bs']=self.df['bs'].astype(float)
        self.df['as']=self.df['as'].astype(float)
        self.df['mid']=(self.df['bid'].astype(float)+self.df['ask'].astype(float))/2
        self.df['imb']=self.df['bs'].astype(float)/(self.df['bs'].astype(float)+self.df['as'].astype(float))
        self.df['wmid']=self.df['ask'].astype(float)*self.df['imb']+self.df['bid'].astype(float)*(1-self.df['imb'])

        print("Observations: " + str(state.observations))
        print("Own trades: " + str(state.own_trades))
        print("\n positions: \n" + str(state.position))

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            self.product_str = str(product)
            order_depth: OrderDepth = state.order_depths[product]
            self.orderbook = order_depth
            # Initialize the list of Orders to be sent as an empty list
            orders: List[Order] = []
            # Define a fair value for the PRODUCT. Might be different for each tradable item
            acceptable_price = self.calculate_convictions_naive(order_depth)
            if str(product) == "AMETHYSTS":
                # amethysts are stable, we won't make a market on them
                # rather we will look to hedge our positions
                # implement this logic later ...
                # 1. let's get our exposure to STARFRUIT
                if state.position == {}:  # if we have no position
                    continue
                if state.position.get("STARFRUIT", 0) == 0:
                    continue
                starfruit_position = state.position.get("STARFRUIT", 0)
                amethysts_position = state.position.get("AMETHYSTS", 0)
                
                # Calculate the hedge for STARFRUIT
                beta = 0.1207
                desired_hedge_position = round(beta * starfruit_position)
                
                # Calculate the difference between the desired hedge and current AMETHYSTS position
                hedge_difference = desired_hedge_position - amethysts_position
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                
                print("Hedge: ", desired_hedge_position)
                print("Starfruit position: ", starfruit_position)
                print("Current AMETHYSTS position: ", amethysts_position)
                print("Hedge difference: ", hedge_difference)
                
                if hedge_difference > 0:
                    # If hedge_difference is positive, buy AMETHYSTS to match the hedge
                    print(f"Buying {hedge_difference} AMETHYSTS to match the hedge")
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    orders.append(Order(product, best_ask, hedge_difference))
                elif hedge_difference < 0:
                    # If hedge_difference is negative, sell AMETHYSTS to match the hedge
                    print(f"Selling {-hedge_difference} AMETHYSTS to match the hedge")
                    orders.append(Order(product, best_bid, hedge_difference))
                else:
                    # If hedge_difference is 0, no action is needed
                    print("No hedge adjustment needed for AMETHYSTS")


            if acceptable_price == "No Fair Value":
                continue
            print("\n calculated fair value : " + str(acceptable_price), "\n")

            base_order_size = 1  # This is your base order size, adjust as necessary

            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                price_deviation_percentage = float(best_ask) / acceptable_price - 1
                if price_deviation_percentage < -0.0005:
                    adjusted_order_size = self.calculate_order_size(
                        price_deviation_percentage, base_order_size
                    )
                    print("\n Placing order: BUY", str(adjusted_order_size) + "x", best_ask, self.product_str, "\n")
                    orders.append(Order(product, best_ask, -adjusted_order_size))

            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                price_deviation_percentage = float(best_bid) / acceptable_price - 1
                if price_deviation_percentage > 0.0005:
                    adjusted_order_size = self.calculate_order_size(
                        price_deviation_percentage,
                        base_order_size,
                    )  # Use negative scaling factor for selling
                    print("\n Placing order: SELL", str(adjusted_order_size) + "x", best_bid, self.product_str, "\n")
                    orders.append(Order(product, best_bid, -adjusted_order_size))

            result[product] = orders

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
        
    def plot_Gstar(self, G1,B):
        G2=np.dot(B,G1)+G1
        G3=G2+np.dot(np.dot(B,B),G1)
        G4=G3+np.dot(np.dot(np.dot(B,B),B),G1)
        G5=G4+np.dot(np.dot(np.dot(np.dot(B,B),B),B),G1)
        G6=G5+np.dot(np.dot(np.dot(np.dot(np.dot(B,B),B),B),B),G1)
        return G6

    def estimate(self, T):
        no_move=T[T['dM']==0]
        no_move_counts=no_move.pivot_table(index=[ 'next_imb_bucket'], 
                        columns=['spread', 'imb_bucket'], 
                        values='time',
                        fill_value=0, 
                        aggfunc='count').unstack()
        Q_counts=np.resize(np.array(no_move_counts[0:(n_imb*n_imb)]),(n_imb,n_imb))
        # loop over all spreads and add block matrices
        for i in range(1,n_spread):
            Qi=np.resize(np.array(no_move_counts[(i*n_imb*n_imb):(i+1)*(n_imb*n_imb)]),(n_imb,n_imb))
            Q_counts=block_diag(Q_counts,Qi)
        #print Q_counts
        move_counts=T[(T['dM']!=0)].pivot_table(index=['dM'], 
                            columns=['spread', 'imb_bucket'], 
                            values='time',
                            fill_value=0, 
                            aggfunc='count').unstack()

        R_counts=np.resize(np.array(move_counts),(n_imb*n_spread,4))
        T1=np.concatenate((Q_counts,R_counts),axis=1).astype(float)
        for i in range(0,n_imb*n_spread):
            T1[i]=T1[i]/T1[i].sum()
        Q=T1[:,0:(n_imb*n_spread)]
        R1=T1[:,(n_imb*n_spread):]

        K=np.array([-0.01, -0.005, 0.005, 0.01])
        move_counts=T[(T['dM']!=0)].pivot_table(index=['spread','imb_bucket'], 
                        columns=['next_spread', 'next_imb_bucket'], 
                        values='time',
                        fill_value=0, 
                        aggfunc='count') #.unstack()

        R2_counts=np.resize(np.array(move_counts),(n_imb*n_spread,n_imb*n_spread))
        T2=np.concatenate((Q_counts,R2_counts),axis=1).astype(float)

        for i in range(0,n_imb*n_spread):
            T2[i]=T2[i]/T2[i].sum()
        R2=T2[:,(n_imb*n_spread):]
        Q2=T2[:,0:(n_imb*n_spread)]
        G1=np.dot(np.dot(np.linalg.inv(np.eye(n_imb*n_spread)-Q),R1),K)
        B=np.dot(np.linalg.inv(np.eye(n_imb*n_spread)-Q),R2)
        
        return G1,B,Q,Q2,R1,R2,K
        
    def calculate_convictions_microprice(self):
        print("\n USING CALCULATE CONVICTIONS V2 \n")
        # second go at calculating fair value given the orderbook,
        # let's try to implement Stolky's Microprice?
        imb = np.linspace(0, 1, n_imb)
        data = self.df[self.df["product"] == self.product_str]
        ticker = self.product_str
        T,ticksize=self.prep_data_sym(data,n_imb,dt,n_spread)
        G1,B,Q,Q2,R1,R2,K=self.estimate(T)
        G6=self.plot_Gstar(G1,B)
        print(T.tail(2))
        index = [str(i + 1) for i in range(0, n_spread)]
        G_star = pd.DataFrame(G6.reshape(n_spread, n_imb), index=index, columns=imb)

        bid, best_bid_amount = list(self.orderbook.buy_orders.items())[0]
        ask, best_ask_amount = list(self.orderbook.sell_orders.items())[0]
        mid = (bid + ask) / 2
        imb = best_bid_amount / (best_bid_amount + best_ask_amount)
        imb_bucket = [abs(x - imb) for x in G_star.columns].index(min([abs(x - imb) for x in G_star.columns]))
        spreads = G_star[G_star.columns[imb_bucket]].values
        spread = ask - bid
        spread_bucket = round(spread / ticksize) * ticksize // ticksize - 1
        if spread_bucket >= n_spread:
            spread_bucket = n_spread - 1
            spread_bucket = int(spread_bucket)
            # Compute adjusted midprice
            adj_midprice = mid + spreads[spread_bucket]
        return adj_midprice

    def prep_data_sym(self, T, n_imb, dt, n_spread):
        spread = T['ask'] - T['bid']
        print(spread)
        ticksize = np.round(min(spread.loc[spread > 0]) * 100) / 100
        T.spread = T['ask'] - T['bid']
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
        T = T.loc[(T['dM'] <= ticksize * 1.1) & (T['dM'] >= -ticksize * 1.1)]
        # symetrize data
        T2 = T.copy(deep=True)
        T2["imb_bucket"] = n_imb - 1 - T2["imb_bucket"]
        T2["next_imb_bucket"] = n_imb - 1 - T2["next_imb_bucket"]
        T2["dM"] = -T2["dM"]
        T2["mid"] = -T2["mid"]
        T3 = pd.concat([T, T2])
        T3.index = pd.RangeIndex(len(T3.index))
        return T3, ticksize
    


