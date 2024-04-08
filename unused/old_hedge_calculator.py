# if str(product) == "AMETHYSTS":  # if we're dealing with amethysts
#     # amethysts are stable, we won't make a market on them
#     # rather we will look to hedge our positions
#     # 1. let's get our exposure to STARFRUIT
#     if state.position == {}:  # if we have no position
#         continue
#     if starfruit_position == 0:
#         continue
#
#     # Calculate the hedge for STARFRUIT
#     beta = 0.1207  # ... i calced this from linear regression using the data in large_price_history, usually you would calc it from a rolling window
#     desired_hedge_position = round(beta * starfruit_position)
#     # for the first round we will hope the products behave the same way
#     # in the next round we will take the new data we're given for these products and then start
#     # calculating beta on the fly so there's no future leak
#     # and the hedge is more responsive.
#     # Rationale for hedges: Looking at the data AMETHYSTS are a very stable product, they do not really move much
#     # but it's likely that the price of amethysts are related to starfruit in some way, so we will hedge our position in starfruit
#     # if the price of starfruits changes for some reason, we will be protected by our amethysts position which may follow the same trend
#     # in practice: say starfruits go up because there is a shortage, we will lose money on our short position in starfruits
#     # but because we can see from the data that amethysts generally move the same way as starfruits in the short term, we will make money on our long position in amethysts
#     # thereby reducing the loss we take on our short position in starfruits.
#
#     if starfruit_position > 0:
#         # we are long starfruits so we want to short amethysts
#         desired_hedge_position = -desired_hedge_position
#     if starfruit_position < 0:
#         # we are short starfruits so we want to long amethysts
#         desired_hedge_position = -desired_hedge_position
#     # Calculate the difference between the desired hedge and current AMETHYSTS position
#     hedge_difference = desired_hedge_position - amethysts_position
#
#     logger.print("Hedge: ", desired_hedge_position)
#     logger.print("Hedge difference: ", hedge_difference)
#
#     if hedge_difference > 0:
#         # If hedge_difference is positive, buy AMETHYSTS to match the hedge
#         logger.print(f"Buying {hedge_difference} AMETHYSTS to match the hedge")
#         # orders.append(Order(product, best_ask, hedge_difference))
#     elif hedge_difference < 0:
#         # If hedge_difference is negative, sell AMETHYSTS to match the hedge
#         logger.print(f"Selling {hedge_difference} AMETHYSTS to match the hedge")
#         # orders.append(Order(product, best_bid, hedge_difference))
#     else:
#         # If hedge_difference is 0, no action is needed
#         logger.print("No hedge adjustment needed for AMETHYSTS")