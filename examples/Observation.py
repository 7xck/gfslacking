class ConversionObservation:

    def __init__(self, bidPrice: float, askPrice: float, transportFees: float, exportTariff: float, importTariff: float, sunlight: float, humidity: float):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sunlight = sunlight
        self.humidity = humidity

"""
In case you decide to place conversion request on product listed integer number should be returned as “conversions” value from run() method. Based on logic defined inside Prosperity container it will convert positions acquired by submitted code. There is a number of conditions for conversion to  happen:

- You need to obtain either long or short position earlier.
- Conversion request cannot exceed possessed items count.
- In case you have 10 items short (-10) you can only request from 1 to 10. Request for 11 or more will be fully ignored.
- While conversion happens you will need to cover transportation and import/export tariff.
- Conversion request is not mandatory. You can send 0 or None as value.
"""