import time

class Convolution:
    def __init__(self, band, target):
        self.band = band
        self.target = target

    def run(self):
        time.sleep(0.2)        
        print(f"[INFO]\tConvoluting {self.band['input']} to {self.target}")
        return 'hi'
