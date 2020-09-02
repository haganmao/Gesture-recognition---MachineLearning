import csv
import numpy as np
from decimal import Decimal
samples = 150
total_moments = 7
def read_moments(filename):
    tokens = np.zeros((total_moments, samples), float)
    fin = open(filename)
    m = 0
    for line in fin:
        if m < 7:
            aLine = line.split(',')
            print (np.size(aLine))
            aLine[samples-1] = aLine[samples-1].replace('\n', '') #Delete \n if its there
            for c in range(0, samples):
                tokens[m, c] = Decimal(aLine[c])
        m += 1
    # print(tokens)
    return tokens
