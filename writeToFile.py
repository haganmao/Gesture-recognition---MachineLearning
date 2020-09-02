import csv
total_moments = 7
samples = 150
def write_moments(filename, moments):
    # write to file
    with open(filename, 'a') as f:
        for m in range(0, total_moments):
            
            for j in range(0, samples-1):
                f.write(str(moments[m, j]))
                f.write(',')
            f.write(str(moments[m, samples-1]))
            f.write('\n')