import numpy as np

def get_machine_precision():
    e = np.float32(1.0)
    i = 0
    while True:
        if 1 + e / 2 == 1 and 1 + e != 1:
            break
        e = e / 2
        i += 1
    return i, e

def calculateArraySumMaxToMin():
    size = 10000
    sum = 0
    for i in range(size, 0, -1):
        sum += ((-1)**(i))/(i)
    return sum

def calculateArraySumMinToMax():
    size = 10000
    sum = 0
    for i in range(1, size+1):
        sum += ((-1)**(i))/(i)
    return sum

def calculateArraySumMinToMaxOddAndEven():
    size = 10000
    sumOdd = 0
    sumEven = 0
    for i in range(1, size+1):
        if i % 2 == 0:
            sumOdd += 1 / i
        else:
            sumEven -= 1 / i
    return sumOdd + sumEven

def calculateArraySumMaxToMinEvenAndOdd():
    size = 10000
    sumOdd = 0
    sumEven = 0
    for i in range(size, 0, -1):
        if i % 2 == 0:
            sumOdd += 1 / i
        else:
            sumEven -= 1 / i
    return sumOdd + sumEven


i, e = get_machine_precision()

print(f"MachinePrecision: {e}, MantiseOrder: {i}")
sum1 = calculateArraySumMaxToMin()
sum2 = calculateArraySumMinToMax()
sum3 = calculateArraySumMinToMaxOddAndEven()
sum4 = calculateArraySumMaxToMinEvenAndOdd()
print(f"OddToEven - EvenToOdd: {sum3 - sum4}")
print(f"MinToMax - MaxToMin: {sum2 - sum1}")
print(f"MinToMax - OddToEven: {sum2 - sum1}")