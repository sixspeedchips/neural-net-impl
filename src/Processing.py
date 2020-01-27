import numpy as np

def list_to_bin(piles, bits=6):

    return_list = list()

    for pile in piles:
        temp = list(f"{pile:b}")
        [temp.insert(0, "0") for _ in range(bits - len(temp))]
        [return_list.append(float(i)) for i in temp]
    return np.array(return_list, ndmin=2, dtype=np.float64)

def bin_to_list(binary_input):

    bits = int(len(binary_input)/3)
    piles = []
    count = 0
    for i in range(3):
        number = 0
        for j in range(bits):
            number += binary_input[count]*2**(bits-j-1)
            count+=1
        piles.append(int(number))
    return piles

def process_move(binary_output):

    move = binary_output[0:6].tolist()
    pile = binary_output[6::].tolist()
    # print(move,pile)
    # print(len(pile))
    value = 0
    move.reverse()
    for j in range(len(move)):
        value += move[j] * 2 ** j
    return int(value), int(np.argmax(pile))