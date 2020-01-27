
from Processing import list_to_bin,process_move
from NN_2 import neural_net
import random

bits = 6


def play_game():

    piles = get_piles()

    while sum(piles)>0:
        output_piles(piles)
        piles = validate_move(piles)
        if check_win(piles,"Player"): return

        piles = computer_turn(piles)
        if check_win(piles,"Computer"): return

def check_win(piles,turn):
    if sum(piles) == 0:
        print(f"the {turn} has won!")
        return True
    else: return False



def output_piles(piles):
    _ = " ".join([str(p) for p in piles])
    print(f"the pile sizes are {_}.")

def get_piles():

    inp = input("Enter a 3-pile list of ints(0-63) separated by commas, or type r for random piles:")
    piles = [int(pile) for pile in inp.split(",") if pile.isdigit() and 0<int(pile)<64]
    if inp =='r':
        piles = [random.randrange(63) for _ in range(3)]
        return piles
    while (len(piles)!= 3):
        print(f"{inp} is an invalid pile set")
        inp = input("Enter a 3-pile list of ints(0-63), separated by commas:")
        piles = [int(pile) for pile in inp.split(",") if pile.isdigit() and 0<int(pile)<64]
    return piles

def validate_move(piles):
    inp = input("Enter a pile(1-3) and an amount to deduct(1-63) separated by a comma(pile, minus): ")
    move = [int(move) for move in inp.split(",") if move.isnumeric() and not '']
    pile,minus = move[0], move[1]
    while (3<pile<1 or minus>piles[pile-1]):
        print("Invalid move")
        inp = input("Enter a pile(1-3) and an amount to deduct(1-63) separated by a comma(pile, minus): ")
        move = [int(move) for move in inp.split(",") if move.isnumeric()]
        pile, new_pile = move[0], move[1]

    piles[pile-1] -= minus
    return piles

def computer_turn(piles):
    t = nn.prediction(list_to_bin(piles)[0])
    move,idx = process_move(t)
    if piles[idx] > move:
        reduction = int(move)
        print(f"The computer takes {piles[idx]-reduction} from pile {idx+1}")
        piles[idx] = reduction
    else:
        for i,pile in enumerate(piles):
            if pile>0:
                print(f"The computer takes 1 from pile {i + 1}")
                piles[i]-=1

                break
    return piles

if __name__ == "__main__":
    nn = neural_net()
    nn.load("trained4")
    c = play_game()
    while(input("Would you like to play again? (Hit enter to Continue) ").lower() is not "no"):
        c = play_game()