import json
import argparse
import pickle

from kyoku import Kyoku
from const_pai import code2disphai


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="paifu file")
    return parser.parse_args()


def load_paifu(file):
    with open(file) as f:
        json_data = json.load(f)
    return json_data


def count_kyoku(json_data):
    cnt = 0
    for entry in json_data:
        if entry["cmd"] == "kyokustart":
            cnt += 1
    return cnt


def extract_one_kyoku(json_data, kyoku_num):
    max_kyoku_num = count_kyoku(json_data)
    if kyoku_num < 0:
        raise ValueError("kyoku_num is too small")
    elif max_kyoku_num < kyoku_num:
        raise ValueError("kyoku_num is too large")

    kyoku_num += 1
    kyoku = []
    for entry in json_data:
        if entry["cmd"] == "kyokustart":
            kyoku_num -= 1
            if kyoku_num == 0:
                kyoku.append(entry)
        elif kyoku_num == 0:
            kyoku.append(entry)
            if entry["cmd"] == "kyokuend":
                break
    return kyoku


def show_kyoku(kyoku_data):
    all_data = []
    kyoku = Kyoku(kyoku_data)
    while True:
        kyoku.check_sutehai()
        if kyoku.is_sutehai:
            print("--------------------")
            kyoku.show()
            trdata = kyoku.make_tr_data()
        playing = kyoku.step()
        if kyoku.is_sutehai:
            print(f"sutehai: {code2disphai[kyoku.sutehai]}")
            sutehai = kyoku.sutehai
            all_data.append([trdata,sutehai])
        if not playing:
            break
    return all_data


"""
        print(f"{kyoku.is_sutehai}")
        if kyoku.is_sutehai == True:
            kyoku.show()
            trdata = kyoku.make_tr_data()
            all_data.append([trdata, "sutehai"])
            kyoku.is_sutehai = False
        return all_data
"""

if __name__ == "__main__":
    args = parse_args()
    hoge = []
    for file in args.files:
        json_data = load_paifu(file)
        #print(count_kyoku(json_data))
    
        for kyoku_num in range(count_kyoku(json_data)):
            print(f"kyoku_num: {kyoku_num} =======================")
            kyoku_data = extract_one_kyoku(json_data, kyoku_num)
            train_kyoku_data = show_kyoku(kyoku_data)
            hoge.extend(train_kyoku_data)
    
    #print(hoge)
    with open("data.pkl", "wb") as f:
        pickle.dump(hoge, f)
    
"""
            for entry in kyoku_data:
                #print(entry)
                #print(entry["cmd"])
                train_kyoku_data = show_kyoku(entry)
                hoge.extend(train_kyoku_data)
                #print(show_kyoku(kyoku_data))
                #print(train_kyoku_data)
"""



