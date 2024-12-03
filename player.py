from const_pai import code2disphai, code2hai

sutehai_flags = {
    "tedashi": 0,
    "tsumogiri": 1,
    "naki": 2,
    "richi": 4,
}

class Player:
    def __init__(self, pid, name, kyoku):
        self.id = pid
        self.name = name
        self.tehai = []
        self.tsumo = 0
        self.furo = []
        self.sutehai = []
        self.sutehai_flags = []
        self.richi = False
        self.tsumogiri = False
        self.point = 0
        self.kaze = 0
        self.kyoku = kyoku
        self.errors = [] # エラーを記録するリスト

    def do_ripai(self):
        self.tehai.sort()

    def do_haipai(self, haipai):
        self.tehai.extend(haipai)
        self.do_ripai()

    def do_tsumo(self, tsumo):
        self.tsumo = tsumo
        self.tsumogiri = False

    def do_sutehai(self, sutehai, tsumogiri):     
        self.sutehai.append(sutehai)
        if tsumogiri:
            self.sutehai_flags.append(sutehai_flags["tsumogiri"])
        else:
            if 0 < self.tsumo:
                self.tehai.append(self.tsumo)
                #if sutehai not in self.tehai:
                    #raise ValueError(f"エラー: ツモ牌{code2disphai[sutehai]}が手牌にありません")
                if sutehai not in self.tehai:
                    raise ValueError(f"エラー: ツモ牌{sutehai}が手牌にありません (tehai: {self.tehai})")
            self.tehai.remove(sutehai)
            self.sutehai_flags.append(sutehai_flags["tedashi"])
        self.tsumo = 0
        self.tsumogiri = tsumogiri
        self.do_ripai()
        
    def do_richi(self):
        self.richi = True
        self.sutehai_flags[-1] += sutehai_flags["richi"]

    def do_open_kakan(self, tedashi, naki):
        if 0 < self.tsumo:
            self.tehai.append(self.tsumo)
        self.tsumo = 0
        self.tehai.remove(naki)
        self.furo.append(naki)

    def do_open_ankan(self, tedashi, naki):
        self.tehai.append(self.tsumo)
        self.tsumo = 0
        for hai in tedashi:
            self.tehai.remove(hai)    
            self.furo.append(hai)

    def do_open_ponchi(self, tedashi, naki):
        for hai in tedashi:
            self.tehai.remove(hai)
            self.furo.append(hai)
        # self.kyoku.teban[-2].sutehai.pop() # remove the last sutehai
        self.kyoku.teban[-2].sutehai_flags[-1] += sutehai_flags["naki"]
        self.furo.append(naki)


    def show(self):
        def is_tsumogiri(flag):
            return 0 < (flag % 2)
        
        def is_naki(flag):
            return 0 < ((flag % 4) // 2)
        
        def is_richi(flag):
            return 0 < ((flag % 8) // 4)

        disp_tehai = [code2disphai[self.tehai[idx] if idx < len(self.tehai) else 0] for idx in range(13)]
        disp_furo = [code2disphai[self.furo[idx] if idx < len(self.furo) else 0] for idx in range(16)]
        # sutehai display
        disp_sutehai = [code2disphai[self.sutehai[idx] if idx < len(self.sutehai) else 0] for idx in range(25)]
        disp_richi_flags = [is_richi(self.sutehai_flags[idx]) if idx < len(self.sutehai_flags) else 0 for idx in range(25)]
        disp_naki_flags = [is_naki(self.sutehai_flags[idx]) if idx < len(self.sutehai_flags) else 0 for idx in range(25)]
        disp_tsumogiri_flags = [is_tsumogiri(self.sutehai_flags[idx]) if idx < len(self.sutehai_flags) else 0 for idx in range(25)]
        disp_sutehai_flags = ["R" if r else "v" if n else "*" if t else " " for r, n, t in zip(disp_richi_flags, disp_naki_flags, disp_tsumogiri_flags)]
        disp_sutehai = [hai + flag for hai, flag in zip(disp_sutehai, disp_sutehai_flags)]

        disp_rich = "R" if self.richi else " "
        disp_str = "".join(disp_tehai) + "|" + code2disphai[self.tsumo] + "|" + "".join(disp_furo) + "|" + "".join(disp_sutehai) + "|" + disp_rich + "|" + str(self.point)
        print(self.name + ":" + disp_str)

    def __str__(self):
        # fmt: off
        str_array = [
            "tehai:",     str(self.tehai),
            "tsumo:",     str(self.tsumo),
            "furo:",      str(self.furo),
            "sutehai:",   str(self.sutehai),
            "richi:",     str(self.richi),
            "tsumogiri:", str(self.tsumogiri),
            "point:",     str(self.point),
            "kaze:",      str(self.kaze),
        ]
        # fmt: on
        return " ".join(str_array)




# 手牌、副露、捨て牌、フラグ、点数を入力として受け取り、それを正規化して返す関数

    def normal_hai_code(self, hai):
        return hai / (len(code2disphai) - 1)

    def make_tehai(self):
        tehai_data = [self.normal_hai_code(self.tehai[idx]) if idx < len(self.tehai) else 0.0 for idx in range(13)]
        return tehai_data
    
    def make_furo(self):
        furo_data = [self.normal_hai_code(self.furo[idx]) if idx < len(self.furo) else 0.0 for idx in range(16)]
        return furo_data
    
    def make_sutehai(self):
        sutehai_data = [self.normal_hai_code(self.sutehai[idx]) if idx < len(self.sutehai) else 0.0 for idx in range(25)]
        return sutehai_data

    def make_flag(self):
        def is_richi(flag):
            return 1 if 0 < ((flag % 8) // 4) else 0.5
        richi_flags_data = [is_richi(self.sutehai_flags[idx]) if idx < len(self.sutehai_flags) else 0.0 for idx in range(25)]
        
        def is_naki(flag):
            return 1 if 0 < ((flag % 4) // 2) else 0.5
        naki_flags_data = [is_naki(self.sutehai_flags[idx]) if idx < len(self.sutehai_flags) else 0.0 for idx in range(25)]
        
        def is_tsumogiri(flag):
            return 1 if 0 < (flag % 2 ) else 0.5
        tsumogiri_flags_data = [is_tsumogiri(self.sutehai_flags[idx]) if idx < len(self.sutehai_flags) else 0.0 for idx in range(25)]
    
        return richi_flags_data, naki_flags_data, tsumogiri_flags_data
    
    
    