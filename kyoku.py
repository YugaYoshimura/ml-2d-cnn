from player import Player
from const_pai import code2hai, code2disphai

class Kyoku:
    def __init__(self, kyoku_data: list):
        self.player_names = ["A0", "B0", "C0", "D0"]
        self.players = {}
        for player_name in self.player_names:
            self.players[player_name] = Player(player_name, self)

        self.oya = None
        self.dora = []
        self.honba = 0
        self.bakaze = 0
        self.kyoutaku = 0

        self.kyoku_data = kyoku_data
        self.current_step = 0
        self.teban = []
        self.is_sutehai = False
        self.sutehai = 0

        # fmt: off
        self.commands = {
            "haipai":     self.do_haipai,
            "ryukyoku":   self.do_dummy,
            "dice":       self.do_dummy,
            "sutehai":    self.do_sutehai,
            "kyokustart": self.do_kyokustart,
            "tsumo":      self.do_tsumo,
            "point":      self.do_point,
            "dora":       self.do_dora,
            "open":       self.do_open,
            "kyokuend":   self.do_kyokuend,
            "say":        self.do_dummy,
            "richi":      self.do_richi,
            "uradora":    self.do_dora,
            "agari":      self.do_dummy,
        }

    # fmt: on

    def get_player(self, name):
        player = self.players[name]
        if len(self.teban) == 0 or self.teban[-1] != player:
            self.teban.append(player)
        return player

    def step(self):
        playing = True
        entry = self.kyoku_data[self.current_step]
        if entry["cmd"] not in self.commands:
            raise ValueError(f"Invalid command: {entry['cmd']}")
        playing = self.commands[entry["cmd"]](entry["args"])
        self.current_step += 1
        return playing

    def check_sutehai(self):
        self.is_sutehai = False
        entry = self.kyoku_data[self.current_step]
        if entry["cmd"] == "sutehai":
            self.is_sutehai = True
        return self.is_sutehai 

    def do_dummy(self, args):
        return True

    def do_kyokustart(self, args):
        self.oya = self.players[args[1]]
        self.honba = args[2]
        self.bakaze = code2hai.index(args[4])
        self.kyoutaku = args[3]
        for idx in range(4):
            self.players[self.player_names[idx]].kaze = code2hai.index(args[5:][idx])
        return True

    def do_kyokuend(self, args):
        return False

    def do_haipai(self, args):
        player = self.get_player(args[0])
        haipai_str = args[1]
        haipai = [code2hai.index(haipai_str[idx : idx + 2]) for idx in range(0, len(haipai_str), 2)]
        player.do_haipai(haipai)
        return True

    def do_tsumo(self, args):
        player = self.get_player(args[0])
        tsumo_code = code2hai.index(args[2])
        player.do_tsumo(tsumo_code)
        return True

    def do_sutehai(self, args):
        player = self.get_player(args[0])
        sutehai_code = code2hai.index(args[1])
        tsumogiri = True if len(args) == 3 and args[2] == "tsumogiri" else False
        player.do_sutehai(sutehai_code, tsumogiri)
        self.sutehai = sutehai_code
        return True

    def do_dora(self, args):
        if args[1] in code2hai:
            dora_code = code2hai.index(args[1])
            self.dora.append(dora_code)
        return True

    def do_open(self, args):
        open_flag = args[1][0]
        if open_flag not in ["[", "(", "<"]:
            return True

        player = self.get_player(args[0])
        open_funcs = {
            "[": player.do_open_kakan,
            "(": player.do_open_ankan,
            "<": player.do_open_ponchi
        }
        tedashi_str = args[1][1:-1]
        tedashi_code = [code2hai.index(tedashi_str[idx : idx + 2]) for idx in range(0, len(tedashi_str), 2)]
        naki_code = code2hai.index(args[2]) if len(args) == 3 else 0
        open_funcs[open_flag](tedashi_code, naki_code)
        return True

    def do_richi(self, args):
        self.players[args[0]].do_richi()
        return True

    def do_point(self, args):
        player = self.players[args[0]]
        point_op = args[1][0]
        if point_op == "+":
            player.point += int(args[1][1:])
        elif point_op == "-":
            player.point -= int(args[1][1:])
        elif point_op == "=":
            player.point = int(args[1][1:])
        else:
            player.point = int(args[1])
        return True

    def show(self):
        dora_disp = "".join([code2disphai[self.dora[idx] if idx < len(self.dora) else 0] for idx in range(4)])
        if 0 < len(self.teban):
            print("teban: " + self.teban[-1].name + " dora: " + dora_disp)
        for player_name in self.player_names:
            self.players[player_name].show()

    def make_tr_data(self):
        point_threshold = (-12000, -4000, 0, 4000, 12000)
        trdata = []
        if 0 < len(self.teban):
            teban_player = self.teban[-1]

            #手牌は自分のものだけ追加
            trdata.extend(teban_player.make_tehai())

            base_idx = self.player_names.index(teban_player.name)
            base_point = teban_player.point
            for add_idx in range(4):
                idx = (base_idx + add_idx) % 4
                player_name = self.player_names[idx]
                p = self.players[player_name]
                
                
                #鳴きと捨て牌を追加
                trdata.extend(p.make_furo() + p.make_sutehai())
                
                # フラグ情報を一度に取得して追加
                richi_flags, naki_flags, tsumogiri_flags = p.make_flag()
                trdata.extend(richi_flags + naki_flags + tsumogiri_flags)
                
                #点棒情報を追加
                if 0  < add_idx:
                    diff_point = p.point - base_point
                    normalized_point = len(point_threshold)
                    for p_idx, poi_t in enumerate(point_threshold):
                        if diff_point < poi_t:
                            normalized_point = p_idx
                            break
                    trdata.append(normalized_point / 5)
                    
                #親情報を追加
                if p == self.oya:
                    trdata.append(1)
                else:
                    trdata.append(0)

            #ドラ情報を追加
            dora_info = self.dora[:]
            dora_data = []
            if len(dora_info) < 4:
                dora_info.extend([0] * (4 - len(dora_info)))
                for idx in range(4):
                    dora_data.append(dora_info[idx] / (len(code2disphai) - 1))
            trdata.extend(dora_data)

            #本場・場風・供託情報を追加
            trdata.append(int(self.honba[0]) / 30)
            trdata.append(self.bakaze / (len(code2disphai) - 1))
            trdata.append(float(self.kyoutaku) / 10000)

        return trdata

            



