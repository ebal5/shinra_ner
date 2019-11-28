import logging

from shinra_error import ShinraError


class NamedEntityRecognizer(object):
    def __init__(self, *, logger=None):
        self.logger = logger or logging.getLogger("NER")

    def collect_iobtag(self, iobtagl):
        tagl = []
        buf = ""
        for _w, _t in iobtagl:
            ne = _t[4:-3]
            ptn = _t[-2]
            if ptn in {"S", "E"}:
                buf += _w
                tagl.append((buf, ne))
                buf = ""
            elif ptn in {"B", "I"}:
                buf += _w
            else:
                self.logger.error(f"invalid:: {_w, _t}")
                raise ShinraError("Invalid IOBTAG")
        return tagl


class IREX_NERer(NamedEntityRecognizer):
    pairs = {'1_1': {'ARTIFACT': ['作品', '受賞歴', '参加イベント'],
                     'DATE': ['生年月日', '没年月日'],
                     'LOCATION': ['生誕地', '居住地', '没地'],
                     'ORGANIZATION': ['所属組織'],
                     'PERSON': ['師匠', '両親', '家族']},
             '1_10_2': {},
             '1_4_6_2': {'ARTIFACT': ['取扱商品', '商品名'],
                         'DATE': ['従業員数（単体）データの年'],
                         'LOCATION': ['創業国', '創業地', '本拠地国', '本拠地'],
                         'MONEY': ['売上高（単体）'],
                         'ORGANIZATION': ['子会社・合弁会社', '買収・合併した会社', '主要株主'],
                         'PERSON': ['代表者']},
             '1_5_1_1': {'ARTIFACT': ['観光地', '恒例行事', '特産品'],
                         'DATE': ['人口データの年'],
                         'LOCATION': ['友好市区町村'],
                         'ORGANIZATION': ['鉄道会社'],
                         'PERSON': ['首長']},
             '1_6_5_3': {'ARTIFACT': ['名称由来人物の地位職業名'],
                         'DATE': ['年間利用者数データの年'],
                         'LOCATION': ['国', '所在地', '母都市'],
                         'ORGANIZATION': ['運営者'],
                         'TIME': ['運用時間']}}

    def __init__(self, *, logger=None):
        super().__init__(logger=logger)
