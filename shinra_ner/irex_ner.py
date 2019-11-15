import argparse
import glob
import logging
import os
import re
import subprocess
import sys
from enum import Flag, auto
import shutil
from itertools import groupby
from logging.handlers import TimedRotatingFileHandler
from operator import itemgetter
from typing import List, Tuple
from functools import reduce
import tempfile

import mojimoji
from joblib import Parallel, delayed

from shinra_ner.shinra_error import ShinraError


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


class ConfigurationError(ShinraError):
    pass


def main():
    parser = _mk_argparser()
    args = parser.parse_args()
    logger = _mk_logger(args.debug_mode)
    dirs = [args.html_dir, args.plain_dir]
    files = [args.id_list]
    if _check_environ(files, dirs, logger=logger):
        exit(1)
        pass
    with open(args.id_list) as f:
        id_list = f.read().split("\n")
    knp_extract_files(args.html_dir, args.plain_dir,
                      id_list, args.output, logger=logger)


def _check_environ(files, dirs, *, logger=None):
    logger = logger or logging.getLogger(__name__)
    logger.info("check if some executables exists")
    result = []
    reqs = {"knp", "juman", "jumanpp"}
    exists = [bool(shutil.which(exe)) for exe in reqs]
    if not reduce(lambda x, acc: x & acc, exists):
        misses = [cmd for cmd, ex in zip(reqs, exists) if not ex]
        results.append(f"missing commands: {*misses}")
    logger.info("check if dirs are exist")
    exists = [path for path in dirs if not os.path.exists(path)]
    if exists != []:
        results.append(f"not exists: {*exists}")
    logger.info("check if files are exist")
    exists = [path for path in files if not os.path.isfile(path)]
    if exists != []:
        results.append(f"not file: {*exists}")
    if results != []:
        print("\n".join(results))
        return False
    return True


def _mk_logger(debug_mode):
    level = logging.WARNING if not debug_mode else logging.DEBUG
    logging.basicConfig(level=level)
    root_logger = logging.getLogger(__name__)
    handler = TimedRotatingFileHandler(
        filename=args.log_file,
        when="D",
        interval=1,
        backupCount=31,
    )
    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(funcName)s [%(levelname)s]: %(message)s'
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    return root_logger


def _mk_argparser():
    parser = argparse.ArgumentParser(
        description="irex ne recognition and attribute extraction")
    parser.add_argument('--multi',
                        action='store',
                        default=1,
                        type=int,
                        help='How many cores can I use?')
    parser.add_argument("--log-file",
                        action='store',
                        default='irex_ner.log',
                        help='Where to save log file')
    parser.add_argument("--knp",
                        action='store',
                        default='knp',
                        help='executable of knp')
    parser.add_argument("--use-jumanpp",
                        action='store',
                        default=True,
                        help="do or don't use jumanpp")
    parser.add_argument("--juman",
                        action='store',
                        default='jumanpp',
                        help='executable of juman(pp)')
    parser.add_argument("category",
                        action='store',
                        help="Category name.")
    parser.add_argument("html_dir",
                        action='store',
                        help="Directory which html files are in")
    parser.add_argument("plain_dir",
                        action='store',
                        help="Directory which plain text files are in")
    parser.add_argument("id_list",
                        action='store',
                        help="Extraction target list")
    parser.add_argument("--debug",
                        action="store_true",
                        default=False,
                        help="Want to show debug logs?")
    parser.add_argument("output",
                        action="store",
                        default=False,
                        help="File path to output")
    return parser


class IREX_NE(Flag):
    BEGIN = auto()
    END = auto()
    INNER = auto()
    SINGLE = auto()
    PERSON = auto()
    DATE = auto()
    TIME = auto()
    LOCATION = auto()
    MONEY = auto()
    ORGANIZATION = auto()
    ARTIFACT = auto()


class Config(object):
    def __init__(self, *, multi=1, logger=None,
                 knp_cmd="knp",
                 knp_opt=["-simple", "-anaphora"],
                 use_jumanpp=True,
                 juman_cmd=None,
                 juman_opt=None):
        self.logger = logger or logging.getLogger(__name__)
        self.knp_cmd = knp_cmd
        self.knp_opt = knp_opt
        self.knp = [self.knp_cmd, *knp_opt] \
            if knp_opt else [self.knp_cmd]
        self.juman_cmd = juman_cmd or ("jumanpp" if use_jumanpp else "juman")
        self.juman_opt = juman_opt or ""
        self.juman = [self.juman_cmd, juman_opt] \
            if juman_opt else [self.juman_cmd]
        self.multi = multi


def knp_irex(line, *, logger=None, config=None):
    """
    KNPによるIREXタグの抽出

    Parameters
    ---------
    line: str
       解析対象行の文字列
    """
    logger = logger or logging.getLogger(__name__)
    config = config or Config(logger=logger)
    text = mojimoji.han_to_zen(line)
    text = text.replace(u'\xa0', '　')
    diff = [i for i, b, a in zip(range(len(text)), line, text) if b != a]
    try:
        jprs = subprocess.run(config.juman, input=text, text=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        kprs = subprocess.run(config.knp, input=jprs.stdout, text=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    except UnicodeDecodeError as e:
        raise ShinraError(e)
    nel = []
    acc = 0
    for line in kprs.stdout.split("\n"):
        if len(line) and line[0] in {"+", "*", "#"}:
            continue
        lst = line.split(" ")
        _t = lst[-1]
        _w = lst[0]
        while diff and (acc + len(_w)) > diff[0]:
            wp = diff[0] - acc
            tmp = _w
            _w = _w[:wp] + mojimoji.zen_to_han(_w[wp]) + _w[wp+1:]
            if tmp == _w:
                _w = _w[:wp] + '\xa0' + _w[wp+1:]
            diff = diff[1:]
        if _t.startswith('<NE:'):
            nel.append((_w, _t, acc))
        else:
            nel.append((_w, "<NE:OTHER:S>", acc))
        acc += len(_w)
    return nel


def net_merge(nel):
    """
    解析結果からIREX NEを作成する．

    B-XXX I-XXX ... E-XXX を一まとまりとして文字列を作成させる# ．
    """
    nes = []
    buf = ""
    start = None
    for word, net, pos in nel:
        ne = net[4:-3]
        if net == '<NE:OTHER:S>':
            continue
        elif net[-2] == "S":
            nes.append((ne, word, pos, pos+len(word)))
        elif net[-2] == "B":
            buf = word
            start = pos
        elif net[-2] == "E":
            buf += word
            nes.append((ne, buf, start, start+len(buf)))
            buf = ""
        elif net[-2] == "I":
            buf += word
    return nes


def knp_analysis_file(stream, *, logger=None, config=None):
    logger = logger or logging.getLogger(__name__)
    config = config or Config(logger=logger)
    idx = -1
    nes = []
    for line in stream.readlines():
        idx += 1
        if line.isascii():
            continue
        try:
            nel = knp_irex(line, logger=logger, config=config)
        except ShinraError as e:
            logger.error(f"UnicodeDecodeError on '{line.encode()}'")
            continue
        nes.extend([(*entry, idx) for entry in net_merge(nel)])
    s_nes = sorted(nes, key=itemgetter(0))
    g_nes = groupby(s_nes, key=itemgetter(0))
    d_nes = {ne: [
        {
            "start": {"line_id": l, "offset": s},
            "end": {"line_id": l, "offset": e},
            "text": w,
        } for _, w, s, e, l in list(lst)] for ne, lst in g_nes}
    return d_nes


def knp_extract_files(html_dir, plain_dir, id_list, output, ene,
                      *, logger=None):
    base = {
        "ENE": ene
    }
    pass


def knp_extract_file(config, *, logger=None):
    logger = logger or logging.getLogger(__name__)
    pass


if __name__ == "__main__":
    main()


def test():
    p = "/home/s173342/Data/Shinra/2019/plain/airport/31781.txt"
    with open(p) as f:
        return knp_analysis_file(f)
