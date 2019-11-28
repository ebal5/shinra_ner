import argparse
import copy
import io
import json
import logging
import logging.handlers
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
from enum import Flag, auto
from functools import reduce
from itertools import groupby, product
from logging.handlers import TimedRotatingFileHandler
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple

import mojimoji
from joblib import Parallel, delayed

from ner import IREX_NERer
from shinra_error import ShinraError
from tools import html as htmltools

cat2ene = {
    "airport": "1.6.5.3",
    "city": "1.5.1.1",
    "company": "1.4.6.2",
    "compound": "1.10.2",
    "person": "1.1"
}


class ConfigurationError(ShinraError):
    pass


def main():
    parser = _mk_argparser()
    args = parser.parse_args()
    queue = multiprocessing.Queue(-1)
    log_listener = multiprocessing.Process(
        target=_listener_process,
        args=(queue, args.debug, args.log_file))
    log_listener.start()
    dirs = [args.html_dir, args.plain_dir]
    files = [args.id_list]
    # logger = worker_configurer(queue, __name__)
    logger = logging.getLogger(__name__)
    if not _check_environ(files, dirs, logger=logger):
        exit(1)
    logger.info("environ check cleared")
    with open(args.id_list) as f:
        id_list = f.read().split("\n")
    if id_list[-1] == "":
        id_list = id_list[:-1]
    logger.info("I've got id list.")
    ene = cat2ene.get(args.category.lower(), None)
    if ene is None:
        print(f"Invalid category name: {args.category}", file=sys.stderr)
        exit(1)
    # cfg = KNPConfig(multi=args.multi, queue=None,
    #                 knp_cmd=args.knp, knp_opt=args.knp_opt,
    #                 use_jumanpp=args.use_jumanpp,
    #                 juman_cmd=args.juman)
    # knp_extract_files(args.html_dir, args.plain_dir,
    #                   id_list, args.output, ene, config=cfg)
    ginza_extract_files(args.html_dir, args.plain_dir,
                        id_list, args.output, ene, multi=3)
    queue.put_nowait(None)
    log_listener.join()


def _check_environ(files, dirs, *, logger=None):
    logger = logger or logging.getLogger(__name__)
    logger.info("check if some executables exists")
    results = []
    reqs = {"knp", "juman", "jumanpp"}
    exists = [bool(shutil.which(exe)) for exe in reqs]
    if not reduce(lambda x, acc: x & acc, exists):
        misses = [cmd for cmd, ex in zip(reqs, exists) if not ex]
        results.append(f"missing commands: {misses}")
    logger.info("check if dirs are exist")
    exists = [path for path in dirs if not os.path.exists(path)]
    if exists != []:
        results.append(f"not exists: {exists}")
    logger.info("check if files are exist")
    exists = [path for path in files if not os.path.isfile(path)]
    if exists != []:
        results.append(f"not file: {exists}")
    if results != []:
        print("\n".join(results), file=sys.stderr)
        return False
    return True


def _listener_process(queue, debug_mode, log_file="irex_ner.log"):
    level = logging.WARNING if not debug_mode else logging.DEBUG
    logging.basicConfig(level=level)
    root_logger = logging.getLogger(__name__)
    handler = TimedRotatingFileHandler(
        filename=log_file,
        when="D",
        interval=1,
        backupCount=31,
    )
    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(funcName)s [%(levelname)s]: %(message)s'
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    while True:
        try:
            record = queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:
            import traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def worker_configurer(queue, name):
    h = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.DEBUG)
    return logging.getLogger(name)


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
    parser.add_argument("--knp-opt",
                        action='store',
                        type=str,
                        nargs="+",
                        default=['-simple', "-anaphora"],
                        help='options for knp')
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


class KNPConfig(object):
    def __init__(self, *, multi=1, queue=None,
                 knp_cmd="knp", knp_opt=["-simple", "-anaphora"],
                 use_jumanpp=True, juman_cmd=None, juman_opt=None):
        self.knp_cmd = knp_cmd
        self.knp_opt = knp_opt
        self.knp = [self.knp_cmd, *knp_opt] \
            if knp_opt else [self.knp_cmd]
        self.juman_cmd = juman_cmd or ("jumanpp" if use_jumanpp else "juman")
        self.juman_opt = juman_opt or ""
        self.juman = [self.juman_cmd, juman_opt] \
            if juman_opt else [self.juman_cmd]
        self.multi = multi
        self.queue = queue


def knp_extract_files(html_dir, plain_dir, id_list, output, ene,
                      *, logger=None, config=None):
    logger = logger or logging.getLogger(__name__)
    logger.info("will extract files")
    config = config or KNPConfig(logger=logger)
    hdir = Path(html_dir)
    pdir = Path(plain_dir)
    tdir = Path(tempfile.mkdtemp())
    logger.info(f"opened {str(tdir)} as tmpdir")
    Parallel(n_jobs=config.multi)([
        delayed(KNP_NERer(config=config).knp_extract_file)
        (hdir.joinpath(pid+".html"), pdir.joinpath(pid+".txt"),
         ene, pid, tdir)
        for pid in id_list
    ])
    logger.info("all files are extracted")
    try:
        with open(Path(output), "w") as outf:
            for resf in tdir.glob("*.json"):
                with open(resf) as _rf:
                    outf.write(_rf.read())
        shutil.rmtree(tdir)
    except Exception as e:
        logger.error(f"some error: {str(e)}")


class KNP_NERer(IREX_NERer):
    def __init__(self, *, logger=None, config: KNPConfig = None):
        super().__init__(logger=logger)
        self.config = config or KNPConfig()

    def knp_extract_file(self, hpath: Path, ppath: Path,
                         ene, pid, odir: Path):
        self.logger.info(f"start {pid}")
        self.pid = pid
        self.ene = ene
        with open(hpath) as _hf:
            html = _hf.read()
        with open(ppath) as _pf:
            plain = _pf.read()
        try:
            self.title = htmltools.get_title(html)
            analyzed = self.knp_analysis_file(plain)
            self.logger.info(
                f"{sum([len(v) for v in analyzed.values()])} NE's")
        except ShinraError as e:
            self.logger.error(f"PID: {pid} :: {str(e)}")
            raise ShinraError(f"Error in PID: {pid}, msg: {str(e)}", e)
        enekey = ene.replace(".", "_")
        base = {
            "page_id": str(pid),
            "title": self.title,
            "ENE": ene
        }
        with open(odir.joinpath(f"{pid}.json"), "w") as _of:
            mypair = IREX_NERer.pairs[enekey]
            for ne, attrs in mypair.items():
                nel = analyzed.get(ne, None)
                if not nel:
                    continue
                for (attr, ne) in product(attrs, nel):
                    obj = copy.copy(base)
                    obj["text_offset"] = ne
                    obj["attribute"] = attr
                    print(json.dumps(obj), file=_of)
        self.logger.info(f"end {pid}")

    def knp_analysis_file(self, target):
        idx = -1
        nes = []
        stream = io.StringIO(target) if type(target) == str else target
        for line in stream.readlines():
            idx += 1
            if line.isascii():
                continue
            try:
                nel = self.knp_string(line)
                self.logger.debug(f"add {len(nel)} to nes")
                nes.extend([(*entry, idx) for entry in nel])
            except TypeError as e:
                if self.pid:
                    self.logger.error(f"in proceedings of {self.pid}")
                self.logger.error(f"TypeError on '{nel}'")
                self.logger.error(f"Error message: {str(e)}")
            except ShinraError as e:
                if self.pid:
                    self.logger.error(f"in proceedings of {self.pid}")
                self.logger.error(f"Error on '{line.encode()}'")
                self.logger.error(f"Error message: {str(e)}")
                continue
        self.logger.debug(f"all ne: {len(nes)}")
        s_nes = sorted(nes, key=itemgetter(1))
        g_nes = groupby(s_nes, key=itemgetter(1))
        d_nes = {ne: [
            {
                "start": {"line_id": l, "offset": s},
                "end": {"line_id": l, "offset": e},
                "text": w,
            } for w, _, s, e, l in list(lst)] for ne, lst in g_nes}
        self.nes = d_nes
        self._nes = nes
        return d_nes

    def knp_string(self, string: str):
        """
        現在はstring = 1行となっている？
        各行毎にjumanでの解析をかけてknpで解析する
        """
        self.logger.debug(f"convert into analyzable format")
        text = mojimoji.han_to_zen(string)
        text = text.replace(u'\xa0', '　')
        jt = self.juman_string(text)
        kt = self.knp_tab2iobtag(jt)
        nel = self.collect_iobtag(kt)
        _nnel = []
        idx = 0
        for _w, _ne in nel:
            lw = len(_w)
            _nnel.append((string[idx:idx+lw], _ne, idx, idx+lw))
            idx += lw
        return _nnel

    def juman_string(self, string: str):
        jprs = subprocess.run(self.config.juman,
                              input=string, text=True, encoding='utf-8',
                              errors='replace',
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        return jprs.stdout

    def knp_tab2iobtag(self, juman_lines):
        """
        KNP の出力結果を("文字列", IOB2タグ) の形に纏める

        Parameters
        ------------
        juman_lines: str
           juman 形式の文字列（複数行）．EOS行で終了する．空行は含まない

        Returns
        ---------
        [] if raised some error
        """
        try:
            self.logger.debug("will execute KNP")
            kprs = subprocess.run(self.config.knp,
                                  input=juman_lines, text=True,
                                  encoding='utf-8',
                                  errors='replace',
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        except Exception as e:
            self.logger.error("Error in exec KNP")
            self.logger.error(f"Error: {str(e)}")
            print(self.config.knp)
            return []
        nel = []
        for _l in kprs.stdout.split("\n"):
            if _l == "EOS":
                break
            if len(_l) and _l[0] in {"+", "*", "#"}:
                continue
            _ll = _l.split(" ")
            _w = _ll[0]
            _t = _ll[-1]
            idx = _t.find("<NE:")
            if idx == -1:
                nel.append((_w, "<NE:OTHER:S>"))
            else:
                end = _t.find(">", idx)
                tag = _t[idx:end+1]
                nel.append((_w, tag))
        return nel


def ginza_extract_files(html_dir, plain_dir, id_list, output, ene,
                        *, logger=None, multi=1):
    logger = logger or logging.getLogger(__name__)
    logger.info("will extract files")
    hdir = Path(html_dir)
    pdir = Path(plain_dir)
    tdir = Path(tempfile.mkdtemp())
    logger.info(f"opened {str(tdir)} as tmpdir")
    nerer = GiNZA_NERer(logger=logger)
    for pid in id_list:
        nerer.extract_file(
            hdir.joinpath(pid+".html"), pdir.joinpath(pid+".txt"),
            ene, pid, tdir)
    logger.info("all files are extracted")
    try:
        with open(Path(output), "w") as outf:
            for resf in tdir.glob("*.json"):
                with open(resf) as _rf:
                    outf.write(_rf.read())
        shutil.rmtree(tdir)
    except Exception as e:
        logger.error(f"some error: {str(e)}")


class GiNZA_NERer(IREX_NERer):

    def __init__(self, *, logger=None):
        import spacy
        super().__init__(logger=logger)
        self.nlp = spacy.load('ja_ginza')

    def eval_string(self, sentence):
        self.logger.info(f"will evaluate string")
        doc = self.nlp(sentence)
        _nnel = [(ent.text, ent.label_, ent.start_char, ent.end_char)
                 for ent in doc.ents]
        return [(sentence[start:end], ne, start, end)
                for _, ne, start, end in _nnel]

    def analysis_file(self, target):
        idx = -1
        nes = []
        stream = io.StringIO(target) if type(target) == str else target
        for line in stream.readlines():
            idx += 1
            if line.isascii():
                continue
            try:
                nel = self.eval_string(line)
                self.logger.debug(f"add {len(nel)} to nes")
                nes.extend([(*entry, idx) for entry in nel])
            except TypeError as e:
                if self.pid:
                    self.logger.error(f"in proceedings of {self.pid}")
                self.logger.error(f"TypeError on '{nel}'")
                self.logger.error(f"Error message: {str(e)}")
            except ShinraError as e:
                if self.pid:
                    self.logger.error(f"in proceedings of {self.pid}")
                self.logger.error(f"Error on '{line.encode()}'")
                self.logger.error(f"Error message: {str(e)}")
                continue
        self.logger.debug(f"all ne: {len(nes)}")
        s_nes = sorted(nes, key=itemgetter(1))
        g_nes = groupby(s_nes, key=itemgetter(1))
        d_nes = {ne: [
            {
                "start": {"line_id": l, "offset": s},
                "end": {"line_id": l, "offset": e},
                "text": w,
            } for w, _, s, e, l in list(lst)] for ne, lst in g_nes}
        self.nes = d_nes
        self._nes = nes
        return d_nes

    def extract_file(self, hpath: Path, ppath: Path,
                     ene, pid, odir: Path):
        self.logger.info(f"start {pid}")
        self.pid = pid
        self.ene = ene
        with open(hpath) as _hf:
            html = _hf.read()
        with open(ppath) as _pf:
            plain = _pf.read()
        try:
            self.title = htmltools.get_title(html)
            analyzed = self.analysis_file(plain)
            self.logger.info(
                f"{sum([len(v) for v in analyzed.values()])} NE's")
        except ShinraError as e:
            self.logger.error(f"PID: {pid} :: {str(e)}")
            raise ShinraError(f"Error in PID: {pid}, msg: {str(e)}", e)
        enekey = ene.replace(".", "_")
        base = {
            "page_id": str(pid),
            "title": self.title,
            "ENE": ene
        }
        with open(odir.joinpath(f"{pid}.json"), "w") as _of:
            mypair = IREX_NERer.pairs[enekey]
            for ne, attrs in mypair.items():
                nel = analyzed.get(ne, None)
                if not nel:
                    continue
                for (attr, ne) in product(attrs, nel):
                    obj = copy.copy(base)
                    obj["text_offset"] = ne
                    obj["attribute"] = attr
                    print(json.dumps(obj), file=_of)
        self.logger.info(f"end {pid}")



if __name__ == "__main__":
    main()
