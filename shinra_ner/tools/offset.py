import re
from enum import Enum, auto
from typing import List, Tuple

from shinra_ner.shinra_error import ShinraError


class OffsetError(ShinraError):
    pass


class Kind(Enum):
    TEXT=auto()
    TAG=auto()


_tag_re = re.compile(r'<("[^"]*"|\'[^\']*\'|[^\'">])*>')


def _mk_len_lst_inr(line: str)->Tuple[Kind, List[int]]:
    span = sum([list(mt.span()) for mt in _tag_re.finditer(line)], [])
    kind = Kind.TAG if len(span) > 0 and span[0] == 0 else Kind.TEXT
    lsd0 = len(span) > 0
    if lsd0 and span[0] != 0:
        span = [0, *span]
    ll = len(line)
    if lsd0 and span[-1] != ll:
        span.append(ll)
    lengths = [elm[1] - elm[0] for elm in zip(span, span[1:])] or [0]
    return (kind, lengths)


def make_length_list(stream)->List[Tuple[Kind, List[int]]]:
    """
    Parameters
    ---------
    stream:
       html文字列のストリーム．ファイルストリームあるいは文字列．
    Returns
    ---------
    length_list: List[Tuple[Kind, List[int]]]
    """
    return [_mk_len_lst_inr(line) for line in stream.readlines()]


def h2p(offset, len_list):
    """
    Parameters
    ---------
    offset: Tuple[int, int]
       htmlでのオフセット
    len_list: List[Tuple[Kind, List[int]]]
       make_length_listで生成される表らしきさむしんぐ
    Returns
    ---------
    po: Tuple[int, int]
       Plain Textでのオフセット
    Raises
    ---------
    AssertionError
       offset[0] > len(len_list)
    OffsetError
       offsetが示す文字がタグの範囲でかつその先頭でない（"<"でない）
    """
    def even(n):
        return n%2==0
    def odd(n):
        return n%2==1
    hr, hc = offset
    pr = hr
    assert len(len_list) >= hr, f"len(len_list) < {hl}"
    tl = len_list[pr]
    check = even if tl[0] == Kind.TAG else odd
    # check: 該当するインデックスの要素はタグ
    tls = [sum(tl[1][:i+1]) for i in range(len(tl[1]))]
    _target_idx = min([i for i in range(len(tl[1]))
                       if tls[i] > hc], default=None)
    if not _target_idx and hc == 0:
        return (pr, 0)
    if not _target_idx or (check(_target_idx) and \
       hc != tl[1][_target_idx-1]):
        raise OffsetError(f"not in text or head of tag, {offset}")
    tl_slice = tl[1][:_target_idx]
    _tag_len = sum(
        [elm for i, elm in zip(range(len(tl_slice)), tl_slice)
         if check(i)]
    )
    pc = hc - _tag_len
    return (pr, pc)


def p2h(offset, len_list):
    """
    Parameters
    ---------
    offset:
    len_list:
    Returns
    ---------
    ho
       Htmlでのオフセット
    Raises
    ---------
    AssertionError
       offset[0] > len(len_list)
    """
    def even(n):
        return n%2==0
    def odd(n):
        return n%2==1
    pr, pc = offset
    hr = pr
    tl = len_list[pr]
    check = even if tl[0] == Kind.TEXT else odd
    # check: 該当するインデックスの要素はテキスト
    texts = [tl[1][i] for i in range(len(tl[1])) if check(i)]
    tsums = [sum(texts[:i+1]) for i in range(len(texts))]
    idx = min([i for i in range(len(tsums))
               if tsums[i] > pc], default=None)
    tag_len = sum(
        [elm for i, elm in zip(range(len(tl[1])), tl[1])
         if not check(i) and i <= (2*idx if tl[0] == Kind.TAG else 2*idx-1)]
    )
    hc = pc + tag_len
    return (hr, hc)
