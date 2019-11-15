import io
import unittest

import pkg_resources as prs

from shinra_ner.tools import offset


class TestOffset(unittest.TestCase):
    """
    特にクラスに属しない関数群に対するテスト
    """
    def test_tag_re(self):
        """
        正規表現がタグを削除しきれるかテスト
        """
        tests = [
            ('', ''),
            ('<a href="http://">test</a>', 'test'),
            ('test <a class=\'TEST\' href="neko">forget', 'test forget')
        ]
        for (test, req) in tests:
            act = offset._tag_re.sub('', test)
            self.assertEqual(act, req)

    def test__mk_len_lst_inr(self):
        tests = [
            ('<a href="http://">test</a>',
            (offset.Kind.TAG, [18, 4, 4])),
            ('test <a class=\'TEST\' href="neko">forget',
             (offset.Kind.TEXT, [5, 28, 6])),
            ('', (offset.Kind.TEXT, [0])),
        ]
        for (test, req) in tests:
            act = offset._mk_len_lst_inr(test)
            self.assertEqual(act, req)

    def test_make_length_list(self):
        html = prs.resource_string("tests", "data/tools/01.html").decode()
        req = [
            (offset.Kind.TAG, [15, 1]),
            (offset.Kind.TAG, [6, 1]),
            (offset.Kind.TEXT, [4, 6, 1]),
            (offset.Kind.TEXT, [0]),
            (offset.Kind.TEXT, [8, 4, 16, 5, 1]),
            (offset.Kind.TEXT, [0]),
            (offset.Kind.TEXT, [8, 3, 19, 4, 1]),
            (offset.Kind.TEXT, [0]),
            (offset.Kind.TEXT, [4, 7, 1]),
            (offset.Kind.TAG, [7, 1]),
        ]
        stream = io.StringIO(html)
        act = offset.make_length_list(stream)
        self.assertEqual(act, req)

    def test_h2p(self):
        # TODO: Error発生チェックの実装
        html = prs.resource_string("tests", "data/tools/01.html").decode()
        stream = io.StringIO(html)
        tests = [
            ((4, 8), (4, 8)),
            ((3, 0), (3, 0)),
            ((4, 12), (4, 8))
        ]
        llt = offset.make_length_list(stream)
        for test, req in tests:
            act = offset.h2p(test, llt)
            self.assertEqual(act, req)

    def test_p2h(self):
        html = prs.resource_string("tests", "data/tools/01.html").decode()
        stream = io.StringIO(html)
        llt = offset.make_length_list(stream)
        tests = [
            ((4, 8), (4, 12)),
            ((4, 9), (4, 13)),
            ((3, 0), (3, 0)),
        ]
        for test, req in tests:
            act = offset.p2h(test, llt)
            self.assertEqual(act, req)


if __name__ == "__main__":
    unittest.main()
