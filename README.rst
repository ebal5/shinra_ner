森羅プロジェクト Named Entity Recognition
=========

NER を利用した抽出器の作成（for Shinra2019 JP-5 and JP-30）

面倒なことは嫌なのでとりあえず CC0 でライセンスしてみた．
よく考えたら Apache との互換性がないとかで死ぬのでは？
まぁあまり気にするでもないか．

データ形式
---------

提出形式
^^^^^^^^^

::
   {
   "page_id": ページ ID: str,
   "title": ページタイトル: str,
   "ENE": Extended Named Entity: str,
   "attribute": 属性名: str,
   "html_offset": {
       "start": {
           "line_id": 開始位置行番号: int,
           "offset": 開始位置文字番号: int,
       }
       "end": {
           "line_id": 終了位置行番号: int,
           "offset": 終了位置文字番号: int,
       }
       "text": 内部に存在する文字列: str
   },
   "text_offset": {
       "start": {
           "line_id": 開始位置行番号: int,
           "offset": 開始位置文字番号: int,
       }
       "end": {
           "line_id": 終了位置行番号: int,
           "offset": 終了位置文字番号: int,
       }
       "text": 内部に存在する文字列: str
   }
   }

irex_ner
---------

IREX の定義に基づく NER プログラム．提出形式での結果を出力する

tools
---------

offset
^^^^^^^^^
h2p, p2h がメイン．それぞれ html のオフセットからプレインテキストのそれへ変換するものとその逆変換である．
ただし，純粋な逆関数となっていないことに注意．
html タグの開始文字"<"は html offset ではそれ自身を示すが，
テキストでは次のテキスト要素の先頭を示される．

- p2h は one-to-one
- h2p は n-to-one
