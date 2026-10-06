"""各ルールの「いつ入って、いつ出るか」を日本語で説明する。

ダッシュボードのルール辞典・検索・説明書で使う。ルール名から機械的に引けるように
正規表現で対応づけている。新しいルールを足したらここにも1行足すこと。
"""
from __future__ import annotations

import re

DOW = {"月": "月曜", "火": "火曜", "水": "水曜", "木": "木曜", "金": "金曜"}


def describe(name: str) -> dict:
    """{cond: 条件, entry: 入り, exit: 出口, idea: 狙い, side: 買い/売り}"""
    m = re.match(r"曜日:(.)曜に買い", name)
    if m:
        d = DOW[m.group(1)]
        return dict(cond=f"その日が{d}日", entry=f"{d}日の引けで買う", exit="翌営業日の引けで売る",
                    idea="曜日によって値動きに偏りがあるか（曜日アノマリー）", side="買い")
    m = re.match(r"順張り:過去(\d+)日上昇", name)
    if m:
        k = m.group(1)
        return dict(cond=f"今日の終値が{k}営業日前の終値より高い", entry="その日の引けで買う", exit="翌営業日の引けで売る",
                    idea="上がっているものは上がり続けるか（モメンタム）", side="買い")
    m = re.match(r"逆張り:過去(\d+)日下落", name)
    if m:
        k = m.group(1)
        return dict(cond=f"今日の終値が{k}営業日前の終値より安い", entry="その日の引けで買う", exit="翌営業日の引けで売る",
                    idea="下がったものは戻るか（平均回帰）", side="買い")
    if name.startswith("逆張り:RSI2<10→翌日"):
        return dict(cond="2日RSIが10未満（直近2日で大きく売られた）", entry="その日の引けで買う", exit="翌営業日の引けで売る",
                    idea="短期の売られすぎからの反発", side="買い")
    if name.startswith("逆張り:RSI2<10→5日"):
        return dict(cond="2日RSIが10未満（直近2日で大きく売られた）", entry="その日の引けで買う", exit="5営業日後の引けで売る",
                    idea="短期の売られすぎからの反発を数日かけて取る", side="買い")
    if name.startswith("逆張り:RSI2>90"):
        return dict(cond="2日RSIが90超（直近2日で大きく買われた）", entry="その日の引けで空売りする", exit="翌営業日の引けで買い戻す",
                    idea="短期の買われすぎからの反落", side="売り")
    if name.startswith("ギャップ:-1%"):
        return dict(cond="寄り付きが前日終値より1%以上安く始まった", entry="その日の引けで買う", exit="翌営業日の引けで売る",
                    idea="下に窓を開けた後の戻り", side="買い")
    if name.startswith("ギャップ:+1%"):
        return dict(cond="寄り付きが前日終値より1%以上高く始まった", entry="その日の引けで空売りする", exit="翌営業日の引けで買い戻す",
                    idea="上に窓を開けた後の反落", side="売り")
    if "オーバーナイト" in name:
        return dict(cond="毎営業日（条件なし）", entry="引けで買う", exit="翌営業日の寄り付きで売る",
                    idea="値上がりが夜間（取引時間外）に偏っているか", side="買い")
    if "ザラ場のみ" in name:
        return dict(cond="毎営業日（条件なし）", entry="寄り付きで買う", exit="同じ日の引けで売る",
                    idea="値上がりが取引時間中に偏っているか", side="買い")
    if name.startswith("暦:月末"):
        return dict(cond="月末の最後の2営業日、または月初の3営業日", entry="その日の引けで買う", exit="翌営業日の引けで売る",
                    idea="月末月初は資金流入で上がりやすいか（月替わり効果）", side="買い")
    if name.startswith("レジーム:低ボラ"):
        return dict(cond="直近20日の値動きの大きさが、それまでの全期間の下位3分の1", entry="その日の引けで買う",
                    exit="翌営業日の引けで売る", idea="落ち着いた相場は上がりやすいか", side="買い")
    if name.startswith("レジーム:高ボラ"):
        return dict(cond="直近20日の値動きの大きさが、それまでの全期間の上位3分の1", entry="その日の引けで買う",
                    exit="翌営業日の引けで売る", idea="荒れた相場は上がりやすいか", side="買い")
    if name.startswith("出来高:"):
        return dict(cond="出来高が直前20日平均の2倍超、かつ前日比マイナスで引けた", entry="その日の引けで買う",
                    exit="翌営業日の引けで売る", idea="大商いの投げ売りの後に戻るか", side="買い")
    if name.startswith("逆張り:3日連続"):
        return dict(cond="3営業日続けて前日比マイナスで引けた", entry="その日の引けで買う", exit="翌営業日の引けで売る",
                    idea="連続安の後の反発", side="買い")
    if name.startswith("順張り:52週高値"):
        return dict(cond="終値が過去1年（252営業日）の最高値から2%以内", entry="その日の引けで買う", exit="5営業日後の引けで売る",
                    idea="高値圏の銘柄はさらに上がるか（高値ブレイク）", side="買い")
    m = re.match(r"時間帯:(\d\d):(\d\d)台", name)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        e = f"{h + (mi + 30) // 60:02d}:{(mi + 30) % 60:02d}"
        return dict(cond="毎営業日（条件なし）", entry=f"{h:02d}:{mi:02d} の5分足の始値で買う", exit=f"{e} 直前の5分足の終値で売る",
                    idea="特定の時間帯に値上がりが偏っているか", side="買い")
    if name.startswith("分足:寄り30分が上昇"):
        return dict(cond="寄り付きから30分間の値動きがプラス", entry="その次の5分足の終値で買う", exit="同じ日の引けで売る",
                    idea="朝の勢いが一日続くか", side="買い")
    if name.startswith("分足:寄り30分が下落"):
        return dict(cond="寄り付きから30分間の値動きがマイナス", entry="その次の5分足の終値で空売りする", exit="同じ日の引けで買い戻す",
                    idea="朝の下げが一日続くか", side="売り")
    if name.startswith("分足:日中が上昇"):
        return dict(cond="寄り付きから引け1時間前までの値動きがプラス", entry="引け1時間前の5分足の終値で買う",
                    exit="同じ日の引けで売る", idea="日中の流れが引けまで続くか", side="買い")
    if name.startswith("分足:0.5%超の下窓"):
        return dict(cond="寄り付きが前日終値より0.5%以上安く始まった", entry="寄り直後（2本目の5分足の終値）で買う",
                    exit="同じ日の引けで売る", idea="窓埋め（下に開けた窓が当日中に戻るか）", side="買い")
    return dict(cond="—", entry="—", exit="—", idea="—", side="買い")
