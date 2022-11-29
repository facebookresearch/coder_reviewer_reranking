# Copyright (c) Meta Platforms, Inc. and affiliates.

from tqdm import tqdm
import os
import sqlite3
import pickle as pkl

# CONSTANT
db_dir = "./dataset/spider/database/"
# preloading spider data to reduce io
from dataset.spider_official.evaluation import (
    build_foreign_key_map_from_json,
    build_valid_col_units,
    rebuild_sql_val,
    rebuild_sql_col,
)
from dataset.spider_official.process_sql import (
    get_schema,
    Schema,
    get_sql,
)

kmaps = build_foreign_key_map_from_json("./dataset/spider/tables.json")
with open("dataset/spider/dev_gold.sql") as f:
    glist = [l.strip().split("\t") for l in f.readlines() if len(l.strip()) > 0]


all_g_res = []
for gold_sql in tqdm(glist, total=len(glist)):
    g_str, db = gold_sql
    db_name = db
    db = os.path.join(db_dir, db, db + ".sqlite")
    schema = Schema(get_schema(db))
    g_sql = get_sql(schema, g_str)
    kmap = kmaps[db_name]
    g_valid_col_units = build_valid_col_units(g_sql["from"]["table_units"], schema)
    g_sql = rebuild_sql_val(g_sql)
    g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cursor = conn.cursor()
    # there are potential utf-8 errors
    try:
        cursor.execute(g_str)
        g_res = cursor.fetchall()
    except:
        g_res = []

    def res_map(res, val_units):
        rmap = {}
        for idx, val_unit in enumerate(val_units):
            key = (
                tuple(val_unit[1])
                if not val_unit[2]
                else (val_unit[0], tuple(val_unit[1]), tuple(val_unit[2]))
            )
            rmap[key] = [r[idx] for r in res]
        return rmap

    g_val_units = [unit[1] for unit in g_sql["select"][1]]
    g_res = res_map(g_res, g_val_units)
    all_g_res.append(g_res)

pkl.dump(
    all_g_res,
    open(
        "./dataset/spider/cached_gold_results.pkl",
        "wb",
    ),
)
