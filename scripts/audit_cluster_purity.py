"""Audit: 클러스터 내 press_release_group 불순도 확인"""
import os
import pandas as pd
from dotenv import load_dotenv
from src.modules.export.sheets import connect_sheets

load_dotenv('../.env')
creds_path = os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH')
sheet_id = os.getenv('GOOGLE_SHEET_ID')
spreadsheet = connect_sheets(creds_path, sheet_id)
worksheet = spreadsheet.worksheet('total_result')
records = worksheet.get_all_records()
df = pd.DataFrame(records)

# 클러스터가 있는 기사만
clustered = df[df['cluster_id'].astype(str).str.strip().ne('')].copy()
print(f'전체: {len(df)}개, 클러스터: {len(clustered)}개')
print()

# cluster_id별 press_release_group 종류 수
cluster_prg = clustered.groupby('cluster_id').agg(
    count=('title', 'size'),
    prg_unique=('press_release_group', 'nunique'),
    prg_list=('press_release_group', lambda x: list(x.unique())),
    source=('source', 'first'),
).reset_index()

# 불순 클러스터 (press_release_group 2개 이상)
impure = cluster_prg[cluster_prg['prg_unique'] > 1].sort_values('count', ascending=False)
pure = cluster_prg[cluster_prg['prg_unique'] == 1]

print(f'순수 클러스터 (PRG 1종류): {len(pure)}개')
print(f'불순 클러스터 (PRG 2+종류): {len(impure)}개')
print(f'불순 클러스터 내 총 기사: {impure["count"].sum()}개')
print()

for _, row in impure.iterrows():
    cid = row['cluster_id']
    print(f'--- {cid} ({row["count"]}개, source={row["source"]}) ---')
    for prg in row['prg_list']:
        sub = clustered[(clustered['cluster_id'] == cid) & (clustered['press_release_group'] == prg)]
        print(f'  [{len(sub)}개] "{prg}"')
        for _, a in sub.head(2).iterrows():
            dt = str(a.get('pub_datetime', '?'))[:10]
            title = str(a.get('title', '?'))[:70]
            print(f'       {dt} | {title}')
    print()

# news_category 불순도도 확인
print('=' * 60)
print('news_category 불순도 확인')
print('=' * 60)
cat_agg = clustered.groupby('cluster_id').agg(
    count=('title', 'size'),
    cat_unique=('news_category', 'nunique'),
    cat_list=('news_category', lambda x: list(x.value_counts().items())),
).reset_index()

cat_impure = cat_agg[cat_agg['cat_unique'] > 1].sort_values('count', ascending=False)
print(f'news_category 불순 클러스터: {len(cat_impure)}개')
for _, row in cat_impure.head(10).iterrows():
    print(f'  {row["cluster_id"]} ({row["count"]}개): {row["cat_list"]}')
