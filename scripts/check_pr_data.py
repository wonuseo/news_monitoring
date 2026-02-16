import os
import pandas as pd
from dotenv import load_dotenv
from src.modules.export.sheets import connect_sheets

load_dotenv()
creds_path = os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH')
sheet_id = os.getenv('GOOGLE_SHEET_ID')

spreadsheet = connect_sheets(creds_path, sheet_id)
worksheet = spreadsheet.worksheet('total_result')
records = worksheet.get_all_records()
df = pd.DataFrame(records)

# 보도자료 분석
pr_df = df[df['source'] == '보도자료']
print(f'총 보도자료: {len(pr_df)}개')
print()

# cluster_id 유무
has_cluster = (pr_df['cluster_id'].notna()) & (pr_df['cluster_id'] != '')
pr_with_cluster = pr_df[has_cluster]
pr_without_cluster = pr_df[~has_cluster]

print(f'cluster_id 있음: {len(pr_with_cluster)}개')
print(f'cluster_id 없음: {len(pr_without_cluster)}개')
print()

# cluster_id 있는 보도자료의 고유 cluster_id 개수
unique_clusters = pr_with_cluster['cluster_id'].nunique()
print(f'고유 cluster_id 개수: {unique_clusters}개')
print()

# cluster_id 있는 보도자료 중 부정 기사 확인
negative = pr_with_cluster[pr_with_cluster['sentiment_stage'].isin(['부정 후보', '부정 확정'])]
print(f'cluster_id 있음 + 부정: {len(negative)}개')
if len(negative) > 0:
    print('\n부정 확정/후보 보도자료 샘플:')
    for idx, row in negative.head(5).iterrows():
        title = str(row.get('title', ''))[:60]
        print(f'  - {title}...')
        print(f'    cluster={row.get("cluster_id")}, sentiment={row.get("sentiment_stage")}')
        print()
