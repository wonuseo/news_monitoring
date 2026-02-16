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

# 롯데호텔_c00001 클러스터 확인
cluster_df = df[df['cluster_id'] == '롯데호텔_c00001'].copy()

print(f'롯데호텔_c00001 클러스터: {len(cluster_df)}개 기사')
print()

# pub_datetime으로 정렬 (첫 번째 기사가 대표)
cluster_df['pub_datetime_parsed'] = pd.to_datetime(cluster_df['pub_datetime'], errors='coerce')
cluster_df = cluster_df.sort_values('pub_datetime_parsed')

print('클러스터 내 기사 (날짜순):')
for idx, row in cluster_df.iterrows():
    title = str(row.get('title', ''))[:60]
    sentiment = row.get('sentiment_stage', '')
    brand_rel = row.get('brand_relevance', '')
    date = row.get('pub_datetime', '')[:10]
    print(f'{date} | {sentiment:10s} | {brand_rel:6s} | {title}...')

print()
print(f'대표 기사 (첫 번째): sentiment={cluster_df.iloc[0]["sentiment_stage"]}, brand_relevance={cluster_df.iloc[0]["brand_relevance"]}')
