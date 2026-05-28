import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import io

# 터미널 출력 설정 (윈도우 한글 깨짐 및 이모지 에러 방지)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# 1. 데이터 불러오기
alert_df = pd.read_csv('alert_augmented_2680.csv') # 피싱 데이터 (2680개, label=1)
chat_df = pd.read_csv('chat.csv')            # 정상 데이터 (2680개, label=0)

# 2. 고정할 테스트 데이터 개수 설정
# 전체 테스트용 360개 = 피싱 180개 + 정상 180개
total_test_size = 360
half_test_size = total_test_size // 2  # 180

# 3. 피싱 데이터와 정상 데이터를 각각 분할 (비율이 아닌 고정 개수로!)
# - test_size에 정수(180)를 넣으면 그 개수만큼만 딱 뽑아줍니다.
alert_train, alert_test = train_test_split(
    alert_df, 
    test_size=half_test_size, 
    random_state=42, 
    shuffle=True
)

chat_train, chat_test = train_test_split(
    chat_df, 
    test_size=half_test_size, 
    random_state=42, 
    shuffle=True
)

# 4. 각각 쪼갠 데이터들을 학습용끼리, 테스트용끼리 다시 합치기
train_df = pd.concat([alert_train, chat_train], ignore_index=True)
test_df = pd.concat([alert_test, chat_test], ignore_index=True)

# 5. 마지막으로 모델이 순서를 외우지 못하게 무작위로 섞어주기
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. 최종 파일 저장 (utf-8-sig로 한글 깨짐 방지)
train_df.to_csv('train_sms.csv', index=False, encoding='utf-8-sig')
test_df.to_csv('test_sms.csv', index=False, encoding='utf-8-sig')

print("="*50)
print("🔥 KoELECTRA용 훈련/테스트 세트 구축 완료!")
print("="*50)
print(f"전체 데이터: 피싱 {len(alert_df)}개 + 정상 {len(chat_df)}개 = 총 {len(alert_df)+len(chat_df)}개")
print(f"테스트 데이터 (test_sms.csv) : {len(test_df)}개 (정상: {len(test_df[test_df['label']==0])}개 / 피싱: {len(test_df[test_df['label']==1])}개)")
print(f"훈련 데이터 (train_sms.csv)   : {len(train_df)}개 (정상: {len(train_df[train_df['label']==0])}개 / 피싱: {len(train_df[train_df['label']==1])}개)")
print("="*50)