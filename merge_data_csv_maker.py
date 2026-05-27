import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import io

# 터미널 출력 설정 (윈도우 한글 깨짐 및 이모지 에러 방지)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# 1. 각각의 데이터 불러오기
phishing_df = pd.read_csv("phone_augmented_6950.csv")     # 피싱 (Label 1)
normal_df = pd.read_csv("final_balanced_normal_data.csv") # 정상 (Label 0)

# 2. 고정할 테스트 데이터 개수 설정
# 전체 테스트용 1900개 = 피싱 950개 + 정상 950개
total_test_size = 1900
half_test_size = total_test_size // 2  # 950

# 3. 클래스별로 정확히 950개씩 도려내기 (비율 꼬임 완전 차단)
phishing_train, phishing_test = train_test_split(
    phishing_df, 
    test_size=half_test_size, 
    random_state=42, 
    shuffle=True
)

normal_train, normal_test = train_test_split(
    normal_df, 
    test_size=half_test_size, 
    random_state=42, 
    shuffle=True
)

# 4. 각각 쪼갠 데이터들을 학습용끼리, 테스트용끼리 병합
train_df = pd.concat([phishing_train, normal_train], ignore_index=True)
test_df = pd.concat([phishing_test, normal_test], ignore_index=True)

# 5. 모델이 정답 순서를 외우지 못하게 무작위 셔플
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. 최종 CSV 파일로 저장 (utf-8-sig로 한글 깨짐 방지)
train_df.to_csv("train_phone.csv", index=False, encoding='utf-8-sig')
test_df.to_csv("test_phone.csv", index=False, encoding='utf-8-sig')

print("="*50)
print("KoELECTRA용 보이스피싱(통화) 데이터셋 구축 완료!")
print("="*50)
print(f"전체 원본 데이터: 피싱 {len(phishing_df)}개 + 정상 {len(normal_df)}개 = 총 {len(phishing_df)+len(normal_df)}개")
print(f"테스트 데이터 (test_phone.csv) : {len(test_df)}개 (정상: {len(test_df[test_df['label']==0])}개 / 피싱: {len(test_df[test_df['label']==1])}개)")
print(f"훈련 데이터 (train_phone.csv)   : {len(train_df)}개 (정상: {len(train_df[train_df['label']==0])}개 / 피싱: {len(train_df[train_df['label']==1])}개)")
print("="*50)