import pandas as pd
import numpy as np
import re
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

# 1. 환경 설정 및 폰트 에러 방지
plt.rcdefaults()

def preprocess_text(text):
    # URL 마스킹 (혹시 모를 링크 대비)
    text = re.sub(r'http[s]?://\S+', '<URL>', str(text))
    # 대화 중 불필요한 특수문자 및 줄바꿈 제거
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
    text = text.replace('\n', ' ').strip()
    return text

print("1. 보이스피싱(Phone) 데이터 로드 및 전처리 중...")
# 파일 경로 (사용자 환경에 맞게 수정 가능)
train_df = pd.read_csv('/kaggle/input/datasets/nockdu/voicephishing/train_phone.csv')
test_df = pd.read_csv('/kaggle/input/datasets/nockdu/voicephishing/test_phone.csv')
weights_df = pd.read_csv('/kaggle/input/datasets/nockdu/voicephishing/500_.csv') 

train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

# 2. 보이스피싱 특화 노이즈 제거 및 단어장 설정
# 전화 데이터 특유의 추임새와 무의미한 명사들을 추가했습니다.
noise_words = [
    '진짜', '근데', '너무', '많이', '우리', '그냥', '이제', '그거', '요즘', '그러니까', 
    '키키', 'ㅠㅠ', 'ㅋㅋ', '나도', '나는', '그건', '그래서', '그렇게', '같이', '무슨', 
    '그게', '보고', '한번', '거지', '사람', '어떻게', '지금', '어디', '부분', '생각', 
    '말씀', '통화', '저기', '아니', '내용', '안녕하세요', '상담원', '상담사', '감사합니다',
    '이었습니다', '반갑습니다', '하루', '문의', '도와드릴까요', '수고하세요', '본인',
    '입니다,' '도움'
]

# 선배님의 500개 단어 중 명사(Noun)만 추출하여 학습 범위 한정
final_vocabulary = weights_df[
    (weights_df['형태소'] == 'Noun') & 
    (~weights_df['단어'].isin(noise_words))
]['단어'].tolist()

# 3. TF-IDF 벡터화 및 Random Forest 학습
print(f"2. {len(final_vocabulary)}개의 핵심 키워드로 보이스피싱 전용 모델 학습 시작...")
vectorizer = TfidfVectorizer(vocabulary=final_vocabulary)

X_train = vectorizer.fit_transform(train_df['text'])
y_train = train_df['label']
X_test = vectorizer.transform(test_df['text'])
y_test = test_df['label']

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 4. 성능 측정 및 상세 보고서 출력
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print(f"최종 모델 정확도 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("="*50)
print("\n[상세 성능 보고서 (Detailed Performance Report)]")
print(classification_report(y_test, y_pred))

# 5. 시각화 (Confusion Matrix & ROC Curve)
plt.figure(figsize=(12, 5))

# (1) Confusion Matrix
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Normal', 'Phishing'], yticklabels=['Normal', 'Phishing'])
plt.title('Confusion Matrix (Voice Phishing)')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# (2) ROC Curve
plt.subplot(1, 2, 2)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig('/kaggle/working/phone_model_performance.png')
plt.show()

# 6. 서버 배포용 모델 저장
print("\n5. 보이스피싱 전용 모델 및 벡터라이저 저장 중...")
joblib.dump(model, '/kaggle/working/phone_phishing_model.pkl')
joblib.dump(vectorizer, '/kaggle/working/phone_tfidf_vectorizer.pkl')

print("\n[작업 완료] 아래 파일들이 생성되었습니다:")
print(os.listdir('/kaggle/working'))