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

# 폰트 에러 방지 (영문으로 출력 설정)
plt.rcdefaults()

def preprocess_text(text):
    text = re.sub(r'http[s]?://\S+', '<URL>', str(text))
    text = text.replace('\n', ' ').strip()
    return text

print("1. 데이터를 불러오는 중...")
train_df = pd.read_csv('/kaggle/input/datasets/nockdu/for-sms/train_sms.csv')
test_df = pd.read_csv('/kaggle/input/datasets/nockdu/for-sms/test_sms.csv')
weights_df = pd.read_csv('/kaggle/input/datasets/nockdu/for-sms/500_.csv') 

train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

# 노이즈 제거 및 단어장 설정
noise_words = [
    '진짜', '근데', '너무', '많이', '우리', '그냥', '이제', '그거', '요즘', '그러니까', 
    '키키', 'ㅠㅠ', 'ㅋㅋ', '나도', '나는', '그건', '그래서', '그렇게', '같이', '무슨', 
    '그게', '보고', '한번', '거지', '사람', '어떻게', '지금', '어디'
]

final_vocabulary = weights_df[
    (weights_df['형태소'] == 'Noun') & 
    (~weights_df['단어'].isin(noise_words))
]['단어'].tolist()

# 벡터화 및 학습
vectorizer = TfidfVectorizer(vocabulary=final_vocabulary, stop_words=noise_words)
X_train = vectorizer.fit_transform(train_df['text'])
y_train = train_df['label']
X_test = vectorizer.transform(test_df['text'])
y_test = test_df['label']

print(f"2. {len(final_vocabulary)}개의 핵심 명사로 학습 시작...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 4. 성능 측정 및 보고서 출력 (빠졌던 부분!) ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print(f"최종 모델 정확도 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("="*50)
print("\n[상세 성능 보고서 (Detailed Performance Report)]")
print(classification_report(y_test, y_pred)) # 이 줄이 빠져서 안 나왔던 겁니다!

# --- 시각화 (영어 라벨로 폰트 에러 방지) ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Phishing'], yticklabels=['Normal', 'Phishing'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig('/kaggle/working/performance_plot.png') # 아웃풋에 확실히 저장
plt.show()

# --- 5. 모델 저장 (PKL 파일) ---
print("\n5. 모델 및 벡터라이저를 저장 중...")
joblib.dump(model, '/kaggle/working/phishing_model.pkl')
joblib.dump(vectorizer, '/kaggle/working/tfidf_vectorizer.pkl')

# 파일이 잘 저장되었는지 목록 확인
print("\n[Output 폴더 파일 목록 확인]")
print(os.listdir('/kaggle/working'))