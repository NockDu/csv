import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 1. 폰트 에러 방지 및 디바이스(GPU) 설정
plt.rcdefaults()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 학습에 사용하는 장치: {device}")

# 2. 스미싱(SMS) 데이터셋 로드
print("\n1. 증강 및 분할 완료된 스미싱 전용 데이터를 로드합니다...")
# RF 코드에서 사용하신 스미싱 경로를 그대로 매핑했습니다.
train_df = pd.read_csv('/kaggle/input/datasets/nockdu/for-sms/train_sms.csv')
test_df = pd.read_csv('/kaggle/input/datasets/nockdu/for-sms/test_sms.csv')

# 결측치 방어 코드
train_df['text'] = train_df['text'].fillna("")
test_df['text'] = test_df['text'].fillna("")

train_texts = train_df['text'].tolist()
train_labels = train_df['label'].tolist()
test_texts = test_df['text'].tolist()
test_labels = test_df['label'].tolist()

# 3. KoELECTRA 토크나이저 및 모델 로드
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
print(f"\n2. 사전 학습된 {MODEL_NAME} 모델 및 토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# 4. 토큰화(Tokenization) 및 데이터셋 커스텀 클래스 정의
print("\n3. 스미싱 데이터를 KoELECTRA가 먹을 수 있는 토큰 형태로 변환 중...")
class PhishingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 스미싱(문자)은 전화 데이터보다 상대적으로 짧으므로 max_length=128로 최적화하여 속도를 올립니다.
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

train_dataset = PhishingDataset(train_encodings, train_labels)
test_dataset = PhishingDataset(test_encodings, test_labels)

# 5. 검증 지표 정의
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

# 6. Trainer 환경 및 하이퍼파라미터 설정 (정석 세팅 유지)
print("\n4. 훈련 하이퍼파라미터 세팅 중...")
training_args = TrainingArguments(
    output_dir='/kaggle/working/sms_results',       # 스미싱 전용 체크포인트 폴더
    num_train_epochs=3,                            # 과적합 방지를 위한 딱 3 에폭
    per_device_train_batch_size=16,                # 안정적인 배치 크기
    per_device_eval_batch_size=16,
    learning_rate=2e-5,                            # 정밀 정석 학습률 0.00002
    weight_decay=0.01,
    logging_dir='/kaggle/working/sms_logs',
    logging_steps=50,                              # 데이터수가 전화보다 적으므로 더 자주 로깅
    eval_strategy="epoch",                         # 매 바퀴마다 테스트 점수 확인
    save_strategy="epoch",                         # 에폭마다 모델 백업
    load_best_model_at_end=True,                   # 가장 똑똑했던 회차의 모델을 최종 선택
    metric_for_best_model="accuracy",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# 7. 파인 튜닝 시작
print("\n5. 스미싱 탐지용 KoELECTRA 파인 튜닝(Fine-tuning)을 시작합니다!")
trainer.train()

# 8. 최종 성능 평가 및 보고서 생성 (테스트셋 기준)
print("\n6. 최종 스미싱 모델 평가 및 성능 측정 중...")
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(-1)

# ROC Curve용 확률값 계산
raw_logits = torch.tensor(predictions.predictions)
y_prob = torch.softmax(raw_logits, dim=-1)[:, 1].numpy()

print("\n" + "="*50)
print(f"KoELECTRA 스미싱 최종 모델 정확도 (Accuracy): {accuracy_score(test_labels, y_pred):.4f}")
print("="*50)
print("\n[상세 성능 보고서 (Detailed Performance Report)]")
print(classification_report(test_labels, y_pred))

# 9. 시각화 (Confusion Matrix & ROC Curve)
plt.figure(figsize=(12, 5))

# (1) Confusion Matrix (시각적 구분을 위해 Purples 테마 사용)
plt.subplot(1, 2, 1)
cm = confusion_matrix(test_labels, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Normal', 'Phishing'], yticklabels=['Normal', 'Phishing'])
plt.title('KoELECTRA SMS Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# (2) ROC Curve
plt.subplot(1, 2, 2)
fpr, tpr, _ = roc_curve(test_labels, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='purple', lw=2, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('KoELECTRA SMS ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig('/kaggle/working/sms_koelectra_performance.png') # 아웃풋 파일 분리
plt.show()

# 10. 서버 배포용 최종 스미싱 모델 및 토크나이저 저장
print("\n7. 서버 배포용 최종 스미싱 모델 및 토크나이저 영구 저장 중...")
model_save_path = "/kaggle/working/best_sms_koelectra_model" # 폴더명 명확히 분리
tokenizer.save_pretrained(model_save_path)
trainer.save_model(model_save_path)

print(f"\n[작업 완료] {model_save_path} 폴더에 스미싱 모델이 무사히 보관되었습니다.")
print("\n[현재 아웃풋 폴더 전체 파일 목록]")
print(os.listdir('/kaggle/working'))