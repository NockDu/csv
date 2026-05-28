import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

# 1. 폰트 에러 방지 및 디바이스(GPU) 설정
plt.rcdefaults()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 학습에 사용하는 장치: {device}")

# 2. 데이터셋 로드
print("\n1. 증강 및 분할 완료된 전용 데이터를 로드합니다...")
train_df = pd.read_csv('/kaggle/input/datasets/nockdu/voicephishing/train_phone.csv')
test_df = pd.read_csv('/kaggle/input/datasets/nockdu/voicephishing/test_phone.csv')

# 데이터에 빈 값이 있을 경우를 대비한 방어 코드
train_df['text'] = train_df['text'].fillna("")
test_df['text'] = test_df['text'].fillna("")

train_texts = train_df['text'].tolist()
train_labels = train_df['label'].tolist()
test_texts = test_df['text'].tolist()
test_labels = test_df['label'].tolist()

# 3. KoELECTRA 토크나이저 및 모델 로드
# 한국어 구어체와 피싱 도메인 문맥 파악에 가장 범용적이고 탁월한 monologg 모델을 사용합니다.
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
print(f"\n2. 사전 학습된 {MODEL_NAME} 모델 및 토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# 4. 토큰화(Tokenization) 및 데이터셋 커스텀 클래스 정의
print("\n3. 데이터를 KoELECTRA가 먹을 수 있는 토큰 형태로 변환 중...")
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

# 긴 통화 데이터를 감안하여 최대 토큰 길이를 256으로 설정 (필요시 512까지 확대 가능)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)

train_dataset = PhishingDataset(train_encodings, train_labels)
test_dataset = PhishingDataset(test_encodings, test_labels)

# 5. 검증 지표 정의 (매 에폭마다 기록될 평가지표)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

# 6. Trainer 환경 및 하이퍼파라미터 설정
print("\n4. 훈련 하이퍼파라미터 세팅 중...")
training_args = TrainingArguments(
    output_dir='/kaggle/working/results',          # 체크포인트 저장 폴더
    num_train_epochs=3,                            # 딱 3번만 돌리기 (과적합 방지)
    per_device_train_batch_size=16,                # GPU 메모리 상황에 맞춘 배치 크기
    per_device_eval_batch_size=16,
    learning_rate=2e-5,                            # 요청하신 정밀 정석 학습률 0.00002
    weight_decay=0.01,
    logging_dir='/kaggle/working/logs',
    logging_steps=100,
    eval_strategy="epoch",                         # 매 바퀴(에폭)가 끝날 때마다 테스트 점수 확인
    save_strategy="epoch",                         # 에폭마다 모델 저장해서 골라 잡을 수 있게 함
    load_best_model_at_end=True,                   # 학습 종료 시 가장 성적이 좋았던 에폭 모델 로드
    metric_for_best_model="accuracy",
    report_to="none"                               # 불필요한 외부 툴 연동 차단
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# 7. 파인 튜닝 시작
print("\n5. KoELECTRA 파인 튜닝(Fine-tuning)을 시작합니다!")
trainer.train()

# 8. 최종 성능 평가 및 보고서 생성 (테스트셋 기준)
print("\n6. 최종 보이스피싱 모델 평가 및 성능 측정 중...")
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(-1)

# ROC Curve를 위한 확률값 계산 (Softmax 적용)
raw_logits = torch.tensor(predictions.predictions)
y_prob = torch.softmax(raw_logits, dim=-1)[:, 1].numpy()

print("\n" + "="*50)
print(f"KoELECTRA 최종 모델 정확도 (Accuracy): {accuracy_score(test_labels, y_pred):.4f}")
print("="*50)
print("\n[상세 성능 보고서 (Detailed Performance Report)]")
print(classification_report(test_labels, y_pred))

# 9. 시각화 (Confusion Matrix & ROC Curve)
plt.figure(figsize=(12, 5))

# (1) Confusion Matrix
plt.subplot(1, 2, 1)
cm = confusion_matrix(test_labels, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Normal', 'Phishing'], yticklabels=['Normal', 'Phishing'])
plt.title('KoELECTRA Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# (2) ROC Curve
plt.subplot(1, 2, 2)
fpr, tpr, _ = roc_curve(test_labels, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('KoELECTRA ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig('/kaggle/working/koelectra_performance_plot.png')
plt.show()

# 10. 서버 배포용 최종 모델 및 토크나이저 저장
print("\n7. 서버 배포용 최종 모델 및 토크나이저 영구 저장 중...")
model_save_path = "/kaggle/working/best_koelectra_model"
tokenizer.save_pretrained(model_save_path)
trainer.save_model(model_save_path)

print(f"\n[작업 완료] {model_save_path} 폴더에 모델이 무사히 보관되었습니다.")
print(os.listdir('/kaggle/working'))