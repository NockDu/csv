import os
import torch
import joblib
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as transformers_logging
from transformers.utils import logging as utils_logging  # 👈 진행 바 완전 차단을 위해 추가
import sys
import io

# 터미널 출력 설정 (윈도우 한글 깨짐 및 이모지 에러 방지)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# 🛑 [강력 차단] Hugging Face 라이브러리 및 하위 내부 로그 완전 음소거
transformers_logging.set_verbosity_error()
utils_logging.set_verbosity_error()  # 👈 허깅페이스 내부 유틸리티 로그 차단
utils_logging.disable_progress_bar()  # 👈 100%|██████████| 이 진행 바를 원천 차단하는 핵심 필살기
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# ==========================================
# [경로 설정] 코치님의 로컬 파일 경로
# ==========================================
RF_SMS_MODEL = "C:/Users/alang/Downloads/model/sms(RF)/sms_randomforest_model/sms_phishing_model.pkl"
RF_SMS_VEC = "C:/Users/alang/Downloads/model/sms(RF)/sms_randomforest_model/sms_tfidf_vectorizer.pkl"
RF_CALL_MODEL = "C:/Users/alang/Downloads/model/call(RF)/call_randomforest_model/call_phishing_model.pkl"
RF_CALL_VEC = "C:/Users/alang/Downloads/model/call(RF)/call_randomforest_model/call_tfidf_vectorizer.pkl"

KE_SMS_DIR = "C:/Users/alang/Downloads/model/sms(KE)/sms_koelectra_model"
KE_CALL_DIR = "C:/Users/alang/Downloads/model/call(KE)/call_koelectra_model"

def test_random_forest(text, model_path, vec_path, mode_name):
    try:
        if not os.path.exists(model_path) or not os.path.exists(vec_path):
            return f"   [{mode_name}] 파일 없음 - 패스"
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        
        vectorized_text = vectorizer.transform([text])
        prediction = model.predict(vectorized_text)[0]
        prob = model.predict_proba(vectorized_text)[0][1]
        
        result = "피싱 위험" if prediction == 1 else "정상 대화"
        return f"   └─ {mode_name} 결과: {result} (피싱 확률: {prob*100:.2f}%)"
    except Exception as e:
        return f"   └─ [{mode_name}] 에러: {e}"

def test_koelectra(text, model_dir, mode_name):
    try:
        if not os.path.exists(model_dir):
            return f"   [{mode_name}] 폴더 없음 - 패스"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # 로드할 때 발생하는 내부 tqdm 락을 완전히 해제
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        model.to(device)
        model.eval()
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
            prob = torch.softmax(logits, dim=-1)[0][1].item()
            
        result = "피싱 위험" if prediction == 1 else "정상 대화"
        return f"   └─ {mode_name} 결과: {result} (피싱 확률: {prob*100:.2f}%)"
    except Exception as e:
        return f"   └─ [{mode_name}] 에러: {e}"

# ==========================================
# 🛑 깔끔한 출력 전용 메인 프로세스
# ==========================================
if __name__ == "__main__":
    # 터미널 창을 깨끗하게 정리하기 위해 강제 줄바꿈 한 번 더 주입
    print("\n" + "="*70)
    print(" [로컬 인퍼런스] 4대 피싱 탐지 모델 실전 스크리닝")
    print("="*70)
    
    # 가상의 테스트 데이터 입력
    sample_sms = "오늘 저녁에 뭐 먹을까? 삼겹살 어때?"
    sample_call = "아 어머님 안녕하세요 저 철수 친구 영민이인데요. 다름이 아니라 오늘 저녁에 저희 동아리 애들이랑 다 같이 저녁 먹기로 했거든요. 메뉴는 삼겹살로 정했는데 어머님 혹시 오늘 저녁에 철수 집에서 밥 먹고 자고 가도 괜찮을까 해서 전화드렸어요. 저희가 맛있는 거 사서 들어갈게요. 이따 저녁에 철수랑 같이 집으로 가겠습니다 감사합니다 어머님."
    
    # 1. SMS 모델 결과 출력
    print(f"\n[테스트 SMS 주입] : \"{sample_sms}\"")
    print("-" * 70)
    print(test_random_forest(sample_sms, RF_SMS_MODEL, RF_SMS_VEC, "RF 스미싱 모델"))
    print(test_koelectra(sample_sms, KE_SMS_DIR, "KoELECTRA 스미싱 모델"))
    print("-" * 70)
    
    # 2. CALL 모델 결과 출력
    print(f"\n[테스트 CALL 주입] : \"{sample_call}\"")
    print("-" * 70)
    print(test_random_forest(sample_call, RF_CALL_MODEL, RF_CALL_VEC, "RF 보이스피싱 모델"))
    print(test_koelectra(sample_call, KE_CALL_DIR, "KoELECTRA 보이스피싱 모델"))
    print("-" * 70)
    print("\n" + "="*70 + "\n")