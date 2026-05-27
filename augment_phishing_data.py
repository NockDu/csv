import os
import glob
import re
import random
import pandas as pd

# ==========================================
# 1. 설정 매개변수 (코치님 환경에 맞게 수정)
# ==========================================
TXT_FOLDER_PATH = "C:/Users/alang/Downloads/my_data(가공 후)/전화/forcut"  # 495개 txt 파일이 모여있는 폴더 경로
OUTPUT_CSV_PATH = "C:/Users/alang/Downloads/my_data(가공 후)/전화/phone_augmented_6950.csv"
TARGET_ROW_COUNT = 6950  # 목표로 하는 최종 피싱 데이터 개수
MAX_TOKEN_CHARS = 250   # KoELECTRA가 먹기 좋게 자를 글자 수 글자 제한 (약 200~250자)

# 2. 증강용 도메인 사전 키워드 (선배님 가중치 파일 기반 변형 팩터)
banks = ['국민은행', '신한은행', '우리은행', '하나은행', '농협은행', '기업은행', '카카오뱅크', '토스뱅크']
govs = ['서울중앙지검', '검찰청', '금융감독원', '법원', '경찰청', '지검', '수사과']
roles = ['김민수 수사관', '이현우 검사', '박과장', '최팀장', '담당 검사']

# 교수님 방어용 CS 말투 인젝션 셋
cs_prefixes = [
    "안녕하세요 고객님, 무엇을 도와드릴까요? 아 네, ",
    "안녕하세요 코치님, OOO 은행 상담원입니다. 다름이 아니라 ",
    "[안내] 고객님 금융 안전 보증 센터입니다. ",
    "네 안녕하십니까 회원님, 문의하신 내역 확인차 연락드렸습니다. ",
    "안녕하세요, "
]

# STT 음성인식 특유의 말더듬 노이즈 셋
stt_noises = [" 어... ", " 그... ", " 저기.. 네, ", " 그러니까 ", " 예 예, ", " 아, "]

def load_and_merge_txts(folder_path):
    """495개 txt 파일을 읽어 하나의 거대한 텍스트 리스트로 병합"""
    file_list = glob.glob(os.path.join(folder_path, "*.txt"))
    if not file_list:
        raise FileNotFoundError(f"'{folder_path}' 폴더 내에 txt 파일이 하나도 없습니다!")
    
    print(f"-> 총 {len(file_list)}개의 원본 txt 파일을 발견했습니다. 병합을 시작합니다.")
    all_sentences = []
    
    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
            # 줄바꿈이나 공백 정제 후 문장 단위 형태로 분할 보관
            sentences = [s.strip() for s in re.split(r'[.\n]', content) if len(s.strip()) > 10]
            all_sentences.extend(sentences)
            
    return all_sentences

def generate_augmented_chunk(base_sentences):
    """원본 문장들을 조합하고 변형하여 KoELECTRA용 실전 피싱 덩어리(Chunk) 생성"""
    # 무작위로 3~5개 문장을 엮어서 하나의 통화 맥락 덩어리를 만듦
    num_sentences = random.randint(3, 5)
    sampled = random.sample(base_sentences, min(num_sentences, len(base_sentences)))
    chunk = " ".join(sampled)
    
    # 1) 도메인 핵심 키워드 무작위 교차 변형
    for b in banks:
        if b in chunk: chunk = chunk.replace(b, random.choice(banks))
    for g in govs:
        if g in chunk: chunk = chunk.replace(g, random.choice(govs))
    for r in roles:
        if r in chunk: chunk = chunk.replace(r, random.choice(roles))
        
    # 2) STT 구어체 노이즈 삽입 (말더듬 효과)
    if random.random() < 0.5:
        noise = random.choice(stt_noises)
        insert_pos = len(chunk) // 2
        chunk = chunk[:insert_pos] + noise + chunk[insert_pos:]
        
    # 3) CS 인삿말 강제 주입 (말투 과적합 원천 차단 트릭!)
    if random.random() < 0.4:
        chunk = random.choice(cs_prefixes) + chunk
        
    # 4) KoELECTRA 토큰 최대 길이에 맞춰 컷팅
    if len(chunk) > MAX_TOKEN_CHARS:
        chunk = chunk[:MAX_TOKEN_CHARS]
        
    return chunk.strip()

# ==========================================
# 🛑 메인 실행 프로세스
# ==========================================
if __name__ == "__main__":
    try:
        # 1. 495개 원본 파일 흡수
        raw_sentences = load_and_merge_txts(TXT_FOLDER_PATH)
        print(f"-> 원본 파일로부터 총 {len(raw_sentences)}개의 기본 대화 파편을 추출했습니다.")
        
        # 2. 목표치(6950개)가 될 때까지 무한 하이브리드 증강 작동
        print(f"-> 목표치 {TARGET_ROW_COUNT}개를 향해 대규모 데이터 증강을 시작합니다...")
        final_phishing_texts = []
        
        # 중복 방지를 위해 셋(Set) 구조 활용하면서 생성
        unique_check = set()
        
        while len(final_phishing_texts) < TARGET_ROW_COUNT:
            new_chunk = generate_augmented_chunk(raw_sentences)
            
            if new_chunk not in unique_check and len(new_chunk) > 30:
                unique_check.add(new_chunk)
                final_phishing_texts.append(new_chunk)
                
            # 무한 루프 방지용 안전장치 (원본 소스가 너무 적을 때를 대비)
            if len(unique_check) > TARGET_ROW_COUNT * 2:
                break

        # 3. CSV 데이터프레임 구축 (Label 1 고정)
        df_augmented = pd.DataFrame({
            'text': final_phishing_texts[:TARGET_ROW_COUNT],
            'label': [1] * TARGET_ROW_COUNT
        })
        
        # 4. 저장
        df_augmented.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        print("\n" + "="*50)
        print(f"대성공! 최종 피싱 데이터셋 생성 완료: {OUTPUT_CSV_PATH}")
        print(f"최종 행 개수: {len(df_augmented)} 행 (Label 1: 피싱 전용)")
        print("="*50)
        
        print("\n[생성된 데이터 미리보기 3개]")
        print(df_augmented['text'].tail(3).values)

    except Exception as e:
        print(f"\n에러 발생: {e}")
        print("TXT_FOLDER_PATH에 495개 파일이 들어있는 폴더 경로가 올바르게 입력되었는지 확인해 주세요!")