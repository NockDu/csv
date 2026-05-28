call: 보이스피싱 탐지 모델
sms: 스미싱 탐지 모델
KE: KoELECTRA
RF: Random Forest

/code
    /call
        augment_phishing_data.py: 피싱 데이터 증강 코드
        make_call_model(KE).py: KoELECTRA 보이스피싱 탐지 모델 학습 코드
        make_call_model(RF).py: Random Forest 보이스피싱 탐지 모델 학습 코드
        merge_data_csv_maker.py: 정상/피싱 데이터 통합 코드

    /sms
        make_sms_model(KE).py: KoELECTRA 스미싱 탐지 모델 학습 코드
        make_sms_model(RF).py: Random Forest 스미싱 탐지 모델 학습 코드
        merge_sms.py: 정상/피싱 데이터 통합 코드

    test.py: KE/RF 테스트 코드

/data
    /final
        500_가중치.csv: RF 학습용 단어장
        test_phone.csv: 보이스피싱 테스트 데이터셋
        test_sms.csv: 스미싱 테스트 데이터셋
        train_phone.csv: 보이스피싱 학습 데이터셋
        train_sms.csv: 스미싱 학습 데이터셋

    /original
        /call
            final_balanced_normal_data.csv: 정상 데이터
            phone_augmented_6950.csv: 증강 피싱 데이터
            processed_phishing_data.csv: 피싱 데이터

        /sms
            alert.csv: 피싱 데이터
            alert_augmented_2680.csv: 증강 피싱 데이터
            chat.csv: 정상 데이터

/model
    /call(KE)
        /call_koelectra_model: KoELECTRA 보이스피싱 탐지 모델
            config.json
            model.safetensors
            tokenizer.json
            tokenizer_config.json
            training_args.bin
        call_koelectra_detaile.PNG: Accuracy 보고서
        call_koelectra_performance.png: Confusion Matrix & ROC Curve
        call_koelectra_training.PNG: 학습 경과

    /call(RF)
        /call_randomforest_model: Random Forest 보이스피싱 탐지 모델
            call_phishing_model.pkl
            call_tfidf_vectorizer.pkl
        call_randomforest_detaile.PNG: Accuracy 보고서
        call_randomforest_performance.png: Confusion Matrix & ROC Curve

    /sms(KE)
        /sms_koelectra_model: KoELECTRA 스미싱 탐지 모델
            config.json
            model.safetensors
            tokenizer.json
            tokenizer_config.json
            training_args.bin
        sms_koelectra_detaile.PNG: Accuracy 보고서
        sms_koelectra_performance.png: Confusion Matrix & ROC Curve
        sms_koelectra_training.PNG: 학습 경과

    /sms(RF)
        /sms_randomforest_model: Random Forest 스미싱 탐지 모델
            sms_phishing_model.pkl
            sms_tfidf_vectorizer.pkl
        sms_randomforest_detaile.PNG: Accuracy 보고서
        sms_randomforest_performance.png: Confusion Matrix & ROC Curve

    /test_results(KEPR): 26.05.28 보고용 KE 분석
        call_detaile.png
        call_performance.png
        call_training.png
        sms_detaile.png
        sms_performance.png
        sms_training.png

    test_results_text.txt: test.py 결과 모음