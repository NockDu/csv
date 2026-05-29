call: 보이스피싱 탐지 모델  
sms: 스미싱 탐지 모델  
KE: KoELECTRA  
KB: KoBERT  
RF: Random Forest  
SVM: Support Vector Machine  
NB: Naive Bayes  
LR: Logistic Regression

```
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
    /call
        /KoBERT(call)
            /kobert_call_model: KoBERT 보이스피싱 탐지 모델
                config.json
                model.safetensors
                tokenizer.json
                tokenizer_config.json
                training_args.bin
            kobert_call_detaile.PNG: Accuracy 보고서
            kobert_call_performance.png: Confusion Matrix & ROC Curve
            kobert_call_training.PNG: 학습 경과

        /KoELECTRA(call)
            /koelectra_call_model: KoELECTRA 보이스피싱 탐지 모델
                config.json
                model.safetensors
                tokenizer.json
                tokenizer_config.json
                training_args.bin
            koelectra_call_detaile.PNG: Accuracy 보고서
            koelectra_call_performance.png: Confusion Matrix & ROC Curve
            koelectra_call_training.PNG: 학습 경과

        /Logistic Regression(call)
            /logisticregression_call_model: Logistic Regression 보이스피싱 탐지 모델
                logisticregression_call_phishing_model.pkl
                logisticregression_call_tfidf_vectorizer.pkl
            logisticregression_call_detaile.PNG: Accuracy 보고서
            logisticregression_call_performance.png: Confusion Matrix & ROC Curve

        /Naive Bayes(call)
            /naivebayes_call_model: Naive Bayes 보이스피싱 탐지 모델
                naivebayes_call_phishing_model.pkl
                naivebayes_call_tfidf_vectorizer.pkl
            naivebayes_call_detaile.PNG: Accuracy 보고서
            naivebayes_call_performance.png: Confusion Matrix & ROC Curve

        /Random Forest(call)
            /randomforest_call_model: Random Forest 보이스피싱 탐지 모델
                randomforest_call_phishing_model.pkl
                randomforest_call_tfidf_vectorizer.pkl
            randomforest_call_detaile.PNG: Accuracy 보고서
            randomforest_call_performance.png: Confusion Matrix & ROC Curve

        /SVM(call)
            /svm_call_model: Random Forest 보이스피싱 탐지 모델
                svm_call_phishing_model.pkl
                svm_call_tfidf_vectorizer.pkl
            svm_call_detaile.PNG: Accuracy 보고서
            svm_call_performance.png: Confusion Matrix & ROC Curve

    /sms
        /KoBERT(sms)
            /kobert_sms_model: KoBERT 스미싱 탐지 모델
                config.json
                model.safetensors
                tokenizer.json
                tokenizer_config.json
                training_args.bin
            kobert_sms_detaile.PNG: Accuracy 보고서
            kobert_sms_performance.png: Confusion Matrix & ROC Curve
            kobert_sms_training.PNG: 학습 경과

        /KoELECTRA(sms)
            /koelectra_sms_model: KoELECTRA 스미싱 탐지 모델
                config.json
                model.safetensors
                tokenizer.json
                tokenizer_config.json
                training_args.bin
            koelectra_sms_detaile.PNG: Accuracy 보고서
            koelectra_sms_performance.png: Confusion Matrix & ROC Curve
            koelectra_sms_training.PNG: 학습 경과

        /Logistic Regression(sms)
            /logisticregression_sms_model: Logistic Regression 스미싱 탐지 모델
                logisticregression_sms_phishing_model.pkl
                logisticregression_sms_tfidf_vectorizer.pkl
            logisticregression_sms_detaile.PNG: Accuracy 보고서
            logisticregression_sms_performance.png: Confusion Matrix & ROC Curve

        /Naive Bayes(sms)
            /naivebayes_sms_model: Naive Bayes 스미싱 탐지 모델
                naivebayes_sms_phishing_model.pkl
                naivebayes_sms_tfidf_vectorizer.pkl
            naivebayes_sms_detaile.PNG: Accuracy 보고서
            naivebayes_sms_performance.png: Confusion Matrix & ROC Curve

        /Random Forest(sms)
            /randomforest_sms_model: Random Forest 스미싱 탐지 모델
                randomforest_sms_phishing_model.pkl
                randomforest_sms_tfidf_vectorizer.pkl
            randomforest_sms_detaile.PNG: Accuracy 보고서
            randomforest_sms_performance.png: Confusion Matrix & ROC Curve

        /SVM(sms)
            /svm_sms_model: SVM 스미싱 탐지 모델
                svm_sms_phishing_model.pkl
                svm_sms_tfidf_vectorizer.pkl
            svm_sms_detaile.PNG: Accuracy 보고서
            svm_sms_performance.png: Confusion Matrix & ROC Curve

    /test_results(KEPR): 26.05.28 보고용 KE 분석
        call_detaile.png
        call_performance.png
        call_training.png
        sms_detaile.png
        sms_performance.png
        sms_training.png

    test_results_text.txt: test.py 결과 모음
```
