[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/feVJoLVn)
# Deep learning 25\_2 : Assignment Readme

This repository contains my solutions to the assignments of the Deep Learning class offered by Professor Heewon Kim at Soongsil University (2nd semester, 2025).

The class is part of [RealityLab](https://reality.ssu.ac.kr/), which focuses on research in deep learning and related areas.

---

# 📘 프로젝트 개요: Image Captioning with RNNs

본 프로젝트에서는 Microsoft COCO 2014 Dataset을 기반으로 이미지에 대한 자연어 캡션을 생성하는 모델을 구현한다.
주요 목표는 다음과 같다.

1. Vanilla RNN 기반 언어 모델 구현

2. 단일 timestep RNN forward/backward 구현

3. 시퀀스 전체를 처리하는 RNN forward/backward 구현

4. Word Embedding layer 구현

5. Temporal affine layer 및 temporal softmax loss 이해

6. Image feature → Initial hidden state 매핑

7. Test-time caption sampling 구현

8. 작은 데이터셋에 대한 overfitting 실험

모든 핵심 연산은 utils/rnn_layers.py와 utils/classifiers/rnn.py 파일에 _______처럼 들어가있는 빈칸을 채워서 구현합니다.
---

## ⚙️ 실습 환경 설정

1. Conda 가상 환경 생성:

```bash
conda create --name ssu_rnn python=3.10
conda activate ssu_rnn
```

2. 필수 라이브러리 설치:

```bash
pip install numpy==2.2.6
pip install opencv-python==4.12.0.88
pip install Pillow==11.3.0
pip install h5py
pip install future
pip install imageio
```

---

## 📦 데이터 준비
Microsoft COCO 캡셔닝 데이터는 약 1GB이며, 아래 스크립트를 실행하여 자동 다운로드한다.
```bash
chmod +x *.sh
./get_assignment3_data.sh
```
다운로드된 데이터는 다음 요소를 포함한다.

* coco2014_captions.h5

* VGG-16 fc7 feature (train2014_vgg16_fc7.h5, val2014_vgg16_fc7.h5)

* PCA 축소 feature (*_pca.h5)

* 이미지 URL 텍스트 (train2014_urls.txt, val2014_urls.txt)

* Vocabulary 정보(coco2014_vocab.json)

* 원본 이미지는 제공되지 않으며 URL 기반으로 필요할 때 실시간 다운로드한다.

2. **성능 평가**

   * validation set을 이용해 최적 k 선택
   * test set 분류 정확도 측정
   * distance 계산 방식별 속도 및 정확도 비교

---

## 🧩 구현해야 할 주요 기능

1. Vanilla RNN — 단일 timestep 구현:

```bash
파일: utils/rnn_layers.py

* rnn_step_forward
* rnn_step_backward
```

정확한 hidden state 갱신과 gradient 계산이 핵심.

2. Vanilla RNN — 전체 시퀀스 처리:
```bash
파일: utils/rnn_layers.py

* rnn_forward
* rnn_backward
```
타임스텝 간 hidden state propagation 구조를 완성한다.

3. Word Embedding Layer
```bash
* word_embedding_forward
* word_embedding_backward
```
→ 동일 단어에 대한 gradient 누적이 핵심 포인트.

4. Temporal Affine Layer
```bash
이미 제공된 함수:
* temporal_affine_forward
* temporal_affine_backward
```
RNN hidden state → vocabulary score 변환.

5. Temporal Softmax Loss
```bash
* <NULL> 마스크를 고려한 시계열 softmax loss
* temporal_softmax_loss(이미 구현됨)
```

6. CaptioningRNN 모델 조립
```bash
파일: utils/classifiers/rnn.py
* CaptioningRNN.loss()
(forward & backward 구현)

전체 데이터 흐름:

* image feature → initial hidden
* word embedding
* RNN 순방향
* vocabulary 점수 계산
* temporal softmax 손실
```

7. Test-time Sampling
```bash
파일: CaptioningRNN.sample()
* autoregressive sampling
* <START> 시작 → <END> 또는 max length까지 생성
```
학습 데이터에서는 자연스러운 문장 생성이 가능하지만
검증 데이터에서는 부정확한 문장이 생성될 수 있다.

---

## 📊 결과 보고

* 본 reop를 본인 컴퓨터에 git pull하시고 필요한 파일 utils/rnn.py 등등을 완성하시오.
* 그 다음 실습한 utils폴더와 실행 로그가 담겨있는 RNN_Captioning.ipynb을 제출하시요.
* git push를 하면 자동으로 과제가 제출됩니다.
**class room 제출 방법** : [https://github.com/WE-SOPT-29th-Web-Part/notice-by-Euijin-Kim] 참고
---

## ❓ 질문 방법

* 코드 실행 에러나 환경 문제: 조교 메일 문의 ([por1329@naver.com](mailto:por1329@naver.com))
* 구현 아이디어/개념 관련: 강의 자료 및 QnA 활용
* **주의:** 지정된 Conda 환경을 사용하지 않아 발생한 문제는 답변하지 않음

---

## 🚨 주의사항

* 무단 코드 복사/붙여넣기 적발 시 0점 처리
