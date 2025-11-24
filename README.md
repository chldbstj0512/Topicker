# Topicker  
**LLM-Reasoning Enhanced Hybrid Statistical Topic Modeling**

Topicker는 **통계 기반 토픽 모델링**과 **LLM reasoning**을 결합하여  
노이즈가 많은 실제 텍스트 환경에서도 의미 기반(topic-level semantics)을 포착하는 것을 목표로 합니다.

---

## 🎯 Research Motivation

### 1) **Semantic Topic Discovery**
단순 단어 빈도가 아닌,  
문장 수준에서 의미적 일관성(semantic coherence)을 고려하여  
텍스트의 잠재 토픽(latent topics)을 추출할 수 있어야 함.

### 2) **Real-world Noisy Data Handling**
현실 데이터는 필연적으로 **노이즈를 포함**함:
- 욕설  
- 깨진 텍스트  
- SNS 특유의 구어체  
- 맥락 단절  

* 이러한 데이터는 **학습/테스트에서 배제하면 안 됨**  
하지만 **별도 클래스로 분리**할 수 있어야 함

---

## 2-way Preprocessing
강한 전처리(토큰만 남기거나 문장 삭제)는  
- **의미 기반 토픽 발견**을 저해하고  
- **현실 적용성**을 떨어뜨림  

Topicker는 **원문 문장 정보를 최대한 보존**하면서  
LLM reasoning으로 노이즈를 구분·완화하는 방향으로 구성

---

## 📌 Project Goal (In Progress)
- LLM reasoning을 통한 **context-aware topic induction**
- 통계 모델 기반 **안정적 topic distribution 추정**
- **노이즈 분리 기능**을 포함한 hybrid modeling 구조
- 실제 SNS/커뮤니티 데이터 적용을 위한 **robust pipeline**

> *This project is actively under research.  
> The repository will be updated as the model architecture and experiments evolve.*

---
