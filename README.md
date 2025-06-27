# 📚 小模型，大貢獻：準確且高效的中文新聞摘要模型訓練研究
**Small Model, Big Impact: A Study on Training for Accurate and Efficient Chinese News Summarization Models**
- **成功大學 115 級專題(展)**
- **專題組員**：王郁豪
- **指導教授**：陳響亮
- **組別**：C-22

---

## 📄 專題簡介、報告與海報

- [📘 專題簡介](docs/專題簡介.pdf)
- [🖼️ 專題海報](docs/專題海報.pdf)
- [📝 專題報告](docs/專題報告.pdf)

---

## 🔗 完整模型與手機應用

- 🤖 **完整模型（Hugging Face）**：
  [Qwen2.5-0.5B-Instruct-Curriculum-5stage-v4-lr_adj](https://huggingface.co/BennyWang/Qwen2.5-0.5B-Instruct-Curriculum-5stage-v4-lr_adj)

- 📱 **手機部署應用**：
  [📂 程式碼 GitHub Repo](https://github.com/BennyWang1007/Individual-studies-app/)
  [⬇️ APK 下載 v1.1.0](https://github.com/BennyWang1007/Individual-studies-app/releases/tag/v1.1.0)

---

## 📑 摘要 Abstract

在資訊爆炸的現代社會中，讀者常因時間有限與聳動標題影響，難以正確理解事件核心。本研究旨在尋找更佳的訓練策略，並開發能在手機等資源受限設備上運行的中文新聞摘要模型，協助用戶快速獲取重點資訊。

最終成果顯示，學生模型不僅能有效生成內容正確、語句流暢的繁體中文摘要，亦能修正教師模型在繁體中文用語上的部分錯誤，展現良好的應用潛力，也為中文新聞摘要領域的輕量化模型應用提供了可行且具潛力的實作框架。

---

## 🔬 實驗方法 Method

### 資料蒐集：
- 爬取聯合新聞網新聞作為訓練資料。
- 整理 YeungNLP/firefly-pretrain-dataset 作為預訓練資料。

### 模型選擇：
- **學生模型**：Qwen2.5-0.5B-Instruct（簡稱 0.5B 模型）
- **教師模型**：Qwen2.5-32B-Instruct（簡稱 32B 模型）

### 課程式訓練流程（S1-S5）：
1. **S1**：使用 OpenCC 進行繁體轉換
2. **S2**：提取關鍵要素（essential aspect）
3. **S3**：構建關聯三元組（triple）
4. **S4**：根據前述資訊生成新聞摘要
5. **S5**：直接從新聞內容生成摘要

### 模型評估：
- 使用 **ROUGE-1/2/L** 與 **BERTScore (F1)** 進行自動化評分
- 使用教師模型針對摘要進行自然性與資訊涵蓋度評分（Judge）

---

## 🧪 實驗設計 Experimental Design

### 資料生成方式比較：
- **V1**：一次性生成所有內容
- **V2**：分步生成（要素 → 三元組 → 摘要）
- **V3**：先摘要，再推導要素與三元組
- **V4**：V3 + 人工修正

### 訓練策略比較：
- **lr_adj**：不使用學習率遞減
- **only_attn**：僅訓練 attention head
- **only_mlp**：僅訓練 decoder 的 MLP 層
- **lora**：使用 LoRA 低秩微調

### 課程訓練流程比較：
- **1-stage**：僅用 S5
- **4-stage**：S2~S5
- **5-stage**：S1~S5（完整課程）

---

## 📊 實驗結果 Results

### 1. 資料生成方式：
- V3 > V4 > V1 > V2（教師模型不擅長推理型摘要）

### 2. 訓練策略：
- 不使用遞減學習率（lr_adj）效果最佳
- 微調全部參數 > LoRA > 凍結部分參數

### 3. 不同訓練階段：
- **5-stage** 最穩定且繁體比例最高，即便 ROUGE-1 稍低於 1-stage

### 4. 與現有模型比較：

| MODEL                    | R-1  | B-F1 | Judge | R-2  | R-L  |
|-------------------------|------|------|--------|------|------|
| Qw2.5-0.5B_4stg_v3       | 45.5 | 77.9 | 70.3   | 24.3 | 37.6 |
| Qw2.5-0.5B_4stg_v1       | 43.8 | 76.8 | 64.0   | 22.1 | 35.5 |
| Qw2.5-0.5B_4stg_v2       | 37.6 | 69.4 | 65.1   | 17.5 | 23.4 |

| MODEL                                | R-1  | B-F1 | Judge | R-2  | R-L  |
|--------------------------------------|------|------|--------|------|------|
| 4stg_v3-lr_adj                       | 48.4 | 79.3 | 72.8   | 25.7 | 40.1 |
| 4stg_v3-lr_adj-only_mlp             | 46.6 | 78.6 | 71.5   | 24.2 | 38.4 |
| 4stg_v3-lr_adj_lora                 | 45.6 | 78.0 | 73.6   | 23.3 | 37.4 |
| 4stg_v3                             | 45.5 | 77.9 | 70.3   | 24.3 | 37.6 |
| 4stg_v3-lr_adj-only_attn           | 45.2 | 77.8 | 69.1   | 23.0 | 37.1 |

---

## 結論 Conclusion

本研究證實，結合模型蒸餾與課程式訓練策略，可有效訓練出高效能、低資源的小模型摘要系統。即便使用僅 0.5B 的模型，輸出品質已接近大型模型，優於同級 Gemma 等模型，展現了實用潛力與技術創新。

### 未來方向
- 探索 Soft Label Distillation，進一步吸收教師模型的知識
- 解決簡繁中文在 tokenizer 層的編碼差異問題

---

## 📱 手機程式體驗 Try It on Mobile

模型已成功部署於手機裝置，具備即時摘要功能：

- 自動爬取新聞，或搜尋關鍵字（如「英偉達」）
- 點選新聞後，自動產生摘要（可能需稍作等待）
- 歷史摘要頁面支援點擊展開完整摘要與原文

> 📲 立即體驗：[下載 APK](https://github.com/BennyWang1007/Individual-studies-app/releases/tag/v1.1.0)

---

