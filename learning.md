各位可能想說我預計一點到六點，有關壓測真的有那麼多可以講解與測試的嗎?
在上次課程前，我已經花了應該超過十小時在做保哥的課前作業，直到課程結束後才全部完成。
而上次課程中我是未完成的學員，沒想到還有機會再來第二次，而且給的時間還蠻多的。
為了充份運用寶貴的租機器時間，我又預先花了應該超過二十小時，來做壓測程式。
其中也查閱了 LLM 一些基礎資料，還有學習 Linux 上的一些技能。
我並不是比較厲害，只是愛分享。想說我投入了這麼多時間，學到的經驗很想跟大家分享。

## 學習重點
- 藉由壓測學習 LLM 內部架構，實際運作機制
- Linux 環境指令/效能監控
- WSL/Ubuntu/Podman/Proxy 闖關保哥課前作業
- 基礎 LLM/Transformer/Attention 架構機制
- VS Code/GitHub Copilot 撰寫程式，WSL/SSH 連入開發

## 預計過程
### 13:00-14:30 實機
- 執行 Deepseek R1 671B
- 說明 vLLM 官方壓測 [Benchmarking vLLM](https://github.com/vllm-project/vllm/tree/main/benchmarks)
- 說明全自製壓測 [llm-load-test](https://github.com/ChrisTorng/llm-load-test)
- 說明與官方相容壓測 [vllm-benchmarks](https://github.com/ChrisTorng/vllm-benchmarks)
- 待 R1 順利執行後，開始壓測

### 14:30-16:40 說明討論
- 先前壓測結果討論，下階段壓測預備
- 效能監控工具 Netdata
- VS Code SSH 連入遠端執行壓測
- WSL/Ubuntu/Podman/Proxy 保哥課前作業闖關/Dockerfile 內容
- LLM/Transformer/Attention/MoE 架構機制

### 16:40-18:10 繼續實機測試
- 跨機壓測
- Netdata 效能監控
- vllm/SGLang/Ollama 效能比較

## 問題
- vllm/SGLang/Ollama 效能比較
- streaming/Batching 效能比較
- Dockerfile
- 連續問同一個問題/連續問相同領域 (同一專家)/全領域問題/簡單免推理問題/複雜推理問題/單一問題裡面包括多領域知識/簡單的長短答案 (比如要求由 1 數到 n)/複雜的長短答案 (比如複雜問題但僅允許答 是/否 或要求詳盡解釋) 以及以上的任意混合等等
- 要求數量由少至多，效能折線圖的變化

## 壓測程式待改善
- 保證回答一定長度以上
- 限制回答長度
- 連續壓測
- 執行進度百分比
- 記錄 GPU 使用率
- 紀錄所有原始答案輸出

不知各位這次擁有三小時的時間，打算怎麼運用呢? 或者前來聽看看我講些什麼，也可以同步自己練習。

我想即使到結束，我提出的問題還是遠比測試完後能解答的問題還多。如果這裡有以上各領域的專家願意賞光參與，還請前來不吝指教。即使沒有，我也會盡力把自己過去這段時間所學的，完整呈現給大家。