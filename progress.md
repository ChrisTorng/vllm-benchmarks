底下用 GitHub Copilot/Agent/GPT-4.1:

1. 請針對 benchmark_serving.py 及其相依程式，為我建立 requirements.txt。
   => 放棄，只需安裝 vllm 套件。
2. 我使用命令列 `python3 benchmarks/benchmark_serving.py --backend vllm --model google/gemma-3-1b-it --endpoint /v1/completions --dataset-name sonnet --dataset-path benchmarks/sonnet.txt --num-prompts 1` 方式呼叫 `benchmarks/benchmark_serving.py`。現在我想要改用一個獨立的 `llm-loadtest.py`，呼叫 `benchmark_serving.py` 裡面的未知 function，可以傳遞以上參數，並反覆以不同的 `num-prompts` 值呼叫，先以 1~3 的值呼叫。由於原本使用命令列呼叫時，會反覆地執行一些初始化動作，我希望新的 py 程式，可以只執行初始化一次，之後就完全執行多次的 `num-prompts` 動作而已，這樣就可以節省很多執行初始化時間。
   => 已完成。建立了 llm-loadtest.py，可使用 `--prompts-list 1,2,3` 參數設定測試值。
3. 將 llm-loadtest.py 移到 vllm 根目錄，修正參考路徑。
   => 已完成。使用 `sys.path.insert(0, benchmarks_dir)` 方式修正路徑。
4. #terminalSelection 請問在不動 benchmarks 資料夾內檔案的前提下，是否能修正引用 py 檔路徑的問題?
   => 已完成。透過增加 benchmarks 目錄到 Python 路徑來解決問題，無需修改 benchmarks 資料夾內容。
5. 請依所有指示更新。
   => 已完成。所有程式碼及註解均使用英文，README.md 也已更新，記載新功能。
6. llm-loadtest.py 請全面檢視，還有很多中文。
7. 請將原始壓測的結果，全部儲存到 `./Results/{model}_{dataset-name}_{prompts-list}_{seq}.raw.txt`。另將所有結果值，以 tab 間隔字元格式，收集到 `./Results/{model}_{dataset-name}_{prompts-list}_{seq}.results.txt`。最後的螢幕顯示結果，請依原始壓測的格式完整呈現。
8. #terminalSelection 請注意 model 名稱會有 / 字元，請規範化為合法檔名字元 (如 _) 後再儲存。
9. 我要完整的原始壓測資料，目前收集的資料不完整，包括 raw/results.txt 都需要更新。
10. 前面的不對，我不要 JSON 格式，我要 raw.txt 與原始壓測的螢幕輸出格式一致， results.txt 收集資料項目也一致，其他多餘的都不要。
11. #terminalSelection 請仔細檢查，benchmark_serving.py 的輸出與 llm-loadtest.py 的輸出不一樣。我要 benchmark_serving.py 一模一樣的輸出，儲存到 raw.txt 中，也要顯示出來。而 results.txt 裡面的欄位值，也要與 benchmark_serving.py 的輸出值項目一致。