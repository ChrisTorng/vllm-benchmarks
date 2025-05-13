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
7. 請將原始壓測的結果，全部儲存到 `./results/{model}_{dataset-name}_{prompts-list}_{seq}.raw.txt`。另將所有結果值，以 tab 間隔字元格式，收集到 `./results/{model}_{dataset-name}_{prompts-list}_{seq}.results.txt`。最後的螢幕顯示結果，請依原始壓測的格式完整呈現。
8. #terminalSelection 請注意 model 名稱會有 / 字元，請規範化為合法檔名字元 (如 _) 後再儲存。
9. 我要完整的原始壓測資料，目前收集的資料不完整，包括 raw/results.txt 都需要更新。
10. 前面的不對，我不要 JSON 格式，我要 raw.txt 與原始壓測的螢幕輸出格式一致， results.txt 收集資料項目也一致，其他多餘的都不要。
11. #terminalSelection 請仔細檢查，benchmark_serving.py 的輸出與 llm-loadtest.py 的輸出不一樣。我要 benchmark_serving.py 一模一樣的輸出，儲存到 raw.txt 中，也要顯示出來。而 results.txt 裡面的欄位值，也要與 benchmark_serving.py 的輸出值項目一致。

12. 請將最後的結果，繪製一張折線圖，橫軸是次數，縱軸左邊是 Request throughput (req/s)/Output token throughput (tok/s) (兩個軸)，右邊是 Mean TTFT (ms)/Mean TPOT (ms) (兩個軸)。
    => 建立新的 py 不符目的，放棄。
13. 請在 llm-loadtest.py 將最後的結果，繪製一張折線圖，橫軸是次數，縱軸左邊是 Request throughput (req/s)/Output token throughput (tok/s) (兩個軸)，右邊是 Mean TTFT (ms)/Mean TPOT (ms) (兩個軸)。圖檔名依先前 txt 前綴，後加 .png。
14. 目前的輸出圖檔檔名最後是 `..png` 請修正。另目前折線圖與指定需求不同，我要橫軸是 Successful requests 次數，縱軸左邊是 Request throughput (req/s)/Output token throughput (tok/s) (兩個軸)，右邊是 Mean TTFT (ms)/Mean TPOT (ms) (兩個軸)。
15. 圖中沒看到 Request throughput (req/s) 值之折線。另 Output token throughput (tok/s) 軸要在左邊。右邊是 Mean TTFT (ms)/Mean TPOT (ms) 兩個軸，因為數值差異太多。還有所有軸均以 0 值為開始。
16. 圖中仍沒看到 Request throughput (req/s) 值之折線，檢查輸出值確定有此值輸出，請再仔細檢查。另輸出路徑改為 `./results/{model}_{dataset-name}/{timestamp}_{prompts-list}.*`。還有輸出圖檔中的標題有模型名稱，但是是由規範化的 (修改 \/ 為 _) 的名稱，請修正為原始模型名稱。
17. #terminalSelection 現在又變成 Output token throughput (tok/s) 值不見了，請再仔細檢查。
18. Ask: 我發現似乎是因為 Request throughput (req/s) 與 Output token throughput (tok/s) 的值同步增長，又列為不同軸，結果所有點都重疊的樣子，因此會誤以為看不到。請為我解釋這個現象是否正常，或者是程式收集數值有誤?
   => 因輸出 sonnet-output-len 預設值為 150，上述兩值為固定 1:150 的關係。
19. 請刪除圖中 Request throughput (req/s) 之軸與值。並把先前偵錯用的多餘輸出或檢查等程式均刪除。

底下使用 Gemini 2.5 Pro:
20. 請在保留目前所有的輸出檔 raw.txt/results.txt/png 之外，再增加輸出到同資料夾下的 all.results.txt/average.results.txt 與 all.png。all.results.txt 會以附加的方式，將原本的 results.txt 內容輸出進去，並額外加入第一欄為 timestamp 值 (實際執行該次測試的 timestamp，而非運用於檔名之多次測試之統一 timestamp)。而 average.results.txt 則為依 Succesful requests 排序，而相同 Succesful requests 的多個 rows 取平均值，並刪除 timestamp 欄位後儲存。all.png 則為依據 average.results.txt 的結果，再次繪製相同 png 的折線圖結果。
21. 執行後沒有看到 all.* 相關檔案。

底下使用 Claude 3.7 Sonnet:
22. 請在保留目前所有的輸出檔 raw.txt/results.txt/png 之外，再增加輸出到同資料夾下的 all.results.txt/average.results.txt 與 all.png。all.results.txt 會以附加的方式，將原本的 results.txt 內容輸出進去，並額外加入第一欄為 timestamp 值 (實際執行該次測試的 timestamp，而非運用於檔名之多次測試之統一 timestamp)。而 average.results.txt 則為由 all.results.txt 內全部資料，依 Succesful requests 排序，而相同 Succesful requests 的多個 rows 取平均值，並刪除 timestamp 欄位後儲存。all.png 則為依據 average.results.txt 的結果，再次繪製相同 png 的折線圖結果。目前有新建的 `plot_llm_loadtest_chart()` 是否有用? 若可用請利用，否之則刪除。
23. #terminalSelection 修正錯誤。
24. 目前最後的結果產出 all.1.results.txt/all.3.average.txt/all.4.png，以上檔名我是手動修改，請幫我確認是否全部程式都有改正。我已經手動加第一欄 Keep 在 all.1.results.txt 中，請修改程式，亦在 all.1.results.txt 所有附加結果中，預設加入第一欄 Keep 值為 1。然後再加一個 all.2.sorted.txt 輸出檔，它會讀 all.1.results.txt 內所有值，於所有 rows 中，將 Keep=1 放前面，隔一空行後放 Keep=0 在後面，這兩組內依 Successful requests 排序，若值一樣就依 timestamp 排序。然後生成 all.3.average.txt 時，僅取得 all.2.sortedtxt 裡的 Keep=1 項目而已，略過 Keep=0 項目。
25. 執行後發現一些問題。目前 all.2.sorted.txt 的筆數較少，經查少了今天的所有資料，是否不小心在 Keep=0 之後的資料都未讀入? 而 all.2.sorted.txt 中 Keep=0 的所有資料，應該與前面 Keep=1 的資料中間隔一個空行。另排序的要求是「這兩組 (Keep=1/0) 內依 Successful requests 排序，若值一樣就依 timestamp 排序」，目前看未滿足 timestamp 排序。再來 all.3.average.txt 中未將 Keep=0 資料均排除，連帶 all.4.png 也包括了 Keep=0 資料。
26. all.2.sorted.txt 中必須保留 Keep 欄位，而且前面有指定 Keep=1 與 Keep=0 資料中間要多空一空行。這樣在計算 all.3.average.txt 時，才不會誤將 Keep=0 欄位也加入。另 all.3.average.txt 計算平均值時，請截到小數兩位。

27. 我要將 llm-loadtest.py 目前的輸入參數均改使用 json 設定檔方式讀取，僅有 prompts-list 參數仍保持在命令列輸入。
28. 我要可以由參數列指定輸出路徑，json 檔內也要增加輸出路徑，json 檔內的預設值為 json 檔所在路徑 "."，因此實際路徑解析由 json 檔所在路徑為基底。

29. 因為我會人工檢視 all.1.results.txt 內容，將有問題的測試項目設定 Keep=0 不採用，故請將 llm-loadtest.py 後段輸出 all.2.sorted.txt/all.3.average.txt/all.4.png 部份拆出來為獨立的 generate-all.py，但輸出 all.1.results.txt 仍保留在原 py 內，這樣就保留可以在修改 Keep 值後重新建立後續三個檔的機制。另請建立 shell 程式，會接連呼叫 llm-loadtest.py 與 generate-all.py，並可傳入任意參數都給 llm-loadtest.py，讓我仍然可以方便執行一個指令做完全部的事。generate-all.py 會接收一個路徑參數，指定 all.1.results.txt 輸入及後續輸出檔的路徑。
30. run-llm-loadtest.sh 只要將所有參數轉傳給 llm-loadtest.py 即可，不要介入讀取或異動。另 output-dir 若有傳入，則用作 generate-all.py 的參數，若沒有 output-dir 但有 config，則以 config 路徑為 generate-all.py 參數；若也沒有，則以目前路徑為 generate-all.py 參數。
31. 目前 llm-loadtest.py 裡會以 {model}_{dataset-name} 為輸出子目錄，現在我要完全以 output_dir/output-dir 參數為準，不要再建立輸出子目錄了。這樣它的輸出路徑才能後續由 generate-all.py 取得。
32. 請將 README.md 全面更新到符合目前狀態。
33. 輸出檔名請勿異動，仍然維持先前的名稱，不要加入 {model}_{dataset-name}。
34. {timestamp}_{prompts-list}.raw.txt/results.txt/png 三個檔改輸出到 details 子資料夾，而 all.1.results.txt 維持不變。