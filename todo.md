# RLMOE_kinase

### TODO 
02/10 : 
修改BPE code
1. 現在code 會把selfies 裡面的[] 中的字拆分開來，但應該要先將[] 整個視為一個獨立的token 在進行BPE
2. 完成Bert 的code 並運行測試

2/14 : 
1. 訓練bert embedding模型中 
2. 進入下一步 pretrain 一個protein compound GPT 的模型 完成建立dataset 以及開始規劃GPT架構， 
訓練好的GPT 將會分成8分進入MOE
