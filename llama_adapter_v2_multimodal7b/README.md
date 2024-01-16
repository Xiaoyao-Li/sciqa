# Finetuning LLaMA-Adapter-V2 Multi-modal 7B for 2023-2024 Machine Learning Project

## Setup

* setup up a new conda env and install necessary packages.
  ```bash
  conda create -n sciqa_finetuning python=3.8 -y
  conda activate sciqa_finetuning
  pip install -r requirements_own.txt
  ```

* Obtain the related checkpoints of our finetuning by following [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/1c040bb2c7b3464a9266/) and [Baidu Cloud](https://pan.baidu.com/s/1oR2bo8UxFYu51xOAQ3yeQA?pwd=4zyk). Organize the downloaded file in the following structure
  ```
  ./output_finetune
  ├── event
  │   └── events.out.tfevents.1704467391.featurize.58727.0
  ├── output.log
  ├── log.txt  
  └── checkpoint-3.pth
  ./llama_checkpoints
  ├── 7B
  │   ├── checklist.chk
  │   ├── consolidated.00.pth
  │   └── params.json
  ├── llama.sh
  ├── tokenizer.model 
  └── tokenizer_checklist.chk
  ./ckpts
  ├── 1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth
  └── 7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth
  ```

## Usage (To Reproduce Our Result)

### Finetuning LLaMA-Adapter-V2-Multimodal7B on ScienceQA TrainingSet
  ```
  . exps/finetune.sh ./llama_checkpoints/ ./ckpts/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth ./sqa_finetune.yaml ./output
  ```

### Evaluating LLaMA-Adapter-V2-Multimodal7B on ScienceQA TestSet before finetuning 
  ```
  python util/evaluate_sqa.py --pretrained_path ./ckpts/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth --llama_path ./llama_checkpoints/ --output_path ./output_evaluation --data_config ./sqa_test.yaml
  ```

### Evaluating Model on ScienceQA TestSet after finetuning 
  ```
  python ./evaluate_sqa.py --pretrained_path ./output/checkpoint-3.pth --llama_path ./llama_checkpoints/ --output_path ./output_evaluation --data_config ./sqa_test.yaml
  ```

### Calculating Accuracy of Testing OR Converting the .txt Result File to .json Result File
  ```
  python ./calculateAcc.py
  ```