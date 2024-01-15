import cv2
import llama
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_dir = "./llama_checkpoints/"

# choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
model, preprocess = llama.load("BIAS-7B", llama_dir, llama_type='7B', device=device)
model.eval()

# prompt = llama.format_prompt('Please introduce this painting.')
# prompt = llama.format_prompt("Which of these states is farthest north?", {"choices":{0: "West Virginia", 1: "Louisiana", 2: "Arizona", 3: "Oklahoma"}})
prompt = llama.format_prompt("Identify the question that Kathleen and Bryant's experiment can best answer, your answer should be index of the choices.", {"choices":{0: "Does Kathleen's snowboard slide down a hill in less time when it has a layer of wax or when it does not have a layer of wax?", 1: "Does Kathleen's snowboard slide down a hill in less time when it has a thin layer of wax or a thick layer of wax?"}})

# img = Image.fromarray(cv2.imread("../docs/logo_v1.png"))
img = Image.fromarray(cv2.imread("./ScienceQA_Data/train/3/image.png"))
img = preprocess(img).unsqueeze(0).to(device)

result = model.generate(img, [prompt])[0]

# def _text_to_index(text: list, is_hint: bool) -> torch.Tensor:
#     if is_hint:
#         max_words = self.max_hint_words
#     else:
#         max_words = self.max_question_words
#     t_indices = []
#     t_length = []
#     for t in text:
#         t = t.split()
#         t = [w.replace(',', '').replace('.', '').replace('?', '').lower() for w in t]
#         t = [self.text_dict[w] if w in self.text_dict else -1 for w in t][:max_words]
#         t_length.append(len(t))
#         t = t + [-1] * (max_words - len(t))
#         t_indices.append(t)
#     t_indices = torch.tensor(t_indices, dtype=torch.long)
#     return t_indices, t_length

print(result)
