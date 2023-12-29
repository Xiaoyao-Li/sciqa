import torch
import torch.nn as nn
from torch.nn import Transformer
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from omegaconf import DictConfig
from collections import OrderedDict
from typing import Dict, Tuple, List
import json
import warnings

import models.modules.word_embedding as word_embedding
from models.modules.utils import Fusion, FCNet
from models.modules.visual_encoder import CustomResNet50

from models.base import MODEL

@MODEL.register()
class TFUSE_SCRATCH(nn.Module):
    def __init__(self, cfg: DictConfig, slurm: bool, charlie: bool):
        super(TFUSE_SCRATCH, self).__init__()
        self.slurm = slurm
        self.charlie = charlie
        self.token_features_dim = cfg.token_features_dim
        self.transformer_nheads = cfg.transformer_nheads
        self.num_encoder_layers = cfg.num_encoder_layers
        self.num_decoder_layers = cfg.num_decoder_layers
        self.global_pool = cfg.global_pool
        self.classifier_mid_features = cfg.classifier_mid_features
        self.classifier_dropout = cfg.classifier_dropout
        self.max_choices = cfg.max_choices
        self.max_question_words = cfg.max_question_words
        self.max_hint_words = cfg.max_hint_words
        self.num_vision_tokens = cfg.num_vision_tokens
        self.apply_mask = cfg.apply_mask
        self.loss_type = cfg.loss_type
        self.enable_image = cfg.enable_image
        self.enable_hint = cfg.enable_hint
        self.vocab = json.load(open(cfg.vocab_path, 'r'))
        
        self.choice_to_index = self.vocab['choice']

        self.visual_encoder = CustomResNet50(self.token_features_dim)

        if not self.enable_hint:
            self.words_list = self.vocab['question'].keys()
            self.text_dict = self.vocab['question']
        else:
            self.words_list = self.vocab['question_and_hint'].keys()
            self.text_dict = self.vocab['question_and_hint']
        self.text = word_embedding.TextProcessor(
            classes=self.words_list,
            embedding_features=300,
            lstm_features=self.token_features_dim,
            use_hidden=False, # use whole output, not just final hidden
            drop=0.0,
            enable_hint=self.enable_hint,
            disable_pretrain_text=True,
        )
        self.transformerBlock = nn.Transformer(
            d_model=self.token_features_dim,
            nhead=self.transformer_nheads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
        )
        self.classifier = Classifier(
            in_features=self.token_features_dim,
            mid_features=self.classifier_mid_features,
            out_features=self.max_choices,
            drop=self.classifier_dropout,
            apply_mask=self.apply_mask,
        )

        # if not self.enable_image:
            # # create a learnable features for visual features
        self.dummy_visual_feature = nn.Parameter(torch.randn(1, self.num_vision_tokens, self.token_features_dim))
        if self.global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.token_features_dim))

    def forward(self, data: Dict) -> torch.Tensor:
        """
            data: {
            'image': visual_image,
            'hint': hint_context, 
            'question': question,
            'choices': choices,
            }
            return: 
            {
                'loss': loss,
                'accuracy': accuracy,
            }
        """
        B = len(data['question'])
        DEVICE = data['incontext_image'].device
        incontext_image = data['incontext_image']
        incontext_image_mask = data['mask_image'].to(DEVICE)
        incontext_hint = data['incontext_hint']
        # incontext_hint_mask = torch.ones(B, self.max_hint_words).to(DEVICE)
        question = data['question']
        # question_mask = torch.ones(B, self.max_question_words).to(DEVICE)
        
        question, question_length = self._text_to_index(question, is_hint=False)
        question = question.to(DEVICE)
        question_feature = self.text(question, question_length)
        
        if self.enable_image:
            context_feature = self._extract_visual_feature(incontext_image)
            dvf = self.dummy_visual_feature.expand([B, -1, -1]).to(DEVICE)
            imask = incontext_image_mask.view(-1, 1, 1)
            imask_inv = ~imask
            context_feature = context_feature * imask.float()
            context_feature = context_feature + dvf * imask_inv.float()
        else:
            context_feature = self.dummy_visual_feature.expand([B, -1, -1]).to(DEVICE)
        
        if self.enable_hint:
            incontexthint, incontexthint_length = self._text_to_index(incontext_hint, is_hint=True)
            incontexthint = incontexthint.to(DEVICE)
            incontexthint_feature = self.text(incontexthint, incontexthint_length)
            # concate incontexthint_feature to context_feature
            context_feature = torch.cat((context_feature, incontexthint_feature), dim=1)
        else:
            pass

        if self.global_pool == 'cls':
            cls_token = self.cls_token.expand([B, -1, -1]).to(DEVICE)
            question_feature = torch.cat((cls_token, question_feature), dim=1)
        # permute to [seq_len, batch, feat_size]
        context_feature = context_feature.permute(1, 0, 2)
        question_feature = question_feature.permute(1, 0, 2)
        question_feature = self.transformerBlock(src=context_feature, tgt=question_feature)
        # permute back to [batch, seq_len, feat_size]
        question_feature = question_feature.permute(1, 0, 2)
        if self.global_pool == 'mean':
            question_feature = question_feature.mean(dim=1)
        elif self.global_pool == 'max':
            question_feature = question_feature.max(dim=1)[0]
        elif self.global_pool == 'cls':
            question_feature = question_feature[:, 0]
        else:
            raise NotImplementedError
        
        answer = self.classifier(question_feature)
        loss, accuracy = self.criterion(answer, data['choices'], data['answer'])

        return {'loss': loss, 'accuracy': accuracy}
    
    def criterion(self, pred_answer, choices, answer):
        if self.loss_type == 'BCE':
            answer_indices = torch.tensor([self.choice_to_index[c[a]] for c, a in zip(choices, answer)], dtype=torch.long).to(pred_answer.device)
            answer_onehot = torch.zeros(pred_answer.shape).to(pred_answer.device)
            answer_onehot.scatter_(1, answer_indices.unsqueeze(1), 1)
            loss = F.binary_cross_entropy_with_logits(pred_answer, answer_onehot)

            pred_answer_indices = pred_answer.argmax(dim=1)
            accuracy = (pred_answer_indices == answer_indices).sum().float() / len(answer_indices)
        else:
            raise NotImplementedError
        
        return loss, accuracy

    @torch.no_grad()
    def _text_to_index(self, text: list, is_hint: bool) -> torch.Tensor:
        if is_hint:
            max_words = self.max_hint_words
        else:
            max_words = self.max_question_words
        t_indices = []
        t_length = []
        for t in text:
            t = t.split()
            t = [w.replace(',', '').replace('.', '').replace('?', '').lower() for w in t]
            t = [self.text_dict[w] if w in self.text_dict else -1 for w in t][:max_words]
            t_length.append(len(t))
            t = t + [-1] * (max_words - len(t))
            t_indices.append(t)
        t_indices = torch.tensor(t_indices, dtype=torch.long)
        return t_indices, t_length

    def _extract_visual_feature(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [batch, 3, H, W]
        Returns:
            visual_feature: [batch, 32, 1024]
        """
        visual_feature = self.visual_encoder(image)
        visual_feature = visual_feature.flatten(2).permute(0, 2, 1)
        return visual_feature

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0, apply_mask=True):
        super(Classifier, self).__init__()
        self.lin1 = FCNet(in_features, mid_features, activate='relu', drop=drop/2.5)
        self.lin2 = FCNet(mid_features, out_features, drop=drop)
        self.apply_mask = apply_mask

    def forward(self, q, q_mask=None):
        """
        v: visual feature      [batch, num_obj, token_features_dim]
        q: question            [batch, max_len, token_features_dim]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        if self.apply_mask:
            raise NotImplementedError
        out = self.lin1(q)
        out = self.lin2(out)
        return out
