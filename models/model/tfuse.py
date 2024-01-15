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

from models.base import MODEL

@MODEL.register()
class TFUSE(nn.Module):
    def __init__(self, cfg: DictConfig, slurm: bool, charlie: bool):
        super(TFUSE, self).__init__()
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
        self.disable_pretrain_image = cfg.disable_pretrain_image
        self.disable_pretrain_text = cfg.disable_pretrain_text
        self.freeze_fasterRCNN = cfg.freeze_fasterRCNN
        self.vocab = json.load(open(cfg.vocab_path, 'r'))
        
        self.choice_to_index = self.vocab['choice']

        if self.disable_pretrain_image:
            self.fasterRCNN = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
            assert cfg.freeze_fasterRCNN == False
        else:
            self.fasterRCNN = fasterrcnn_resnet50_fpn(pretrained=True, weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

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
            disable_pretrain_text=self.disable_pretrain_text,
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

        ## mokey patching for fasterRCNN
        def fasterRCNN_tfuse_forward(self, images, targets=None):
            if self.training:
                if targets is None:
                    self.training = False
                    self.rpn.training = False
                    self.roi_heads.training = False
                else:
                    for target in targets:
                        boxes = target["boxes"]
                        if isinstance(boxes, torch.Tensor):
                            torch._assert(
                                len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                                f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                            )
                        else:
                            torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

            original_image_sizes: List[Tuple[int, int]] = []
            for img in images:
                val = img.shape[-2:]
                torch._assert(
                    len(val) == 2,
                    f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
                )
                original_image_sizes.append((val[0], val[1]))

            images, targets = self.transform(images, targets)

            # Check for degenerate boxes
            # TODO: Move this to a function
            if targets is not None:
                for target_idx, target in enumerate(targets):
                    boxes = target["boxes"]
                    degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                    if degenerate_boxes.any():
                        # print the first degenerate box
                        bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                        degen_bb: List[float] = boxes[bb_idx].tolist()
                        torch._assert(
                            False,
                            "All bounding boxes should have positive height and width."
                            f" Found invalid box {degen_bb} for target at index {target_idx}.",
                        )

            features = self.backbone(images.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])
            proposals, proposal_losses = self.rpn(images, features, targets)
            for idx in range(len(proposals)):
                p = proposals[idx]
                if p.shape[0] < 1000:
                    p = torch.cat([p, torch.zeros([1000 - p.shape[0], 4]).to(p)], dim=0)
                elif p.shape[0] > 1000:
                    p = p[:1000]
                proposals[idx] = p
            detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)

            if torch.jit.is_scripting():
                if not self._has_warned:
                    warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                    self._has_warned = True
                return losses, detections
            else:
                return self.eager_outputs(losses, detections)
        self.fasterRCNN.forward = fasterRCNN_tfuse_forward.__get__(self.fasterRCNN)

        if cfg.freeze_fasterRCNN:
            for param in self.fasterRCNN.parameters():
                param.requires_grad_(False)

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

        if self.loss_type == 'TEST':
            answer = self.criterion(answer, data['choices'], data['answer'])
            return {'pred_answer': answer}
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
            
            return loss, accuracy
        elif self.loss_type == 'TEST':
            choices_indices = [[self.choice_to_index[c] for c in cs] for cs in choices]
            pred_answer_indices = []
            pred_answer = pred_answer.detach().cpu().numpy()
            for i, cs in enumerate(choices_indices):
                cs = [c if c < self.max_choices else -1 for c in cs]
                ans = pred_answer[i][cs]
                pred_answer_indices.append(ans.argmax().item())

            return pred_answer_indices
        else:
            raise NotImplementedError

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

    # @torch.no_grad()
    def _extract_visual_feature(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [batch, 3, H, W]
        Returns:
            visual_feature: [batch, 32, 1024]
        """
        if self.freeze_fasterRCNN:
            self.fasterRCNN.eval()
        visual_features = []
        feature_scores = []
        hook_handles = []
        def save_features(mod, inp, outp):
            visual_features.append(outp)
        def save_scores(mod, inp, outp):
            feature_scores.append(outp)
        for name, layer in self.fasterRCNN.named_modules():
            if name == 'roi_heads.box_head.fc6':
                hook_handles.append(layer.register_forward_hook(save_features))
            elif name == 'roi_heads.box_predictor.cls_score':
                hook_handles.append(layer.register_forward_hook(save_scores))
        with torch.no_grad():
            self.fasterRCNN(image)
        visual_feature = visual_features[0].reshape(image.shape[0], 1000, 1024)
        feature_score = feature_scores[0].reshape(image.shape[0], 1000, 91)

        # find top-32 object features
        feature_score = torch.max(feature_score, dim=2)[0]
        _, topk_obj_idx = torch.topk(feature_score, self.num_vision_tokens, dim=1)
        visual_feature = torch.gather(visual_feature, 1, topk_obj_idx.unsqueeze(2).expand([-1, -1, visual_feature.shape[2]]))
        
        # remove the hook to avoid gpu memory leaking
        for h in hook_handles:
            h.remove()

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
