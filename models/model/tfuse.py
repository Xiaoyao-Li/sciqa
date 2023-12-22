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
        self.vocab = json.load(open(cfg.vocab_path, 'r'))
        
        self.choice_to_index = self.vocab['choice']

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

        if not self.enable_image:
            # create a learnable features for visual features
            self.dummy_visual_feature = nn.Parameter(torch.randn(1, self.num_vision_tokens, self.token_features_dim))
        if self.global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.token_features_dim))

        ## mokey patching for fasterRCNN
        def fasterRCNN_tfuse_forward(self, images, targets=None):
            if self.training:
                if targets is None:
                    torch._assert(False, "targets should not be none when in training mode")
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
        # incontext_image_mask = torch.ones(B, incontext_image.shape[1]).to(DEVICE)
        incontext_hint = data['incontext_hint']
        # incontext_hint_mask = torch.ones(B, self.max_hint_words).to(DEVICE)
        question = data['question']
        # question_mask = torch.ones(B, self.max_question_words).to(DEVICE)
        
        question, question_length = self._text_to_index(question, is_hint=False)
        question = question.to(DEVICE)
        question_feature = self.text(question, question_length)
        
        if self.enable_image:
            context_feature = self._extract_visual_feature(incontext_image)
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

    @torch.no_grad()
    def _extract_visual_feature(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [batch, 3, H, W]
        Returns:
            visual_feature: [batch, 32, 1024]
        """
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

class SingleBlock(nn.Module):
    """
    Single Block Inter-/Intra-modality stack multiple times
    """
    def __init__(self, num_block, v_size, q_size, output_size, num_inter_head, num_intra_head, drop=0.0, apply_mask=True):
        super(SingleBlock, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_inter_head = num_inter_head
        self.num_intra_head = num_intra_head
        self.num_block = num_block

        self.v_lin = FCNet(v_size, output_size, drop=drop)
        self.q_lin = FCNet(q_size, output_size, drop=drop)

        self.v2q_interBlock = OneSideInterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop)
        self.q2v_interBlock = OneSideInterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop)
        self.intraBlock = DyIntraModalityUpdate(output_size, output_size, output_size, num_intra_head, drop)

        self.apply_mask = apply_mask

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        # transfor features
        v = self.v_lin(v)
        q = self.q_lin(q)
        v_container = [v]
        q_container = [q]
        result_v = [v]
        result_q = [q]
        for i in range(self.num_block):
            q1 = self.v2q_interBlock(v_container[-1], q_container[-1], v_mask, q_mask)
            q_container.append(q1)
            v1 = self.q2v_interBlock(q_container[-1] + q_container[-2], v_container[-1], q_mask, v_mask)
            v_container.append(v1)
            v2, q2 = self.intraBlock(v_container[-1] + v_container[-2], q_container[-1] + q_container[-2], v_mask, q_mask)  ## original `= intraBlock()`
            v_container.append(v2)
            q_container.append(q2)
            result_v.append(v1)
            result_v.append(v2)
            result_q.append(q1)
            result_q.append(q2)
            v_container.append(v_container[-1] + v_container[-2] + v_container[-3])
            q_container.append(q_container[-1] + q_container[-2] + q_container[-3])
        return sum(result_v), sum(result_q)

class MultiBlock(nn.Module):
    """
    Multi Block (different parameters) Inter-/Intra-modality
    """
    def __init__(self, num_block, v_size, q_size, output_size, num_inter_head, num_intra_head, drop=0.0, apply_mask=True):
        super(MultiBlock, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_inter_head = num_inter_head
        self.num_intra_head = num_intra_head
        self.num_block = num_block

        self.v_lin = FCNet(v_size, output_size, drop=drop)
        self.q_lin = FCNet(q_size, output_size, drop=drop)

        blocks = []
        for i in range(num_block):
            blocks.append(OneSideInterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop, apply_mask=apply_mask))
            blocks.append(OneSideInterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop, apply_mask=apply_mask))
            blocks.append(DyIntraModalityUpdate(output_size, output_size, output_size, num_intra_head, drop, apply_mask=apply_mask))
        self.multi_blocks = nn.ModuleList(blocks)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        v = self.v_lin(v)
        q = self.q_lin(q)
        v_container = [v]
        q_container = [q]
        result_v = [v]
        result_q = [q]
        # use dense residual 
        for i in range(self.num_block):
            q1 = self.multi_blocks[i*3+0](v_container[-1], q_container[-1], v_mask, q_mask)
            q_container.append(q1)
            v1 = self.multi_blocks[i*3+1](q_container[-1] + q_container[-2], v_container[-1], q_mask, v_mask)
            v_container.append(v1)
            v2, q2 = self.multi_blocks[i*3+2](v_container[-1] + v_container[-2], q_container[-1] + q_container[-2], v_mask, q_mask)
            v_container.append(v2)
            q_container.append(q2)
            result_v.append(v1)
            result_v.append(v2)
            result_q.append(q1)
            result_q.append(q2)
            v_container.append(v_container[-1] + v_container[-2] + v_container[-3])
            q_container.append(q_container[-1] + q_container[-2] + q_container[-3])
            
        return sum(result_v), sum(result_q)

class InterModalityUpdate(nn.Module):
    """
    Inter-modality Attention Flow
    """
    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0, apply_mask=True):
        super(InterModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head

        self.v_lin = FCNet(v_size, output_size * 3, drop=drop)
        self.q_lin = FCNet(q_size, output_size * 3, drop=drop)

        self.v_output = FCNet(output_size + v_size, output_size, drop=drop)
        self.q_output = FCNet(output_size + q_size, output_size, drop=drop)
        self.apply_mask = apply_mask

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        batch_size, num_obj = v_mask.shape
        _         , max_len = q_mask.shape
        # transfor features
        v_trans = self.v_lin(v)
        q_trans = self.q_lin(q)
        # mask all padding object/word features
        if self.apply_mask:
            v_trans = v_trans * v_mask.unsqueeze(2)
            q_trans = q_trans * q_mask.unsqueeze(2)
        # split for different use of purpose
        v_key, v_qry, v_val = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_key, q_qry, q_val = torch.split(q_trans, q_trans.size(2) // 3, dim=2)
        # apply multi-head
        v_key_set = torch.split(v_key, v_key.size(2) // self.num_head, dim=2)
        v_qry_set = torch.split(v_qry, v_qry.size(2) // self.num_head, dim=2)
        v_val_set = torch.split(v_val, v_val.size(2) // self.num_head, dim=2)
        q_key_set = torch.split(q_key, q_key.size(2) // self.num_head, dim=2)
        q_qry_set = torch.split(q_qry, q_qry.size(2) // self.num_head, dim=2)
        q_val_set = torch.split(q_val, q_val.size(2) // self.num_head, dim=2)
        # multi-head
        for i in range(self.num_head):
            v_key_slice, v_qry_slice, v_val_slice = v_key_set[i], v_qry_set[i], v_val_set[i]  #[batch, num_obj, feat_size]
            q_key_slice, q_qry_slice, q_val_slice = q_key_set[i], q_qry_set[i], q_val_set[i]  #[batch, max_len, feat_size]
            # inner product & set padding object/word attention to negative infinity & normalized by square root of hidden dimension
            q2v = (v_qry_slice @ q_key_slice.transpose(1,2)) / ((self.output_size // self.num_head) ** 0.5)  #[batch, num_obj, max_len]
            v2q = (q_qry_slice @ v_key_slice.transpose(1,2)) / ((self.output_size // self.num_head) ** 0.5)  #[batch, max_len, num_obj]
            if self.apply_mask:
                q2v.masked_fill_(q_mask.unsqueeze(1).expand([batch_size, num_obj, max_len]) == 0, -float('inf')) 
                v2q.masked_fill_(v_mask.unsqueeze(1).expand([batch_size, max_len, num_obj]) == 0, -float('inf')) 
            # softmax attention
            interMAF_q2v = F.softmax(q2v, dim=2).unsqueeze(3) #[batch, num_obj, max_len, 1]
            interMAF_v2q = F.softmax(v2q, dim=2).unsqueeze(3) #[batch, max_len, num_obj, 1]
            # calculate update input (each head of multi-head is calculated independently and concatenate together)
            v_update = (interMAF_q2v * q_val_slice.unsqueeze(1)).sum(2) if (i==0) else torch.cat((v_update, (interMAF_q2v * q_val_slice.unsqueeze(1)).sum(2)), dim=2)
            q_update = (interMAF_v2q * v_val_slice.unsqueeze(1)).sum(2) if (i==0) else torch.cat((q_update, (interMAF_v2q * v_val_slice.unsqueeze(1)).sum(2)), dim=2)
        # update new feature
        cat_v = torch.cat((v, v_update), dim=2)
        cat_q = torch.cat((q, q_update), dim=2)
        updated_v = self.v_output(cat_v)
        updated_q = self.q_output(cat_q)
        return updated_v, updated_q


class OneSideInterModalityUpdate(nn.Module):
    """
    One-Side Inter-modality Attention Flow
    
    According to original paper, instead of parallel V->Q & Q->V, we first to V->Q and then Q->V
    """
    def __init__(self, src_size, tgt_size, output_size, num_head, drop=0.0, apply_mask=True):
        super(OneSideInterModalityUpdate, self).__init__()
        self.src_size = src_size
        self.tgt_size = tgt_size
        self.output_size = output_size
        self.num_head = num_head

        self.src_lin = FCNet(src_size, output_size * 2, drop=drop)
        self.tgt_lin = FCNet(tgt_size, output_size, drop=drop)

        self.tgt_output = FCNet(output_size + tgt_size, output_size, drop=drop)
        self.apply_mask = apply_mask

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        src: src feature      [batch, num_src, feat_size]
        tgt: tgt feautre      [batch, num_tgt, feat_size]
        src_mask              [batch, num_src]
        tgt_mask              [batch, num_tgt]
        """
        batch_size, num_src = src_mask.shape
        _         , num_tgt = tgt_mask.shape
        
        src_trans = self.src_lin(src)
        tgt_trans = self.tgt_lin(tgt)
        
        if self.apply_mask:
            src_trans = src_trans * src_mask.unsqueeze(2)
            tgt_trans = tgt_trans * tgt_mask.unsqueeze(2)
        
        src_key, src_val = torch.split(src_trans, src_trans.size(2) // 2, dim=2)
        tgt_qry = tgt_trans

        src_key_set = torch.split(src_key, src_key.size(2) // self.num_head, dim=2)
        src_val_set = torch.split(src_val, src_val.size(2) // self.num_head, dim=2)
        tgt_qry_set = torch.split(tgt_qry, tgt_qry.size(2) // self.num_head, dim=2)
        for i in range(self.num_head):
            src_key_slice, tgt_qry_slice, src_val_slice = src_key_set[i], tgt_qry_set[i], src_val_set[i]
            src2tgt = (tgt_qry_slice @ src_key_slice.transpose(1,2)) / ((self.output_size // self.num_head) ** 0.5)  #[batch, tgt_num, src_num]
            if self.apply_mask:
                src2tgt.masked_fill_(src_mask.unsqueeze(1).expand([batch_size, num_tgt, num_src]) == 0, -float('inf'))
            interMAF_src2tgt = F.softmax(src2tgt, dim=2).unsqueeze(3)
            tgt_update = (interMAF_src2tgt * src_val_slice.unsqueeze(1)).sum(2) if (i==0) else torch.cat((tgt_update, (interMAF_src2tgt * src_val_slice.unsqueeze(1)).sum(2)), dim=2)
        cat_tgt = torch.cat((tgt, tgt_update), dim=2)
        update_tgt = self.tgt_output(cat_tgt)
        return update_tgt


class DyIntraModalityUpdate(nn.Module):
    """
    Dynamic Intra-modality Attention Flow
    """
    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0, apply_mask=True):
        super(DyIntraModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head

        self.v4q_gate_lin = FCNet(v_size, output_size, drop=drop)
        self.q4v_gate_lin = FCNet(q_size, output_size, drop=drop)

        self.v_lin = FCNet(v_size, output_size * 3, drop=drop)
        self.q_lin = FCNet(q_size, output_size * 3, drop=drop)

        self.v_output = FCNet(output_size, output_size, drop=drop)
        self.q_output = FCNet(output_size, output_size, drop=drop)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.apply_mask = apply_mask
    
    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        batch_size, num_obj = v_mask.shape
        _         , max_len = q_mask.shape
        # conditioned gating vector
        if self.apply_mask:
            v_mean = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
            q_mean = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1)
        else:
            v_mean = v.sum(1) / num_obj
            q_mean = q.sum(1) / max_len

        v4q_gate = self.sigmoid(self.v4q_gate_lin(v_mean)).unsqueeze(1) #[batch, 1, feat_size]
        q4v_gate = self.sigmoid(self.q4v_gate_lin(q_mean)).unsqueeze(1) #[batch, 1, feat_size]

        # key, query, value
        v_trans = self.v_lin(v)
        q_trans = self.q_lin(q)
        # mask all padding object/word features
        if self.apply_mask:
            v_trans = v_trans * v_mask.unsqueeze(2)
            q_trans = q_trans * q_mask.unsqueeze(2)
        # split for different use of purpose
        v_key, v_qry, v_val = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_key, q_qry, q_val = torch.split(q_trans, q_trans.size(2) // 3, dim=2)
        # apply conditioned gate
        gated_v_qry = (1 + q4v_gate) * v_qry
        gated_v_key = (1 + q4v_gate) * v_key
        gated_v_val = (1 + q4v_gate) * v_val
        gated_q_qry = (1 + v4q_gate) * q_qry
        gated_q_key = (1 + v4q_gate) * q_key
        gated_q_val = (1 + v4q_gate) * q_val

        # apply multi-head
        v_key_set = torch.split(gated_v_key, gated_v_key.size(2) // self.num_head, dim=2)
        v_qry_set = torch.split(gated_v_qry, gated_v_qry.size(2) // self.num_head, dim=2)
        v_val_set = torch.split(gated_v_val, gated_v_val.size(2) // self.num_head, dim=2)
        q_key_set = torch.split(gated_q_key, gated_q_key.size(2) // self.num_head, dim=2)
        q_qry_set = torch.split(gated_q_qry, gated_q_qry.size(2) // self.num_head, dim=2)
        q_val_set = torch.split(gated_q_val, gated_q_val.size(2) // self.num_head, dim=2)
        # multi-head
        for i in range(self.num_head):
            v_key_slice, v_qry_slice, v_val_slice = v_key_set[i], v_qry_set[i], v_val_set[i]  #[batch, num_obj, feat_size]
            q_key_slice, q_qry_slice, q_val_slice = q_key_set[i], q_qry_set[i], q_val_set[i]  #[batch, max_len, feat_size]
            # calculate attention
            v2v = (v_qry_slice @ v_key_slice.transpose(1,2)) / ((self.output_size // self.num_head) ** 0.5)
            q2q = (q_qry_slice @ q_key_slice.transpose(1,2)) / ((self.output_size // self.num_head) ** 0.5)

            if self.apply_mask:
                v2v.masked_fill_(v_mask.unsqueeze(1).expand([batch_size, num_obj, num_obj]) == 0, -float('inf')) 
                q2q.masked_fill_(q_mask.unsqueeze(1).expand([batch_size, max_len, max_len]) == 0, -float('inf')) 
            dyIntraMAF_v2v = F.softmax(v2v, dim=2).unsqueeze(3) #[batch, num_obj, num_obj, 1]
            dyIntraMAF_q2q = F.softmax(q2q, dim=2).unsqueeze(3) #[batch, max_len, max_len, 1]
            # calculate update input
            v_update = (dyIntraMAF_v2v * v_val_slice.unsqueeze(1)).sum(2) if (i==0) else torch.cat((v_update, (dyIntraMAF_v2v * v_val_slice.unsqueeze(1)).sum(2)), dim=2)
            q_update = (dyIntraMAF_q2q * q_val_slice.unsqueeze(1)).sum(2) if (i==0) else torch.cat((q_update, (dyIntraMAF_q2q * q_val_slice.unsqueeze(1)).sum(2)), dim=2)
        # update
        updated_v = self.v_output(v + v_update)
        updated_q = self.q_output(q + q_update)
        return updated_v, updated_q