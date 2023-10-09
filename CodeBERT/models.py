import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_utils import PreTrainedModel


class ModelContra(PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelContra, self).__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.loss_func = CrossEntropyLoss()
        self.args = args

    def forward(self, code_inputs, nl_inputs, labels=None, code_lens=None, sim_matrix=None, return_vec=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]

        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]

        if return_vec:
            return code_vec, nl_vec

        logits = torch.matmul(nl_vec, code_vec.T)

        loss_mask = labels.diag()
        sim_neg = sim_matrix[loss_mask != 1].view(sim_matrix.size(0), -1)

        if self.args.softinfonce:
            weights = torch.nn.functional.softmax(sim_neg / self.args.t, dim=1)
            alpha, beta = self.args.weight_kl, self.args.weight_unif
            weights = torch.clip((beta - alpha * weights) / (beta - alpha / weights.size(1)), min=0.1)
            weights = torch.cat([torch.ones((weights.size(0), 1), device=weights.device), weights], dim=1)
            scores = torch.cat([logits[loss_mask == 1].unsqueeze(1), logits[loss_mask != 1].view(logits.size(0), -1)],
                               dim=1)
            maxes = torch.max(scores, 1, keepdim=True)[0]
            x_exp = torch.exp(scores - maxes)
            x_exp_sum = torch.sum(weights * x_exp, 1, keepdim=True)
            probs = x_exp / x_exp_sum
            loss = - torch.mean(torch.log(probs[:, 0] + 1e-15))

        if self.args.weightedinfonce:
            weights = torch.nn.functional.softmax(sim_neg / self.args.t, dim=1)
            weights = torch.cat([torch.ones((weights.size(0), 1), device=weights.device), weights], dim=1)
            scores = torch.cat(
                [logits[loss_mask == 1].unsqueeze(1), logits[loss_mask != 1].view(logits.size(0), -1)],
                dim=1)
            scores = torch.nn.functional.softmax(scores, dim=1)
            loss = - torch.mean((torch.log(scores + 1e-15) * weights).sum(1))

        if self.args.bce:
            weights = torch.nn.functional.softmax(sim_neg / self.args.t, dim=1)
            weights = torch.cat([torch.ones((weights.size(0), 1), device=weights.device), weights], dim=1)
            scores = torch.cat([logits[loss_mask == 1].unsqueeze(1), logits[loss_mask != 1].view(logits.size(0), -1)],
                               dim=1)
            loss_fct = torch.nn.BCELoss()
            scores = torch.nn.functional.softmax(scores, dim=1)
            loss = loss_fct(scores, weights)

        if self.args.klregularization:
            weights = torch.nn.functional.softmax(sim_neg / self.args.t, dim=1)
            loss_fct2 = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
            loss2 = loss_fct2(torch.nn.functional.log_softmax(logits[loss_mask != 1].view(logits.size(0), -1), dim=1),
                              weights.log())
            scores = torch.cat([logits[loss_mask == 1].unsqueeze(1), logits[loss_mask != 1].view(logits.size(0), -1)],
                               dim=1)
            loss_fct1 = CrossEntropyLoss()
            loss1 = loss_fct1(scores, torch.zeros(code_inputs.size(0), device=scores.device).long())
            loss = self.args.weight_unif * loss1 + self.args.weight_kl * loss2

        if self.args.infonce:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, torch.arange(code_inputs.size(0), device=logits.device))

        predictions = None
        return loss, predictions
