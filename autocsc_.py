import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from transformers import BertPreTrainedModel, BertModel, BertForMaskedLM
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler

class AutoCSCReLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.vocab_size
        self.softmax = nn.Softmax(-1)

    def forward(self, src_ids, trg_ids, attention_mask):
        labels = trg_ids.clone()
        labels[(src_ids == trg_ids)] = -100

        outputs = self.bert(
            input_ids=src_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]
        logits = self.cls(sequence_output)

        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        probs = self.softmax(logits)
        _, predict_ids = torch.max(probs, -1)

        return {
            "loss": loss,
            "predict_ids": predict_ids,
        }

class AutoCSCCOIN(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.vocab_size
        self.softmax = nn.Softmax(-1)
        self.bert = new_BertModel(config, add_pooling_layer=False)

    def forward(self, src_ids, trg_ids, attention_mask, error_pos):
        labels = trg_ids.clone()
        labels[(src_ids == trg_ids)] = -100

        outputs = self.bert(
            input_ids=src_ids,
            attention_mask=attention_mask,
            error_pos = error_pos,
        )

        sequence_output = outputs[0]
        logits = self.cls(sequence_output)

        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        probs = self.softmax(logits)
        _, predict_ids = torch.max(probs, -1)

        return {
            "loss": loss,
            "predict_ids": predict_ids,
        }


class AutoCSCfinetune(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.vocab_size

        self.bert = BertModel(config, add_pooling_layer=False)
        self.generate_linear = nn.Linear(config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(-1)

        self.post_init()

    def forward(self, src_ids, trg_ids, attention_mask):
        outputs = self.bert(
            src_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]

        logits = self.generate_linear(sequence_output)
        probs = self.softmax(logits)
        _, predict_ids = torch.max(probs, -1)
        predict_ids = predict_ids.masked_fill(attention_mask == 0, 0)

        loss_fct = nn.CrossEntropyLoss(ignore_index=0) # ignore padding
        loss = loss_fct(logits.view(-1, self.num_labels), trg_ids.view(-1))

        return {
            "loss": loss,
            "predict_ids": predict_ids,
        }


class AutoCSCSoftMasked(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.vocab_size

        self.bert = BertModel(config, add_pooling_layer=False)
        self.generate_linear = nn.Linear(config.hidden_size, self.num_labels)

        self.rnn = nn.GRU(config.hidden_size, 256, batch_first=True,
                          bidirectional=True, num_layers=2, dropout=0.2)

        self.mask_embed = nn.Linear(1, config.hidden_size, bias=False)
        self.copy_linear = nn.Linear(256 * 2, 1)

        self.softmax = nn.Softmax(-1)
        self.sigmoid = nn.Sigmoid()
        
        self.post_init()

    def forward(self, src_ids, trg_ids, attention_mask):
        embedding_output = self.bert.embeddings(src_ids)
        detect_embed = self.bert.embeddings.word_embeddings(src_ids)

        rnn_hidden_states, _ = self.rnn(detect_embed)
        copy_logits = self.copy_linear(rnn_hidden_states)
        copy_probs = self.sigmoid(copy_logits)
        embedding_output = copy_probs * embedding_output + self.mask_embed(1 - copy_probs)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_output = self.bert.encoder(embedding_output, extended_attention_mask)[0]
        sequence_output = sequence_output + embedding_output

        logits = self.generate_linear(sequence_output)
        probs = self.softmax(logits)
        _, predict_ids = torch.max(probs, -1)
        predict_ids = predict_ids.masked_fill(attention_mask == 0, 0)

        loss_fct = nn.CrossEntropyLoss(ignore_index=0) # ignore padding
        loss = loss_fct(logits.view(-1, self.num_labels), trg_ids.view(-1))

        b_loss_fct = nn.BCEWithLogitsLoss(reduction="none")
        b_logits_loss = b_loss_fct(copy_logits.view(-1), (src_ids == trg_ids).float().view(-1))
        
        b_loss = (b_logits_loss * attention_mask.view(-1)).mean()
        loss = 0.8 * loss + (1 - 0.8) * b_loss

        return {
            "loss": loss,
            "predict_ids": predict_ids,
        }


class AutoCSCMDCSpell(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.vocab_size

        self.bert = BertModel(config, add_pooling_layer=False)
        self.generate_linear = nn.Linear(config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(-1)

        self.detect_layers = deepcopy(self.bert.encoder.layer[:2])
        self.detect_sigmoid = nn.Sequential(nn.Linear(config.hidden_size, 1), nn.Sigmoid())
        self.emb = self.bert.embeddings
        
        self.post_init()

    def forward(self, src_ids, trg_ids, attention_mask):
        outputs = self.bert(
            src_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]

        position_ids = self.emb.position_ids[:, : src_ids.shape[1]]
        token_type_ids = torch.zeros(*src_ids.shape, dtype=torch.long, device=position_ids.device)
        detect_output = self.emb.word_embeddings(src_ids) + self.emb.position_embeddings(position_ids) + self.emb.token_type_embeddings(token_type_ids)
        for layer in self.detect_layers:
            detect_output = layer(detect_output)[0]
        detect_probs = self.detect_sigmoid(detect_output)

        logits = self.generate_linear(sequence_output + detect_output)
        probs = self.softmax(logits)
        _, predict_ids = torch.max(probs, -1)
        predict_ids = predict_ids.masked_fill(attention_mask == 0, 0)

        loss_fct = nn.CrossEntropyLoss(ignore_index=0) # ignore padding
        loss = loss_fct(logits.view(-1, self.num_labels), trg_ids.view(-1))

        detect_labels = (src_ids != trg_ids).float()
        detect_loss_fct = nn.BCEWithLogitsLoss(size_average=True)
        detect_loss = detect_loss_fct(detect_probs.squeeze(-1) * attention_mask, detect_labels)
        loss = 0.85 * loss + 0.15 * detect_loss

        return {
            "loss": loss,
            "predict_ids": predict_ids,
        }


class new_BertModel(BertModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()
        def seq_Gd(vari=0.5,sample_step=0.3,limit=0.3):
            i=0.0
            distribute_sample=[]
            while True:
                temp=np.exp(-i*i/(2*vari*vari))/(np.sqrt(2*np.pi)*vari)
                distribute_sample.append(temp)
                if temp<limit:
                    break
                i=i+sample_step
            return distribute_sample
    
        self.distribution = seq_Gd()

    def seq_Gd_trans(self,output_prob,distribute_sample,b,l):
        output_prob_copy=0*torch.ones(b,l)
        if output_prob is not None:
            output_prob_copy.requires_grad = False
            dl=len(distribute_sample)
            for i in range(b):
                for j in range(l):
                    if(output_prob[i][j]==1):
                        for k in range(dl):
                            if (j+k<l):output_prob_copy[i][k+j]+=distribute_sample[k]
                            if(j-k>=0):output_prob_copy[i][j-k]+=distribute_sample[k]
        
        return output_prob_copy

    def get_gauss(self, error_pos,b,l):
        distribution = self.distribution
        gauss_pos = self.seq_Gd_trans(error_pos, distribution,b,l)
        return gauss_pos


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        error_pos = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        b,l,d = embedding_output.shape
        gauss_pos = self.get_gauss(error_pos,b,l).to(embedding_output.device)
        # gauss_pos = error_pos.to(embedding_output.device)
        gauss_pos = gauss_pos.unsqueeze(2)
        gauss_pos = gauss_pos.repeat(1,1,d)
        embedding_output = embedding_output + gauss_pos
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

