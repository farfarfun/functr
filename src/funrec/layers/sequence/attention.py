import torch
import torch.nn as nn
import torch.nn.functional as F

from funrec.layers.core import LocalActivationUnit


class AttentionSequencePoolingLayer(nn.Module):
    """The Attentional sequence pooling operation used in DIN & DIEN.

    Arguments
      - **att_hidden_units**:list of positive integer, the attention net layer number and units in each layer.

      - **att_activation**: Activation function to use in attention net.

      - **weight_normalization**: bool.Whether normalize the attention score of local activation unit.

      - **supports_masking**:If True,the input need to support masking.

    References
      - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(
        self,
        att_hidden_units=(80, 40),
        att_activation="sigmoid",
        weight_normalization=False,
        return_score=False,
        supports_masking=False,
        embedding_dim=4,
        **kwargs,
    ):
        super(AttentionSequencePoolingLayer, self).__init__()
        self.return_score = return_score
        self.weight_normalization = weight_normalization
        self.supports_masking = supports_masking
        self.local_att = LocalActivationUnit(
            hidden_units=att_hidden_units,
            embedding_dim=embedding_dim,
            activation=att_activation,
            dropout_rate=0,
            use_bn=False,
        )

    def forward(self, query, keys, keys_length, mask=None):
        """
        Input shape
          - A list of three tensor: [query,keys,keys_length]

          - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``

          - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``

          - keys_length is a 2D tensor with shape: ``(batch_size, 1)``

        Output shape
          - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.
        """
        batch_size, max_length, _ = keys.size()

        # Mask
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking"
                )
            keys_masks = mask.unsqueeze(1)
        else:
            keys_masks = torch.arange(
                max_length, device=keys_length.device, dtype=keys_length.dtype
            ).repeat(batch_size, 1)  # [B, T]
            keys_masks = keys_masks < keys_length.view(-1, 1)  # 0, 1 mask
            keys_masks = keys_masks.unsqueeze(1)  # [B, 1, T]

        attention_score = self.local_att(query, keys)  # [B, T, 1]

        outputs = torch.transpose(attention_score, 1, 2)  # [B, 1, T]

        if self.weight_normalization:
            paddings = torch.ones_like(outputs) * (-(2**32) + 1)
        else:
            paddings = torch.zeros_like(outputs)

        outputs = torch.where(keys_masks, outputs, paddings)  # [B, 1, T]

        # Scale
        # outputs = outputs / (keys.shape[-1] ** 0.05)

        if self.weight_normalization:
            outputs = F.softmax(outputs, dim=-1)  # [B, 1, T]

        if not self.return_score:
            # Weighted sum
            outputs = torch.matmul(outputs, keys)  # [B, 1, E]

        return outputs
