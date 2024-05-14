import torch.nn as nn
import torch


class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x): # A, B, C   A: dim to self-attend; B: batch size; C: feature channels.
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class TemporalTransformer(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, H, W, max_T, batch_first=False, dropout=0.0):
        """
        Inputs:
            input_dim - Hidden dimensionality of the input
            hidden_dim - Hidden dimensionality to use inside the Transformer
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            max_seq_len - Maximum allowed number of frames of DSA series.
            batch_first - Whether batch is the first dim of the input.
            dropout - Dropout to apply inside the model.
        """

        super().__init__()

        self.batch_first = batch_first

        # self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Sequential(
            *[AttentionBlock(input_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1, 1, max_T, 1))

        '''spatial transformer'''
        # self.spatial_transformer = nn.Sequential(
        #     *[AttentionBlock(input_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        # self.spatial_pos_embedding = nn.Parameter(torch.randn(1, 1, H, W))
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(hidden_dim),
        #     nn.Linear(hidden_dim, input_dim)
        # )

    def forward(self, x):
        """
        Parameters
        ----------
        x: input 5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            x = x.permute(1, 0, 2, 3, 4)
        x = x.permute(0, 3, 4, 1, 2)  # B, H, W, T, C
        B, H, W, T, _ = x.shape
        
        # x = self.input_layer(x)
        B, H, W, T, _ = x.shape
        # Add CLS token and positional encoding
        # cls_token = self.cls_token.repeat(B, H, W, 1, hidden_C)
        # x = torch.cat([cls_token, x], dim=3)
        pos_embedding = self.pos_embedding.repeat(B, H, W, 1, x.shape[-1])
        x = x + pos_embedding[:, :, :, :T, :]

        # Apply Transformer
        x = self.dropout(x)
        x = x.permute(3, 0, 1, 2, 4)  # T, B, H, W, C
        x = x.view(x.shape[0], -1, x.shape[-1])
        x = self.transformer(x)
        x = x.view(x.shape[0], B, H, W, -1)

        x = x[-1].permute(0, 3, 1, 2) # B, C, H, W; We are taking the last time point as output

        '''spatial transformer'''
        # spatial_pos_embedding = self.spatial_pos_embedding.repeat(B, hidden_C, 1, 1)
        # x = x + spatial_pos_embedding
        # # Apply Transforrmer
        # x = self.dropout(x)
        # x = x.permute(2, 3, 0, 1) # H, W, B, C
        # H, W, B, _ = x.shape
        # x = x.view(H*W, B, -1)  # H*W, B, C
        # x = self.spatial_transformer(x)
        # x = x.view(H, W, B, -1)
        # x = x.permute(2, 0, 1, 3)

        # x = x.permute(0, 2, 3, 1)
        # x = self.mlp_head(x)  # B, H, W, C
        # x = x.permute(0, 3, 1, 2) # B, C, H, W

        return x  


if __name__ == "__main__":
    model = TemporalTransformer(input_dim=64, hidden_dim=128, num_heads=4, num_layers=4,
                                max_T=20, batch_first=True, dropout=0.0)
    model = model.cuda()
    print(repr(model))
