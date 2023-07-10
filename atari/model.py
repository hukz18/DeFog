import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_cnn_output_dim(input_size, kernel_size, stride, padding):
    return math.floor((input_size - kernel_size + 2 * padding) / stride + 1)

def cnn(input_shape, input_channel, hidden_channels, kernel_sizes, strides, paddings, hidden_dim, output_activation=nn.Identity()):
    layers = []
    embedding_h, embedding_w = input_shape
    for i in range(len(hidden_channels)):
        layers.append(nn.Conv2d(input_channel, hidden_channels[i], kernel_sizes[i], strides[i], paddings[i]))
        layers.append(nn.ReLU())
        input_channel = hidden_channels[i]
        embedding_h = get_cnn_output_dim(embedding_h, kernel_sizes[i], strides[i], paddings[i])
        embedding_w = get_cnn_output_dim(embedding_w, kernel_sizes[i], strides[i], paddings[i])
    layers.append(nn.Flatten())
    layers.append(nn.Linear(embedding_h * embedding_w * hidden_channels[-1], hidden_dim))
    layers.append(output_activation)
    return nn.Sequential(*layers)

class MaskedCausalAttention(nn.Module):
    def __init__(self, hidden_dim, context_len, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.context_len = context_len

        self.q_net = nn.Linear(hidden_dim, hidden_dim)
        self.k_net = nn.Linear(hidden_dim, hidden_dim)
        self.v_net = nn.Linear(hidden_dim, hidden_dim)

        self.proj_net = nn.Linear(hidden_dim, hidden_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((context_len, context_len))
        mask = torch.tril(ones).view(1, 1, context_len, context_len)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape  # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads  # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[..., :T, :T] == 0, float('-inf'))

        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, hidden_dim, context_len, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(hidden_dim, context_len, n_heads, drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x)  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, n_heads, n_blocks, hidden_dim, context_len, drop_p, 
                 action_space, reward_scale, max_timestep, drop_aware, device,
                 cnn_channels, cnn_kernels, cnn_strides, cnn_paddings):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # transformer blocks
        self.context_len = context_len
        blocks = [Block(hidden_dim, 3 * context_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        # projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(hidden_dim)
        self.embed_timestep = nn.Embedding(max_timestep, hidden_dim)
        self.embed_dropstep = nn.Embedding(max_timestep, hidden_dim)
        self.embed_rtg = torch.nn.Linear(1, hidden_dim)
        self.embed_state = cnn(state_dim[1:], state_dim[0], cnn_channels, cnn_kernels, cnn_strides, cnn_paddings, hidden_dim)

        self.embed_action = torch.nn.Embedding(action_dim, hidden_dim)

        # prediction heads
        self.predict_action = nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Tanh())

        self.action_space = action_space
        self.reward_scale = reward_scale

        self.max_timestep = max_timestep
        self.drop_aware = drop_aware
        self.to(device)

    
    def _norm_reward_to_go(self, reward_to_go):
        return reward_to_go / self.reward_scale

    def __repr__(self):
        return "DecisionTransformer"
    
    def freeze_trunk(self):
        freezed_models = [self.embed_state, self.embed_action, self.embed_rtg, self.embed_timestep, self.blocks, self.embed_ln]
        for model in freezed_models:
            for p in model.parameters():
                p.requires_grad = False

    def forward(self, states, actions, rewards_to_go, timesteps, dropsteps):
        states = states.div_(255.0)
        rewards_to_go = self._norm_reward_to_go(rewards_to_go)
        batch_size, context_len = states.shape[:2]

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states.reshape(-1, *self.state_dim)).reshape(batch_size, context_len, -1) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(rewards_to_go) + time_embeddings

        if self.drop_aware:
            drop_embeddings = self.embed_dropstep(dropsteps)
            state_embeddings += drop_embeddings
            returns_embeddings += drop_embeddings
        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=2
        ).reshape(batch_size, 3 * context_len, self.hidden_dim)

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.blocks(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus
        # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.
        h = h.reshape(batch_size, context_len, 3, self.hidden_dim).permute(0, 2, 1, 3)

        # get predictions
        action_logits = self.predict_action(h[:, 1])  # predict action given r, s

        return action_logits
    
    def save(self, save_name):
        os.makedirs('models', exist_ok=True)
        torch.save(self.state_dict(), os.path.join('models', f'{save_name}.pt'))
    
    def load(self, load_name):
        self.load_state_dict(torch.load(os.path.join('models', f'{load_name}.pt')))