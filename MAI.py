import torch
import torch.nn as nn
import torch.nn.functional as F


class MAI(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(MAI, self).__init__()
        TEXT_DIM = 768
        VISUAL_DIM = 2048
        BEHAVIOR_DIM = 4
        self.W_cv = nn.Linear(VISUAL_DIM + TEXT_DIM, TEXT_DIM)
        self.W_ca = nn.Linear(BEHAVIOR_DIM + TEXT_DIM, TEXT_DIM)

        self.W_v = nn.Linear(VISUAL_DIM, TEXT_DIM)
        self.W_a = nn.Linear(BEHAVIOR_DIM, TEXT_DIM)

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, L_i, V_i, A_i):
        c_vi = F.relu(self.W_cv(torch.cat((V_i, L_i), dim=-1)))
        c_ai = F.relu(self.W_ca(torch.cat((A_i, L_i), dim=-1)))

        N_i = c_vi * self.W_v(V_i) + c_ai * self.W_a(A_i)

        M_i = L_i + L_i * N_i

        output = self.dropout(
            self.LayerNorm(M_i)
        )

        return output
