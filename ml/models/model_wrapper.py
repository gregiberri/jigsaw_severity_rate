from torch import nn


class JigsawModelWrapper(nn.Module):
    def __init__(self, model):
        super(JigsawModelWrapper, self).__init__()
        self.model = model
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(768, 1)

    def forward(self, ids, mask):
        out = self.model(input_ids=ids,
                         attention_mask=mask,
                         output_hidden_states=False)
        out = self.drop(out[1])
        outputs = self.fc(out)
        return outputs