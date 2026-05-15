import torch
from kltn_utils import kltn_const
from torch import nn
from transformers import CLIPTokenizer, GPT2Config, GPT2LMHeadModel


class MLP(nn.Module):
    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ConceptTranslator(nn.Module):
    def __init__(self, clip_model_name):
        super().__init__()

        embedding_size = kltn_const.EMBEDDING_DIM[clip_model_name]
        self.decoder = GPT2LMHeadModel(
            GPT2Config(
                n_layer=12,
                n_head=12,
            )
        )
        self.decoder_embedding_size = self.decoder.transformer.wte.weight.shape[1]

        self.clip_project = MLP((embedding_size, self.decoder_embedding_size))
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, clip_features, tokens=None, project=True, attention_mask=None):
        if project:
            clip_features = self.clip_project(clip_features)
            clip_features = clip_features.reshape(-1, 1, self.decoder_embedding_size)

        if tokens is not None:
            embedding_text = self.decoder.transformer.wte(tokens)
            embedding_cat = torch.cat([clip_features, embedding_text], dim=1)
        else:
            embedding_cat = clip_features

        token_logits = self.decoder(
            inputs_embeds=embedding_cat, attention_mask=attention_mask
        ).logits
        token_logits = token_logits[:, :-1]
        token_logits = token_logits.reshape(-1, token_logits.shape[-1])

        return token_logits, embedding_cat

    def output_to_token(self, output, temperature):
        logits = output.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)  # B, dim
        logits = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.argmax(logits, -1).unsqueeze(-1)
        return next_token

    def decode(self, clip_features, entry_length=30, temperature=1, batch_size=128):
        tokens = []

        with torch.no_grad():
            for start in range(0, clip_features.shape[0], batch_size):
                end = min(start + batch_size, clip_features.shape[0])
                clip_feature = clip_features[start:end].to(self.device)
                clip_feature = clip_feature / clip_feature.norm(dim=-1, keepdim=True)
                outputs, embedding_cat = self.forward(clip_feature, project=True)
                bc_tokens = []
                for i in range(entry_length):
                    next_token = self.output_to_token(outputs, temperature)
                    outputs, embedding_cat = self.forward(
                        embedding_cat, next_token, project=False
                    )
                    bc_tokens.append(next_token)
                bc_tokens = torch.cat(bc_tokens, dim=1).cpu()
                tokens.append(bc_tokens)

        tokens = torch.cat(tokens, dim=0).numpy()
        output_list = []

        for token_step in list(tokens):
            output = self.untokenize(token_step)
            output_list.append(output)

        return output_list if len(output_list) > 1 else output_list[0]

    def untokenize(self, tokens):
        output = []

        eos_token_id = self.tokenizer.eos_token_id  # thường là 49407

        for token in tokens.squeeze():
            token_id = token.item() if torch.is_tensor(token) else int(token)

            # bỏ qua token ngoài vocab nếu có
            if token_id >= self.tokenizer.vocab_size:
                break

            output.append(token_id)

            # gặp end-of-text thì dừng
            if token_id == eos_token_id:
                break

        text = self.tokenizer.decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return text.strip()

    def load(self, weight_path):
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
        self.load_state_dict(checkpoint)
