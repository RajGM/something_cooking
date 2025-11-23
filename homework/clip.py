from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments

from .base_vlm import BaseVLM
from .data import CaptionDataset, MultiChoiceQADataset

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load(model_name: str = "clip_model"):
    from pathlib import Path

    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    vlm = BaseVLM()
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model

    clip = CLIP(vision_encoder, text_encoder)

    # Load PEFT wrapper
    clip = PeftModel.from_pretrained(clip, model_path)
    # Move underlying model to device
    clip.model.to(device)

    # If there are any Lazy modules left, initialize them by doing a dummy forward
    with torch.no_grad():
        clip.model.eval()
        dummy_pixel = torch.randn(1, 3, 192, 192, device=device)
        dummy_input_ids = torch.ones((1, 8), dtype=torch.long, device=device)
        dummy_attn = torch.ones_like(dummy_input_ids, device=device)
        try:
            _ = clip.model(dummy_pixel, dummy_input_ids, dummy_attn)
        except Exception:
            # ignore forward errors here (we just want lazy init if possible)
            pass

    # Load extra projection weights if present
    clip.model.load_pretrained(model_path)
    clip.model.eval()
    if device == "cuda":
        # safe to convert after lazy params (if any) were (attempted) to be initialized
        clip = clip.to(dtype=torch.bfloat16)

    return clip


def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Custom data collator for CLIP training.
    """
    # Get max sequence length
    max_length = max(f["input_ids"].shape[0] for f in features)

    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_tensor(f["input_ids"], pad_value=processor.tokenizer.eos_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], pad_value=0) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])  # assume all are same shape
    labels = torch.stack([pad_tensor(f["labels"], pad_value=-100) for f in features])

    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "pixel_values": pixel_values.float(),
        "labels": labels.long(),
    }


class CaptionDatasetForTraining(Dataset):
    def __init__(self, dataset: CaptionDataset, processor: AutoProcessor):
        self.dataset = dataset
        self.image_processor = tv.transforms.Compose(
            [
                tv.transforms.Resize(192),
                tv.transforms.RandomResizedCrop(192, scale=(0.5, 1.0)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.image_processor(image)
        text = item["caption"] + self.processor.tokenizer.eos_token
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        input_ids = text_inputs["input_ids"].squeeze(0).long()
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,  # placeholder to fit the collator
        }


class CLIP(nn.Module):
    def __init__(
        self, vision_encoder: nn.Module, text_encoder: nn.Module, proj_dim: int = 64, temperature: float = 0.07
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        # Try to read hidden sizes from encoder configs to avoid LazyLinear.
        v_hidden = None
        t_hidden = None
        try:
            v_cfg = getattr(self.vision_encoder, "config", None)
            t_cfg = getattr(self.text_encoder, "config", None)
            if v_cfg is not None:
                v_hidden = getattr(v_cfg, "hidden_size", None) or (
                    getattr(v_cfg, "hidden_sizes", None)[-1] if getattr(v_cfg, "hidden_sizes", None) else None
                )
            if t_cfg is not None:
                t_hidden = getattr(t_cfg, "hidden_size", None) or (
                    getattr(t_cfg, "hidden_sizes", None)[-1] if getattr(t_cfg, "hidden_sizes", None) else None
                )
        except Exception:
            v_hidden = None
            t_hidden = None

        # If we have the hidden sizes, create explicit Linear layers. Otherwise fall back to LazyLinear.
        if v_hidden is not None and t_hidden is not None:
            self.vision_proj = nn.Linear(v_hidden, proj_dim, bias=False)
            self.text_proj = nn.Linear(t_hidden, proj_dim, bias=False)
            self._used_lazy = False
        else:
            # LazyLinear as a fallback; we'll ensure initialization before dtype conversion / training.
            self.vision_proj = nn.LazyLinear(proj_dim, bias=False)
            self.text_proj = nn.LazyLinear(proj_dim, bias=False)
            self._used_lazy = True

        # Learnable logit scale as in CLIP: logits = exp(logit_scale) * cos_sim
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

        self.proj_dim = proj_dim

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(image)

    def encode_text(self, text: str) -> torch.Tensor:
        return self.text_encoder(text)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Customize save method, save additional parameters"""

        additional_state_dict = {}
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            additional_state_dict[name] = param.data

        torch.save(additional_state_dict, Path(save_directory) / "additional_weights.pt")

    def load_pretrained(self, load_directory: str, **kwargs):
        """Customize load method, load projection additional parameters"""

        additional_weights_path = Path(load_directory) / "additional_weights.pt"
        if additional_weights_path.exists():
            additional_state_dict = torch.load(additional_weights_path, map_location="cpu")

            for name, param in self.named_parameters():
                if "vision_encoder." in name or "text_encoder." in name:
                    continue
                param.data = additional_state_dict[name]

    def set_trainable_parameters(self):
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            param.requires_grad = True

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Enable gradient checkpointing for the vision and text backbones.
        (You don't need to touch this method)
        """
        self.vision_encoder.gradient_checkpointing_enable(**kwargs)
        self.text_encoder.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        """
        Enable input require grads for the vision and text backbones.
        (You don't need to touch this method)
        """

        # Reference: https://discuss.huggingface.co/t/peft-lora-gpt-neox-backward-pass-failing/35641
        def make_inputs_require_grads(module, input, output):  # noqa: A002
            output.requires_grad_(True)

        if hasattr(self.vision_encoder, "embeddings"):
            self.vision_encoder.embeddings.register_forward_hook(make_inputs_require_grads)
        if hasattr(self.text_encoder, "get_input_embeddings"):
            self.text_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the CLIP model.
        Args:
            pixel_values: [B, C, H, W] float tensor, preprocessed image.
            input_ids: [B, L] token ids.
            attention_mask: [B, L] attention mask.
            labels: The labels for the text features (unused here, kept for Trainer).
        Returns:
            (image_features, text_features, logits_per_image)
                image_features: [B, D] normalized image embeddings
                text_features:  [B, D] normalized text embeddings
                logits_per_image: [B, B] similarity logits (image->text)
        """
        # --- Encode images ---
        vision_out = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        if hasattr(vision_out, "last_hidden_state"):
            vision_hidden = vision_out.last_hidden_state  # [B, N, H]
        else:
            vision_hidden = vision_out[0]

        # Mean-pool image tokens
        image_embeds = vision_hidden.mean(dim=1)  # [B, H]

        # --- Encode text ---
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if hasattr(text_out, "last_hidden_state"):
            text_hidden = text_out.last_hidden_state  # [B, L, H]
        else:
            text_hidden = text_out[0]

        mask = attention_mask.unsqueeze(-1).type_as(text_hidden)  # [B, L, 1]
        text_embeds = (text_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)  # [B, H]

        # --- Project to shared space ---
        image_features = self.vision_proj(image_embeds)  # [B, D]
        text_features = self.text_proj(text_embeds)      # [B, D]

        # --- Normalize CLIP-style ---
        image_features = nn.functional.normalize(image_features, dim=-1)
        text_features = nn.functional.normalize(text_features, dim=-1)

        # --- Similarity logits ---
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = logit_scale * image_features @ text_features.t()  # [B, B]

        return image_features, text_features, logits_per_image


def compute_clip_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    num_items_in_batch: int | None = None,
) -> torch.Tensor:
    """
    Compute the symmetric CLIP loss.

    Args:
        outputs: (image_features, text_features, logits_per_image)
        labels: The labels for the text features (unused, for Trainer compat).
        num_items_in_batch: Unused (Trainer compat).
    Returns:
        The loss for the CLIP model.
    """
    image_features, text_features, logits_per_image = outputs
    logits = logits_per_image  # [B, B]
    batch_size = logits.size(0)

    target = torch.arange(batch_size, device=logits.device)

    # Image -> Text
    loss_i = nn.functional.cross_entropy(logits, target)
    # Text -> Image
    loss_t = nn.functional.cross_entropy(logits.t(), target)

    loss = (loss_i + loss_t) / 2.0
    return loss


def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    target_modules = []
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and ("vision_encoder" in name or "text_encoder" in name)
            and "projection" not in name
        ):
            target_modules.append(name)

    return target_modules


def train(
    data_dir: Path | None = None,
    train_dataset_name: str = "train",
    output_dir: str = "clip",
    num_train_epochs: float = 0.05,  # for debugging purpose, increase this once the dry run works
    per_device_train_batch_size: int = 1024,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-4,
    num_workers: int = 16,
):
    vlm = BaseVLM()

    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize model and processor
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model

    # Create CLIP instance WITHOUT converting dtype yet
    model = CLIP(vision_encoder, text_encoder)

    # Move to device so that any parameter creation happens on correct device
    model = model.to(device)

    # If LazyLinear was used (fallback), initialize by a dummy forward before converting dtype
    if getattr(model, "_used_lazy", False):
        with torch.no_grad():
            model.eval()
            dummy_pixel = torch.randn(1, 3, 192, 192, device=device)
            dummy_input_ids = torch.ones((1, 8), dtype=torch.long, device=device)
            dummy_attn = torch.ones_like(dummy_input_ids, device=device)
            try:
                _ = model(dummy_pixel, dummy_input_ids, dummy_attn)
            except Exception:
                # If the forward fails, continue — conversion might still fail but user will get a clear error.
                pass

    # Now safe to convert to bfloat16 if using CUDA and parameters are initialized
    if device == "cuda":
        model = model.bfloat16()

    model.set_trainable_parameters()

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=get_target_modules_for_lora(model),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # load dataset
    train_dataset = CaptionDataset(train_dataset_name, data_dir)
    train_dataset = CaptionDatasetForTraining(train_dataset, processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        bf16=True if device == "cuda" else False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        label_names=["labels"],
        dataloader_num_workers=num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
        compute_loss_func=compute_clip_loss,
    )

    trainer.train()

    # save model
    trainer.save_model(output_dir)
    # model is a PeftModel wrapper here; .model is the underlying model with save_pretrained method
    model.model.save_pretrained(output_dir)

    writer.close()

    return model, processor


def demo_train():
    train(
        train_dataset_name="train_demo",
        output_dir="demo_clip",
        num_train_epochs=10,
        per_device_train_batch_size=64,
        num_workers=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
    )


def test(ckpt_path: str, val_dataset: str = "valid_grader"):
    import tqdm

    testset = MultiChoiceQADataset(val_dataset)

    clip = load(ckpt_path)
    clip = clip.model.to(device)

    image_processor = tv.transforms.Compose(
        [
            tv.transforms.Resize(192),
            tv.transforms.CenterCrop(192),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    correct_count = 0
    total_count = 0

    for pair in tqdm.tqdm(testset):
        image = Image.open(pair["image_path"]).convert("RGB")
        pixel_values = image_processor(image).unsqueeze(0).to(device).bfloat16()
        text_inputs = processor(
            text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = text_inputs["input_ids"].long().to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        vision_feature, text_feature, _ = clip(pixel_values, input_ids, attention_mask)
        prediction = torch.matmul(vision_feature, text_feature.T).argmax(dim=-1)
        if prediction == pair["correct_index"]:
            correct_count += 1
        total_count += 1

    print(f"Accuracy: {correct_count / total_count}")


def main():
    from fire import Fire

    Fire({"train": train, "test": test})


if __name__ == "__main__":
    main()
