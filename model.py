from transformers import AutoModelForObjectDetection, AutoImageProcessor
from constants import ID2LABEL, LABEL2ID, MODEL_NAME
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection



def initialize_model():
    """
    Initialize a model for object detection.

    Returns:
        A model for object detection.

    NOTE: Below is an example of how to initialize a model for object detection.

    from transformers import AutoModelForObjectDetection
    from constants import ID2LABEL, LABEL2ID, MODEL_NAME

    model = AutoModelForObjectDetection.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,  # specify the model checkpoint
        id2label=ID2LABEL,  # map of label id to label name
        label2id=LABEL2ID,  # map of label name to label id
        ignore_mismatched_sizes=True,  # allow replacing the classification head
    )

    You are free to change this.
    But make sure the model meets the requirements of the `transformers.Trainer` API.
    ref: https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer
    """
    # Write your code here.
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    model = AutoModelForObjectDetection.from_pretrained(
        "hustvl/yolos-base",
        # revision="no_timm",
        cache_dir="./hfmodel",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    # for name, param in model.named_parameters():
    #     if 'class_labels_classifier' in name or 'bbox_predictor' in name:
    #         param.requires_grad_(True)
    #     else:
    #         param.requires_grad_(False)

    # target_modules = []
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Linear) and not name.startswith("model.backbone"):
    #         target_modules.append(name)

    # # Define LoRA configuration
    # lora_config = LoraConfig(
    #     r=16,                        # Rank of the LoRA update matrices
    #     lora_alpha=32,               # Scaling factor for LoRA updates
    #     target_modules=target_modules,  # Target modules (e.g., attention layers)
    #     lora_dropout=0.1,            # Dropout for LoRA layers
    #     bias="none",                 # Whether to adapt biases ("none", "all", or "lora_only")
    #     task_type="SEQ_2_SEQ_LM",    # Task type (e.g., sequence-to-sequence for DETR)
    # )

    # # Add LoRA to the model
    # model = get_peft_model(model, lora_config)

    return model


def initialize_processor():
    """
    Initialize a processor for object detection.

    Returns:
        A processor for object detection.

    NOTE: Below is an example of how to initialize a processor for object detection.

    from transformers import AutoImageProcessor
    from constants import MODEL_NAME

    processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME
    )

    You are free to change this.
    But make sure the processor meets the requirements of the `transformers.Trainer` API.
    ref: https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer
    """
    # Write your code here.
    processor = AutoImageProcessor.from_pretrained(
        "hustvl/yolos-base",
        # revision="no_timm",
        cache_dir="./hfmodel"
    )
    # processor = AutoImageProcessor.from_pretrained(
    #     pretrained_model_name_or_path=MODEL_NAME,  # specify the model checkpoint
    #     cache_dir="./hfmodel"
    # )
    return processor


if __name__ == '__main__':
    # Initialize the model and processor
    model = initialize_model()
    processor = initialize_processor()

    # Example usage of the processor
    from PIL import Image

    image = Image.open("sample.png")
    inputs = processor(images=image, return_tensors="pt")

    # Forward pass through the model
    breakpoint()
    outputs = model(**inputs)
    breakpoint()


