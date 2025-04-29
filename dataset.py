from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from datasets import load_dataset, load_from_disk
from functools import partial
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image


def build_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Build the dataset for object detection.

    Returns:
        The dataset.

    Below is an example of how to load an object detection dataset.

    ```python
    from datasets import load_dataset

    raw_datasets = load_dataset("cppe-5")
    if "validation" not in dataset_base:
        split = dataset_base["train"].train_test_split(0.15, seed=1337)
        dataset_base["train"] = split["train"]
        dataset_base["validation"] = split["test"]
    ```

    Ref: https://huggingface.co/docs/datasets/v3.2.0/package_reference/main_classes.html#datasets.DatasetDict

    You can replace this with your own dataset. Make sure to include
    the `test` split and ensure that it is consistent with the dataset format expected for object detection.
    For example:
        raw_datasets["test"] = load_dataset("cppe-5", split="test")
    """
    # Write your code here.
    raw_datasets = load_dataset("cppe-5", cache_dir="./hfdata")
    split = raw_datasets["train"].train_test_split(test_size=0.15, seed=1337)
    raw_datasets = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
        "test": raw_datasets["test"]
    })
    return raw_datasets


def filter_invalid_bboxes(example):
    valid_bboxes = []
    valid_bbox_ids = []
    valid_categories = []
    valid_areas = []

    width, height = example["image"].size

    for i, bbox in enumerate(example["objects"]["bbox"]):
        x_min, y_min, x_span, y_span = bbox[:4]
        x_max = x_min + x_span
        y_max = y_min + y_span
        if x_max <= width + 1e-4 and y_max <= height + 1e-4:
            valid_bboxes.append(bbox)
            valid_bbox_ids.append(example["objects"]["bbox"][i])
            valid_categories.append(example["objects"]["category"][i])
            valid_areas.append(example["objects"]["area"][i])
        else:
            print(
                f"Image with invalid bbox: {example['image_id']} "
                f"Invalid bbox detected and discarded: {bbox} "
                f"- bbox_id: {example['objects']['bbox'][i]} "
                f"- category: {example['objects']['category'][i]}"
            )

    example["objects"]["bbox"] = valid_bboxes
    example["objects"]["category"] = valid_categories
    example["objects"]["area"] = valid_areas

    return example

def preprocess_function_batch(batch, augmentations, processor):
        batch_image_array = []
        batch_target = []

        for image, image_id, objects in zip(batch["image"], batch["image_id"], batch["objects"]):
            # import requests
            # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            # image = Image.open(requests.get(url, stream=True).raw)
            # breakpoint()

            image_array = np.array(image.convert("RGB"))[:, :, ::-1].copy()
            outputs = augmentations(image=image_array, bboxes=objects["bbox"], category_id=objects["category"])
            target = {
                "image_id": image_id,
                "annotations": [{
                    "category_id": outputs["category_id"][i],
                    "bbox": outputs["bboxes"][i],
                    "area": objects["area"][i],
                    "isCrowd": 0,
                } for i in range(len(outputs["category_id"]))]
            }
            batch_image_array.append(outputs["image"])
            batch_target.append(target)

        outputs = processor(images=batch_image_array, annotations=batch_target, return_tensors="pt")
        return outputs

def add_preprocessing(dataset, processor) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Add preprocessing to the dataset.

    Args:
        dataset: The dataset to preprocess.
        processor: The image processor to use for preprocessing.

    Returns:
        The preprocessed dataset.

    In this function, you can add any preprocessing steps to the dataset.
    For example, you can add data augmentation, normalization or formatting to meet the model input, etc.

    Hint:
    # You can use the `with_transform` method of the dataset to apply transformations.
    # Ref: https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.with_transform

    # You can also use the `map` method of the dataset to apply transformations.
    # Ref: https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.map

    # For Augmentation, you can use the `albumentations` library.
    # Ref: https://albumentations.ai/docs/

    from functools import partial

    # Create the batch transform functions for training and validation sets
    train_transform_batch = # Callable for train set transforming with batched samples passed
    validation_transform_batch = # Callable for val/test set transforming with batched samples passed

    # Apply transformations to dataset splits
    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch)
    dataset["test"] = dataset["test"].with_transform(validation_transform_batch)
    """
    # Write your code here.
    for key in dataset.keys():
        dataset[key] = dataset[key].map(filter_invalid_bboxes, batched=False)
        dataset[key] = dataset[key].filter(lambda x: len(x["objects"]["bbox"]) > 0)

    train_augmentations = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.GaussianBlur(p=0.5),
            A.GaussNoise(p=0.5),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category_id"]),
    )

    val_test_augmentations = A.Compose(
        [
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category_id"]),
    )



    preprocess_training = partial(preprocess_function_batch, augmentations=train_augmentations, processor=processor)
    preprocess_validation = partial(preprocess_function_batch, augmentations=val_test_augmentations, processor=processor)

    dataset["train"] = dataset["train"].with_transform(preprocess_training)
    dataset["validation"] = dataset["validation"].with_transform(preprocess_validation)
    dataset["test"] = dataset["test"].with_transform(preprocess_validation)

    return dataset


def test_load_preprocess():
    from datasets import DatasetDict
    from transformers import AutoImageProcessor

    print("Loading dataset...")
    dataset = build_dataset()
    print("Dataset loaded successfully.")
    print(f"Dataset structure: {dataset}")

    print("Initializing image processor...")
    processor = AutoImageProcessor.from_pretrained("hustvl/yolos-base", cache_dir="./hfmodel")
    print("Processor initialized successfully.")

    print("Applying preprocessing...")
    preprocessed_dataset = add_preprocessing(dataset, processor)
    print("Preprocessing completed successfully.")

    a = preprocessed_dataset['train'][0]
    breakpoint()


if __name__ == '__main__':
    test_load_preprocess()
