import os
import time

from dataset import build_dataset, add_preprocessing
from model import initialize_model, initialize_processor
from trainer import build_trainer
from utils import not_change_test_dataset
from pprint import pprint
from utils import set_random_seeds
import torch

# Configuration Constants
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    """
    Main function to execute model training and evaluation.
    """
    # Set seed for reproducibility
    set_random_seeds()

    # Build the dataset
    raw_datasets = build_dataset()

    # assert not_change_test_dataset(raw_datasets), "You should not change the test dataset"

    # Initialize the image processor
    processor = initialize_processor()

    # Add preprocessing to the dataset
    datasets = add_preprocessing(raw_datasets, processor)

    # Build the object detection model
    model = initialize_model()

    # batch = datasets['train'][0]
    # batch['pixel_values'] = batch['pixel_values'].unsqueeze(0).cuda()
    # batch['pixel_mask'] = batch['pixel_mask'].unsqueeze(0).cuda()
    # batch['labels'] = [{key: value.cuda() for key,value in batch['labels'].items()}]
    # model.cuda()
    # breakpoint()
    # outputs = model(**batch)
    # breakpoint()
    # target_sizes = [[480, 640]]
    # results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]
    # for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    #     box = [round(i, 2) for i in box.tolist()]
    #     print(
    #         f"Detected {model.config.id2label[label.item()]} with confidence "
    #         f"{round(score.item(), 3)} at location {box}"
    #     )
    # breakpoint()

    # Build and train the model
    trainer = build_trainer(
        model=model,
        processor=processor,
        datasets=datasets,
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"Training time: {(end_time - start_time) / 60:.2f} minutes")

    # Evaluate the model on the test dataset
    test_metrics = trainer.evaluate(
        eval_dataset=datasets["test"],
        metric_key_prefix="test",
    )
    pprint(test_metrics)


if __name__ == "__main__":
    main()
