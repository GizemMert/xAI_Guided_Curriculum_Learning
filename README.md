# Curriculum Scheduler with xAI

This project is developed as part of an Advanced Topics in Machine Learning and Optimization course, which is included in of M.Sc. Artificial Intelligence Systems from University of Trento. It is based on evaluating explainable AI within curriculum learning strategies, implementing a framework designed to enhance the training process of machine learning models. It involves a series of scheduling methods (from easy to hard examples; applied dropout on images), explainability tools, and different model architectures, different datasets. The repository includes utilities for creating datasets, scheduling training, and evaluating explainability.

## Repository Structure
1. Dataset: Dataset.py to load and preprocess the datasets.

2. Train Models: We use different scripts based on scheduling method, dataset, model, etc.:

    - Curriculum training with dropout on cifar dataset : train.py
    - Curriculum training with dropout on food101 dataset : train_f101.py
    - Easy to hard training via explainability score on cifar dataset: train_easy_hard.py
    - Easy to hard training via explainability score on food101 dataset: train_f101_easy_to_hard.py
    - Baseline model for food101 dataset: train_f101_base.py
    - Baseline model for cifar dataset: train_cifar_base.py

3. Explainability Analysis: Explainability.py to analyze trained models and calculate metrics such as sufficiency, infedility, sensitivity based on gradient base explainers.
4. Visualization: visualize_f101.py or visualize.py to visualize attribution maps via explaniers.
5. Curriculum scheduler with dropout: curriculum_scheduler_dropout.py script defines a functionality that gradually introduces dropouts to input images over epochs. It starts after defined warmup epochs.
6. Easy to hard sample defining and curriculum_dataloader: easy_to_hard.py script defines functions to group datasets based on their confidency and new dataloader which scheduling loading easy medium and hard samples based on epoch or selected explainability metric.
7. Models: We use two different pretrained model architecture and train them from sctrach; ConvNeXt and DeiT-Tiny models.
8. Sufficiency score during training: sufficiency_easyhard_schedule.py is designed to calculate sufficiency metric (15% retention proportion) according to training sample attributions during training.
9. Run script: There are some versions of run.sbatch files to show how to train models with different arguments and to control epochs, batch size, learning rate, etc. 
