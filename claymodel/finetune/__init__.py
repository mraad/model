"""
Fine-tuning modules for Clay Foundation Model.

This subpackage contains modules for fine-tuning the Clay model on various
downstream tasks including classification, segmentation, regression, and
embedding extraction. Import from the task modules directly, e.g.
`claymodel.finetune.embedder.factory`; classify and segment need the
`finetune` extra (torchgeo, segmentation-models-pytorch).
"""
