# Phase-Aware and Rotation-Consistent Weakly Supervised Transformer for Brain Tumor MR Segmentation

This repository contains the official implementation of the manuscript:

> **Phase-Aware and Rotation-Consistent Weakly Supervised Transformer for Brain Tumor MR Segmentation**

**Status:** Manuscript under review at *Journal of Translational Internal Medicine*.


---

## Code Organization and Key Components

- **`wetrPRWT`** is the proposed model. The **Polar-Gated Spectral Mixing Block (PoSMB)** is implemented in `wetrPRWT/mix_transformer.py`.

- **`scripts/server_dist_train_brats.py`** is the training entry point, which contains the **Joint Rotation Consistency (JRC)** module.

- **`scripts/infer_seg_extended_metrics.py`** is the inference script used in the manuscript. **All compared methods reported in the manuscript use this same script**; they differ only in how the model is loaded (i.e., which checkpoint/architecture is instantiated). This ensures a consistent evaluation protocol across all methods.

- **`datascript/databrats21`** and **`datascript/dataMen`** contain the preprocessing pipelines for the two datasets, respectively. The preprocessing **strictly follows the protocol described in the CTAC paper [1]**, ensuring fair comparison with prior work.

- **`Figure_sourceData.zip`** contains numerical source data associated with the figures presented in the manuscript.

[1] Channel-Gated Transformers With Affinity CAM for Weakly Supervised Multi-Class Brain Tumor Segmentation, IEEE, 2025, pp. 1–14.
