# Retinal Exudate Detection and Segmentation

**University research project | Python | Image processing | Machine learning**

An academic project exploring the detection and segmentation of retinal exudates in fundus photographs. We studied published image-processing methods, adapted a reference approach to exudate detection, and compared it with an alternative segmentation pipeline. Both approaches include a blob-level Support Vector Machine (SVM) classifier.

This is a **research/educational implementation**, not a medical diagnostic tool.

## What the project does

The code implements two approaches for identifying candidate exudate regions:

| Stage | Reference-inspired approach | Alternative approach |
| --- | --- | --- |
| Contrast enhancement | KMMRC on the HSV value channel | CLAHE on the HSV value channel |
| Candidate segmentation | INRG region-growing method | Adaptive thresholding |
| Common post-processing | Optic-disc removal, morphology, watershed | Optic-disc removal, morphology, watershed |
| Candidate classification | Blob features + SVM | Blob features + SVM |

The comparison is implemented in `main.py`. It performs image-level **5-fold cross-validation** and calculates segmentation metrics against available ground-truth masks. The code is designed to compare approaches, not to claim a universally superior method.

## Techniques and tools

- **Python:** NumPy, pandas, SciPy, Matplotlib
- **Image processing:** scikit-image, contrast enhancement, thresholding, morphological operations, optic-disc masking, watershed
- **Machine learning:** scikit-learn, blob feature extraction, feature selection with chi-squared scoring, RBF-kernel SVM
- **Evaluation:** precision, recall, F1, Dice, IoU and accuracy; image-separated training and testing

## Repository structure

```text
.
├── main.py                        # Pipeline comparison and evaluation
├── config.py
├── src/
│   ├── B1_metriques.py             # Segmentation metrics
│   ├── B2_preprocessament.py       # Load images and masks
│   ├── B4_millora_contrast.py      # Contrast enhancement
│   ├── B4_segmentacio.py          # Threshold-based segmentation
│   ├── B5_nick_inrg.py            # NICK / INRG methods
│   ├── B6_disc_optic.py           # Optic-disc detection and masking
│   ├── B7_postprocessament.py     # Morphological cleanup
│   ├── B8_watershed.py            # Separation of connected regions
│   ├── B9_deteccio_blobs.py       # Blob measurements and features
│   └── B10_classificador_SVM.py   # Classifier and mask reconstruction
└── data_samples/                  # Example images and ground-truth masks
```

## Running the code

A Python environment with NumPy, pandas, SciPy, Matplotlib, scikit-image and scikit-learn is required. From the repository root:

```bash
python -m venv .venv
# Activate the environment for your operating system
pip install numpy pandas scipy matplotlib scikit-image scikit-learn
python main.py
```

The script expects the dataset folders under `data_samples/`. Matching image and ground-truth filenames are used to assemble the pairs. Execution and package-version compatibility have **not been independently verified as part of this documentation update**.

Individual processing modules contain demonstration code, so stages can also be explored separately. For example:

```bash
python src/B4_segmentacio.py
```

## Method and evaluation

1. Load a retinal fundus image and its ground-truth annotation.
2. Enhance contrast and identify candidate bright regions using each pipeline.
3. Mask the optic disc, reduce noise with morphology and split connected blobs with watershed.
4. Extract geometric and intensity-based features for candidate blobs.
5. Train the SVM on training images, then evaluate on held-out images.
6. Compare the final segmentation against the ground-truth masks using Dice, IoU and other metrics.

The SVM uses min-max scaling, chi-squared feature selection and an RBF kernel. Image-level splitting is used to avoid including blobs from the same image in both training and test sets.

**Results:** the implementation prints quantitative summaries when run. This README intentionally does not report numeric scores until reproducible evaluation outputs have been checked and documented.

## Scope and limitations

- Academic prototype; no clinical validation or diagnostic claim.
- Performance depends on image quality, annotations, chosen parameters and dataset composition.
- The source code and sample images are separate items: **check the original dataset terms before redistributing or reusing medical imagery**.
- The work was developed collaboratively; the repository represents a team project.

## References and acknowledgements

The implementation is informed by published methods for retinal image processing. The original papers and dataset licenses should be listed here with complete citations before treating the repository as a reproducible research release.

AI tools were used for limited coding/documentation assistance during development, as documented in source comments; technical decisions and evaluation still require human review.
