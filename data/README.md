The CounterCount benchmark data is provided to reviewers via the link.

Place the dataset here with the following structure:

```
data/CounterCount/
├── Birds/
│   ├── Original/          # Factual images (img0.png, img1.png, ...)
│   ├── Anomaly/           # Counterfactual images
│   ├── masks/             # Segmentation masks + bbox JSON
│   │   ├── img0_mask.png
│   │   ├── birds_bbox.json
│   │   └── ...
│   ├── birds_prompts.json
│   └── birds_metadata.json
├── Mammals/
│   └── ... (same structure)
├── Housing/
├── Functional/
├── Landmarks/
├── Transportation/
├── Bugs/
├── Sea/
├── Food/
└── Currency/
```
