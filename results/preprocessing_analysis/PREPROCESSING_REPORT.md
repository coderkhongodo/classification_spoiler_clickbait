# PREPROCESSING DATA ANALYSIS REPORT
==================================================

## Task 1: Spoiler Generation

### Train Data
- Samples: 3,200
- Input Length: μ=403.7, σ=128.6
- Target Length: μ=18.5, σ=21.8
- Truncated Samples: 1445 (45.2%)

### Validation Data
- Samples: 800
- Input Length: μ=405.5, σ=128.6
- Target Length: μ=19.2, σ=23.6
- Truncated Samples: 382 (47.8%)

## Task 2: Spoiler Classification

### Train Data
- Samples: 3,200
- Label Distribution:
  - phrase: 1367 (42.7%)
  - passage: 1274 (39.8%)
  - multi: 559 (17.5%)
- Text Features:
  - Post Length: μ=10.8
  - Spoiler Length: μ=14.6
  - Target Length: μ=514.8

### Validation Data
- Samples: 800
- Label Distribution:
  - phrase: 335 (41.9%)
  - passage: 322 (40.2%)
  - multi: 143 (17.9%)
- Text Features:
  - Post Length: μ=10.9
  - Spoiler Length: μ=15.2
  - Target Length: μ=542.9

## Embeddings

### Train Embeddings
- post_text: (3200, 384) (4.7 MB)
- spoiler: (3200, 384) (4.7 MB)
- target_paragraphs: (3200, 384) (4.7 MB)
- target_keywords: (3200, 384) (4.7 MB)

### Validation Embeddings
- post_text: (800, 384) (1.2 MB)
- spoiler: (800, 384) (1.2 MB)
- target_paragraphs: (800, 384) (1.2 MB)
- target_keywords: (800, 384) (1.2 MB)
