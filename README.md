# Metro Pictogram Detection

This project detects Paris metro line pictograms in challenge images.

## Pipeline
1. **Preprocessing**: each input image is resized, converted in LAB color
space and enhanced with CLAHE then blurred.
2. **Candidate detection**: colours typical of metro lines are segmented and
contours are analysed with a circularity criterion to keep potential
pictograms.
3. **Classification**: every candidate region is classified by a small SVM
model if `metro_classifier.pkl` is available, otherwise a colour heuristic is
used.
4. **Output**: detections are returned in the format expected by the
evaluation script and can be visualised with `visual_debug`.

To train the classifier run `python train_classifier.py` once you have the
`Apprentissage.mat` dataset in this folder.
