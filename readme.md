# Rainforest scene classification

Classification of satellite images of the Amazon rainforest using features and a random forest classifier.

### Feature extractors

All of the implemented features extractors can be found in `main/feature_extractors.py`. Feature extractors are implemented as scikit-learn transformers. Thus, to extract features from a set of images, first create an instance of a particular feature class and then call the `.fit_transform()` method, passing your images as arguments to the function

```python
from feature_extractors import ChannelsFeatureExtractor
fx = ChannelsFeatureExtractor()
features = fx.fit_transform(your_images)
```

#### Multiple features

Because the feature extractors are scikit-learn transformers, multiple features can be obtained in parallel using `FeatureUnion`

```python
from feature_extractors import ChannelsFeatureExtractor, NDVIFeatureExtractor
from sklearn.pipeline import FeatureUnion

spectral_fx = FeatureUnion(transformer_list=[
    ("spectral", SpectralFeatureExtractor()),
    ("ndvi", NDVIFeatureExtractor())
])

spectral_features = spectral_fx.fit_transform(your_images)
```

### Classifier

Such features can subsequently be used for training a classifier

```python
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=500, n_jobs=-1)
classifier.fit(features, image_labels)
```

And the trained classifier can be used to make predictions

```python
predictions = classifier.predict(features_test_data)
```
