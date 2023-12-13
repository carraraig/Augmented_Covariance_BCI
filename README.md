# Augmented Covariance Method

This repository contain the results and the code for the paper ACM [1].
If you use the code please cite the connected paper.

## Abstract 
Electroencephalography signals are recorded as multidimensional datasets. We propose a new framework based on the augmented covariance that stems from an autoregressive model to improve motor imagery classification.
From the autoregressive model can be derived the Yule-Walker equations, which show the emergence of a symmetric positive definite matrix: the augmented covariance matrix. The state-of the art for classifying covariance matrices is based on Riemannian Geometry. A fairly natural idea is therefore to apply this Riemannian Geometry based approach to these augmented covariance matrices.
The methodology for creating the augmented covariance matrix shows a natural connection with the delay embedding theorem proposed by Takens for dynamical systems. Such an embedding method is based on the knowledge of two parameters: the delay and the embedding dimension, respectively related to the lag and the order of the autoregressive model. This approach provides new methods to compute the hyper-parameters in addition to standard grid search.
The augmented covariance matrix performed noticeably better than any state-of-the-art methods. We will test our approach on several datasets and several subjects using the MOABB framework, using both within-session and cross-session evaluation.
The improvement in results is due to the fact that the augmented covariance matrix incorporates not only spatial but also temporal information. As such, it contains information on the nonlinear components of the signal through the embedding procedure, which allows the leveraging of dynamical systems algorithms.
These results extend the concepts and the results of the Riemannian distance based classification algorithm.

## Requirement
To run the following code you need MOABB:
- Nested Cross Validation using GridSearch you can use MOABB 1.0 [2]
- Nested Cross Validation + MDOP use the branch "https://github.com/carraraig/moabb/tree/Takens_NoParallel_1Metric"

All the packages dependencies are listed in the environment.yml file.

## Example of usage
```python
import moabb
import mne
import resource
from moabb.paradigms import MotorImagery
from moabb.datasets import Zhou2016
from pyriemann.estimation import Covariances
from sklearn.pipeline import Pipeline
from moabb.evaluations import CrossSessionEvaluation
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from pyriemann.tangentspace import TangentSpace
from pyriemann.spatialfilters import CSP
from pyriemann.classification import MDM
from pyriemann.classification import FgMDM

# Import AugmentedDataset from moabb
from moabb.pipelines.features import AugmentedDataset

# Initialize parameter for the Band Pass filter
fmin = 8
fmax = 35
tmin = 0
tmax = None

# Select the Subject
subjects = [1]
# Load the dataset
dataset = Zhou2016()

events = ["right_hand", "feet"]

paradigm = MotorImagery(events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmax=tmax)

# Pipelines
pipelines = {}
pipelines["ACM+TGSP+SVM"] = Pipeline(steps=[
    ("augmenteddataset", AugmentedDataset()),
    ("Covariances", Covariances("cov")),
    ("Tangent_Space", TangentSpace(metric="riemann")),
    ("SVM", SVC(kernel="rbf"))
])

pipelines["ACM+MDM"] = Pipeline(steps=[
    ("augmenteddataset", AugmentedDataset()),
    ("Covariances", Covariances("cov")),
    ("MDM", MDM(metric=dict(mean='riemann', distance='riemann')))
])


# ====================================================================================================================
# GridSearch
# ====================================================================================================================
param_grid = {}
param_grid["ACM+TGSP+SVM"] = {
    'augmenteddataset__order': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'augmenteddataset__lag': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'SVM__C': [0.5, 1, 1.5],
    'SVM__kernel': ["linear", "rbf"],
}

param_grid["ACM+MDM"] = {
    'augmenteddataset__order': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'augmenteddataset__lag': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

# Evaluation For MOABB
# ========================================================================================================
dataset.subject_list = dataset.subject_list[0:1]
# Select an evaluation Within Session
evaluation = CrossSessionEvaluation(paradigm=paradigm,
                                    datasets=dataset,
                                    overwrite=False,
                                    random_state=42,
                                    n_jobs=-1)

# Print the results using Nested GridSearch
result = evaluation.process(pipelines, param_grid, nested=True)

# Print the results using Nested MDOP
result = evaluation.process(pipelines, nested=True, takens="MDOP")
```


## References:
[1] Carrara, I., & Papadopoulo, T. (2023). Classification of BCI-EEG based on augmented covariance matrix. arXiv preprint arXiv:2302.04508.

[2] Aristimunha, B., Carrara, I., Guetschel, P., Sedlar, S., Rodrigues, P., Sosulski, J., Narayanan, D., Bjareholt, E., Quentin, B., Schirrmeister, R. T., Kalunga, E., Darmet, L., Gregoire, C., Abdul Hussain, A., Gatti, R., Goncharenko, V., Thielen, J., Moreau, T., Roy, Y., … Chevallier, S. (2023). Mother of all BCI Benchmarks (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.10034224