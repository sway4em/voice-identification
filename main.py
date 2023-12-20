import os
from pyannote.audio import Model
model = Model.from_pretrained("pyannote/embedding", 
                              os.getenv("PYANNOTE_AUTH_TOKEN"))

from pyannote.audio import Inference
inference = Inference(model, window="whole")
embedding1 = inference("trump3.wav")
embedding2 = inference("trump4.wav")
# `embeddingX` is (1 x D) numpy array extracted from the file as a whole.

from scipy.spatial.distance import cdist
distance = cdist(embedding1, embedding2, metric="cosine")[0,0]
print(distance)
# `distance` is a `float` describing how dissimilar speakers 1 and 2 are.