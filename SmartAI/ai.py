#DOWNLOAD ResNet50 to your project folder
from imageai.Prediction import ImagePrediction
import os
execution_path=os.getcwd()

prediction = ImagePrediction()
prediction.setModelTypeAsResNet() #Using Microsoft Research to make predictions
prediction.setModelPath(os.path.join(execution_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5")) 
prediction.loadModel()

#Pick files to analyze, I have provided three examples from high to low accuracies
predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "sanfrancisco.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
