import os.path

import numpy as np
import pandas as pd
from jedi.api import file_name
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import joblib
from VariableAE import FeatureAttentionLayer
from tensorflow.keras import config
from VariableAE import FeatureAttentionLayer, SelfAttentionLayer
# config.enable_unsafe_deserialization()

# -------------------------------
# 1. Load the trained Siamese model
# -------------------------------
def load_siamese_model(model_path):
    return load_model(model_path, compile=False)


# -------------------------------
# 2. Prepare test input vectors
# -------------------------------
def prepare_test_inputs(vec_a, vec_b, scaler):
    input_a = np.array([vec_a], dtype=np.float32)
    input_b = np.array([vec_b], dtype=np.float32)
    input_a = scaler.transform(input_a)
    input_b = scaler.transform(input_b)
    return input_a, input_b


# -------------------------------
# 3. Predict similarity score and label
# -------------------------------
def predict_similarity(model, input_a, input_b, threshold=0.5):
    prediction = model.predict([input_a, input_b])
    score = prediction[0][0]
    #label = int(score > threshold)
    return score #, label


# -------------------------------
# 4. Compute AUC
# -------------------------------
# Compute AUC
def Compute_AUC(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


# -------------------------------
# 5. VAE Similarity Score
# -------------------------------
def getVAEAttnSimilarityScore(model_dir, model_name, scalar_model, file_dir, csvFileName):
    # Model path

    model_path = os.path.join(model_dir, model_name)
    model = load_siamese_model(model_path)
    scaler = joblib.load(os.path.join(model_dir, scalar_model))
    fileName = os.path.join(file_dir, csvFileName)
    print(f'Working with file: {fileName}')

    dF = pd.read_csv(fileName)
    scores = []
    for _, row in dF.iterrows():
        '''
        vec_a = [row['src_feature1'], row['src_feature2'], row['src_feature3'], row['src_feature4'], row['src_feature5'],
                 row['src_feature6'], row['src_feature7'], row['src_feature8'], row['src_feature9']]
        vec_b = [row['des_feature1'], row['des_feature2'], row['des_feature3'], row['des_feature4'], row['des_feature5'],
                 row['des_feature6'], row['des_feature7'], row['des_feature8'], row['des_feature9']]
        '''
        vec_a = np.array([float(row[f"src_feature{i}"]) for i in range(1, 10)], dtype=np.float32)
        vec_b = np.array([float(row[f"des_feature{i}"]) for i in range(1, 10)], dtype=np.float32)
        input_a, input_b = prepare_test_inputs(vec_a, vec_b, scaler)
        # Predict score, label = predict_similarity(model, input_a, input_b)

        score = predict_similarity(model, input_a, input_b)
        scores.append(score)
    dF['attn_vae_similarity'] = scores
    dF.to_csv(fileName, index=False)
    # print(f"Similarity Score: {score:.4f}")
    # print(f"Predicted Label (0 = dissimilar, 1 = similar): {label}")


# -------------------------------
# 5. Code File Selection Block
# -------------------------------

def getAttnVAESim_CLCDSA(model_dir, file_dir, csvFileNames):
    for csvFileName in csvFileNames:
        print(f'working with {csvFileName} model and {csvFileName} for getting vae similarity')
        trueCloneFeatures = csvFileName + 'Features.csv'
        falseCloneFeatures = csvFileName + 'NonCloneFeatures.csv'

        model_name = csvFileName + 'VAEModelLarge.keras'
        scaler_model = csvFileName + 'ScalarLarge.pkl'
        getVAEAttnSimilarityScore(model_dir, model_name, scaler_model, file_dir, trueCloneFeatures)
        getVAEAttnSimilarityScore(model_dir, model_name, scaler_model, file_dir, falseCloneFeatures)

def getAttnVAESim_GPTCloneBench(model_dir, file_dir, csvFileNames, gptFileNames):
    for csvFileName, gptFileName in zip(csvFileNames, gptFileNames):
        print(f'working with {csvFileName} model and {gptFileName} for getting vae similarity')
        trueCloneFeatures = gptFileName + '.csv'
        falseCloneFeatures = gptFileName +'NonClone.csv'

        model_name = csvFileName + 'VAEModelLarge.keras'
        scaler_model = csvFileName + 'ScalarLarge.pkl'
        getVAEAttnSimilarityScore(model_dir, model_name, scaler_model, file_dir, trueCloneFeatures)
        getVAEAttnSimilarityScore(model_dir, model_name, scaler_model, file_dir, falseCloneFeatures)


def getAttnVAESim_XLCost(model_dir, file_dir, csvFileNames, xlcostFileNames):
    for csvFileName, xlcostFileName in zip(csvFileNames, xlcostFileNames):
        print(f'working with {csvFileName} model and {xlcostFileName} for getting vae similarity')
        trueCloneFeatures = xlcostFileName + 'code_pairs.csv'
        falseCloneFeatures = xlcostFileName +'nonclone_code_pairs.csv'

        model_name = csvFileName + 'VAEModelLarge.keras'
        scaler_model = csvFileName + 'ScalarLarge.pkl'
        getVAEAttnSimilarityScore(model_dir, model_name, scaler_model, file_dir, trueCloneFeatures)
        getVAEAttnSimilarityScore(model_dir, model_name, scaler_model, file_dir, falseCloneFeatures)
# -------------------------------
# 6. Main execution block
# -------------------------------
def main():
    file_dir = './../Data/CSVFiles'
    model_dir = './SaveModels'
    csvFileNames = ['JavaCSharp', 'JavaPython', 'CSharpPython']
    gptFileNames = ['javacs_gptclonebench', 'javapython_gptclonebench', 'cspython_gptclonebench']
    xlcostFileNames = ["Java_C#_", "Java_Python_", "C#_Python_"]
    #gptFileNames = ['javacs_gptclonebench']
    # getAttnVAESim_CLCDSA(model_dir, file_dir, csvFileNames)
    getAttnVAESim_GPTCloneBench(model_dir, file_dir, csvFileNames, gptFileNames)
    getAttnVAESim_XLCost(model_dir, file_dir, csvFileNames, xlcostFileNames)


if __name__ == "__main__":
    main()
