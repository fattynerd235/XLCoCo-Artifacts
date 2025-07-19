import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Multiply, Softmax, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from pathlib import Path

# ----------------------------------
# 1. Load and preprocess training data
# ----------------------------------
def load_and_preprocess_data(csv_file, label=True, save_scaler_path=None):
    df = pd.read_csv(csv_file)
    feature_cols = ['gpt_semantic_similarity', 'gpt_intent_similarity', 'attn_vae_similarity']
    df['label'] = 1 if label else 0

    features = df[feature_cols].values.astype(np.float32)
    labels = df['label'].values.astype(np.float32)

    '''
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    if save_scaler_path:
        joblib.dump(scaler, save_scaler_path)
    '''
    return features, labels

# ----------------------------------
# 2. Load and preprocess test data
# ----------------------------------
def load_and_preprocess_test_data(test_csv_file, scaler_path):
    df = pd.read_csv(test_csv_file)
    feature_cols = ['gpt_semantic_similarity', 'gpt_intent_similarity', 'attn_vae_similarity']
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in test CSV: {col}")
    features = df[feature_cols].values.astype(np.float32)
    scaler = joblib.load(scaler_path)
    return scaler.transform(features)

# ----------------------------------
# 3. Build attention-based fusion model
# ----------------------------------
def build_attention_fusion_model(input_dim):
    inputs = Input(shape=(input_dim,), name="input_layer")
    attention_scores = Dense(input_dim, activation='softmax', name='attention_scores')(inputs)
    weighted_inputs = Multiply(name='attention_mul')([inputs, attention_scores])

    x = BatchNormalization()(weighted_inputs)  # optional normalization
    #x = Dense(16, activation='relu')(weighted_inputs)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(8, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    return model

# ----------------------------------
# 4. Train the attention-based fusion model
# ----------------------------------
def train_attention_fusion(X_train, y_train, model_path, model_name, class_weights=None):
    model = build_attention_fusion_model(X_train.shape[1])

    checkpoint = ModelCheckpoint(
        os.path.join(model_path, model_name), monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights
    )
    save_path = os.path.join(model_path, model_name + "_history.csv")
    plot_training_history(history, save_path)
    return model

# ----------------------------------
# 5. Save predictions
# ----------------------------------
def save_predictions_to_csv(test_csv_file, probs):
    df = pd.read_csv(test_csv_file)
    df['final_similarity'] = probs.ravel()
    df.to_csv(test_csv_file, index=False)

# ----------------------------------
# 6. Plot training
# ----------------------------------
def plot_training_history(history, save_path=None):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.ylim(0, 1)
    plt.legend(), plt.title('Loss'), plt.xlabel('Epoch'), plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.ylim(0, 1)
    plt.legend(), plt.title('Accuracy'), plt.xlabel('Epoch'), plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()

    if save_path:
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(save_path, index=False)

# ----------------------------------
# 7. Evaluation
# ----------------------------------


# ----------------------------------
# Model Analysis Evaluation
# ----------------------------------

def evaluate_classification_performance(y_true, y_pred_probs, threshold=0.5, output_path = None):
    y_pred = (y_pred_probs >= threshold).astype(int)
    report = classification_report(y_true, y_pred, target_names=["Non-Clone", "Clone"])
    print("Classification Report:\n", report)

    if output_path:
        with open(output_path, "w") as f:
            f.write("Classification Report\n")
            f.write("======================\n")
            f.write(report)

def evaluate_on_true_set(test_csv_file, model_path, model_name, scaler_path):
    df = pd.read_csv(test_csv_file)

    # Dynamically decide label
    label = 1 if "NonClone" not in test_csv_file else 0
    y_true = np.full(len(df), label)  # Create a full array of label

    test_data = load_and_preprocess_test_data(test_csv_file, scaler_path)
    model = load_model(os.path.join(model_path, model_name))
    y_pred_probs = model.predict(test_data)

    name_without_extension = Path(model_name).stem
    datasetName = '_gptclone_' if 'gptclone' in test_csv_file else '_atcoder_'
    file_suffix = "_True" if label == 1 else "_False"
    output_path = os.path.join(model_path, name_without_extension + datasetName + file_suffix + "_Classification.txt")

    evaluate_classification_performance(y_true, y_pred_probs, output_path=output_path)

# ----------------------------------
# 8. Train and evaluate
# ----------------------------------
def train_model(true_csv_file, false_csv_file, model_dir, model_name, scaler_path,
                use_smote=True, use_class_weight=False):

    X_true, y_true = load_and_preprocess_data(true_csv_file, label=True)
    X_false, y_false = load_and_preprocess_data(false_csv_file, label=False)
    X_train = np.concatenate([X_true, X_false], axis=0)
    y_train = np.concatenate([y_true, y_false], axis=0)

    # Now fit and save the scaler on full X_train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    joblib.dump(scaler, scaler_path)

    if use_smote:
        print("Applying SMOTE...")
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    indices = np.arange(len(y_train))
    np.random.shuffle(indices)
    X_train, y_train = X_train[indices], y_train[indices]

    class_weights = None
    if use_class_weight and not use_smote:
        class_labels = np.unique(y_train)
        class_weights_arr = compute_class_weight(class_weight='balanced', classes=class_labels, y=y_train)
        class_weights = dict(zip(class_labels, class_weights_arr))

    return train_attention_fusion(X_train, y_train, model_dir, model_name, class_weights)

# ----------------------------------
# 9. Inference
# ----------------------------------
def gen_prediction(test_csv_file, model_path, model_name, scaler_path):
    test_data = load_and_preprocess_test_data(test_csv_file, scaler_path)
    model = load_model(os.path.join(model_path, model_name))
    predictions = model.predict(test_data)
    print("Probabilities:", predictions.ravel())
    save_predictions_to_csv(test_csv_file, predictions)


# ----------------------------------
# 10. Main runner
# ----------------------------------
def main_train(model_dir, dataset_dir, src_trg_names, class_balancing):
    for src_trg_name in src_trg_names:
        model_name = src_trg_name + class_balancing +"Fusion_Attn.keras"
        true_csv_file = os.path.join(dataset_dir, src_trg_name + "Features.csv")
        false_csv_file = os.path.join(dataset_dir, src_trg_name + "NonCloneFeatures.csv")
        scaler_path = os.path.join(model_dir, src_trg_name + class_balancing + "_attn_scaler.pkl")
        use_smote  = True
        use_class_weight = False

        if '_weight_' in class_balancing:
            use_smote = False
            use_class_weight = True

        print("Training the attention-based fusion model...")
        model = train_model(true_csv_file, false_csv_file, model_dir, model_name, scaler_path,
                    use_smote=use_smote, use_class_weight=use_class_weight)

        # Display model summary
        model.summary()
        # write it in file
        name_without_extension = Path(model_name).stem
        with open(os.path.join(model_dir, name_without_extension + class_balancing + "_summary.txt"), "w") as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        print("Generating predictions...")
        gen_prediction(true_csv_file, model_dir, model_name, scaler_path)

        print("Evaluating classification report...")
        evaluate_on_true_set(true_csv_file, model_dir, model_name, scaler_path)

def getFusionSimilarity(model_dir, dataset_dir, data_file_map, class_balancing):
    for src_trg_name, file_list in data_file_map.items():
        model_name = src_trg_name + class_balancing + "Fusion_Attn.keras"
        scaler_path = os.path.join(model_dir, src_trg_name + class_balancing + "_attn_scaler.pkl")

        for dataFile in file_list:
            csv_file = os.path.join(dataset_dir, dataFile)
            print("Generating predictions...")
            gen_prediction(csv_file, model_dir, model_name, scaler_path)
            print("Evaluating classification report...")
            evaluate_on_true_set(csv_file, model_dir, model_name, scaler_path)

        print("=" * 100)
        print("=" * 100)


if __name__ == "__main__":
    os.makedirs("SavedModels", exist_ok=True)

    model_dir = "./SavedModels"
    dataset_dir = "./../Data/CSVFiles"
    # Define mapping from src_trg_name to its 4 related files
    '''
    data_file_map = {
        "JavaCSharp": [
            "JavaCSharpFeatures.csv",
            "JavaCSharpNonCloneFeatures.csv",
            "javacs_gptclonebench.csv",
            "javacs_gptclonebenchNonClone.csv",
            "Java_C#_code_pairs.csv",
            "Java_C#_nonclone_code_pairs.csv"
        ],
        "JavaPython": [
            "JavaPythonFeatures.csv",
            "JavaPythonNonCloneFeatures.csv",
            "javapython_gptclonebench.csv",
            "javapython_gptclonebenchNonClone.csv",
            "Java_Python_code_pairs.csv",
            "Java_Python_nonclone_code_pairs.csv"
        ],
        "CSharpPython": [
            "CSharpPythonFeatures.csv",
            "CSharpPythonNonCloneFeatures.csv",
            "cspython_gptclonebench.csv",
            "cspython_gptclonebenchNonClone.csv",
            "C#_Python_code_pairs.csv",
            "C#_Python_nonclone_code_pairs.csv"
        ]
    }
    '''

    data_file_map = {
        "JavaCSharp": [
            "javacs_gptclonebench.csv",
            "javacs_gptclonebenchNonClone.csv",
            "Java_C#_code_pairs.csv",
            "Java_C#_nonclone_code_pairs.csv"
        ],
        "JavaPython": [
            "javapython_gptclonebench.csv",
            "javapython_gptclonebenchNonClone.csv",
            "Java_Python_code_pairs.csv",
            "Java_Python_nonclone_code_pairs.csv"
        ],
        "CSharpPython": [
            "cspython_gptclonebench.csv",
            "cspython_gptclonebenchNonClone.csv",
            "C#_Python_code_pairs.csv",
            "C#_Python_nonclone_code_pairs.csv"
        ]
    }

    src_trg_names = ["JavaCSharp", "JavaPython", "CSharpPython"]
    # _weight_ or _smote_
    class_balancing = "_weight_"
    #main_train(model_dir, dataset_dir, src_trg_names, class_balancing)
    #evaluation mapping
    getFusionSimilarity(model_dir, dataset_dir, data_file_map, class_balancing)



