import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc_scaled = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None 
    return mfcc_scaled

# Path to dataset gtzan
dataset_path = "C:/data/archive/Data/genres_original" 
audio_files = []
labels = []



genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

for genre in genres:
    genre_folder = os.path.join(dataset_path, genre)
    for filename in os.listdir(genre_folder):
        file_path = os.path.join(genre_folder, filename)
        features = extract_features(file_path)
        if features is not None:
            audio_files.append(features)
            labels.append(genre)
            
# Convert to numpy arrays
X = np.array(audio_files)
y = np.array(labels)


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)


from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

# Scale features, train Support Vector Machine
svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale'))
svm_model.fit(X_train, y_train)



from sklearn.metrics import classification_report, accuracy_score

y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

            
            

