import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import io
import base64
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store models and data
models = {}
data = None
X_train = None
X_test = None
y_train = None
y_test = None
X_pca = None
pca = None
scaler = None
feature_names = []

class QSARModel:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'KNN': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42)
        }
        self.trained_models = {}
        self.performance = {}
        
    def load_data(self, file_path):
        """Load and preprocess the dataset"""
        global data, feature_names
        
        df = pd.read_excel(file_path)
        data = df.copy()
        
        # Apply activity labeling rule
        df['Activity'] = (df['IC50 (µM)'] < 10).astype(int)
        df['Activity_Label'] = df['Activity'].map({1: 'ACTIVE', 0: 'INACTIVE'})
        
        return df
    
    def extract_features(self, smiles_list):
        """Extract RDKit features from SMILES"""
        features = []
        valid_smiles = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # 2D Descriptors
                    mol_wt = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)
                    
                    # Morgan Fingerprints
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=128)
                    fp_array = np.array(fp)
                    
                    # Combine all features
                    feature_vector = np.concatenate([[mol_wt, logp, hbd, hba], fp_array])
                    features.append(feature_vector)
                    valid_smiles.append(smiles)
            except:
                continue
                
        return np.array(features), valid_smiles
    
    def preprocess_data(self, df):
        """Preprocess data and apply PCA"""
        global X_train, X_test, y_train, y_test, X_pca, pca, scaler, feature_names
        
        # Extract features
        X, valid_smiles = self.extract_features(df['SMILES Notation'].tolist())
        y = df.loc[df['SMILES Notation'].isin(valid_smiles), 'Activity'].values
        
        # Create feature names
        descriptor_names = ['MolWt', 'MolLogP', 'HBD', 'HBA']
        fingerprint_names = [f'FP_{i}' for i in range(128)]
        feature_names = descriptor_names + fingerprint_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply PCA
        pca = PCA(n_components=20, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        X_pca = pca.transform(scaler.transform(X))
        
        return X_train_pca, X_test_pca, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        self.trained_models = {}
        self.performance = {}
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Calculate training accuracy
            train_pred = model.predict(X_train)
            self.performance[name] = {
                'train_accuracy': accuracy_score(y_train, train_pred)
            }
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models"""
        results = {}
        
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            cm = confusion_matrix(y_test, y_pred)
            
            results[name] = {
                'test_accuracy': accuracy_score(y_test, y_pred),
                'confusion_matrix': cm.tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'predictions': y_pred.tolist(),
                'probabilities': y_proba.tolist() if y_proba is not None else None
            }
            
            self.performance[name].update(results[name])
        
        return results
    
    def predict_compound(self, smiles, model_name):
        """Predict activity for a single compound"""
        if model_name not in self.trained_models:
            return None
            
        try:
            features, _ = self.extract_features([smiles])
            if len(features) == 0:
                return None
                
            features_scaled = scaler.transform(features)
            features_pca = pca.transform(features_scaled)
            
            model = self.trained_models[model_name]
            prediction = model.predict(features_pca)[0]
            probability = model.predict_proba(features_pca)[0] if hasattr(model, 'predict_proba') else None
            
            return {
                'prediction': int(prediction),
                'probability': probability.tolist() if probability is not None else None,
                'activity': 'ACTIVE' if prediction == 1 else 'INACTIVE'
            }
        except:
            return None
    
    def generate_counterfactual(self, smiles, model_name, target_class=1):
        """Generate counterfactual explanation"""
        if model_name not in self.trained_models:
            return None
            
        try:
            features, _ = self.extract_features([smiles])
            if len(features) == 0:
                return None
                
            original_pred = self.predict_compound(smiles, model_name)
            if original_pred is None:
                return None
                
            if original_pred['prediction'] == target_class:
                return {
                    'message': 'Compound already belongs to target class',
                    'original_prediction': original_pred,
                    'counterfactual_found': False
                }
            
            # Simple counterfactual: modify key descriptors
            counterfactuals = []
            feature_modifications = {
                'MolWt': [-50, -30, -20, +20, +30, +50],
                'MolLogP': [-1.0, -0.5, +0.5, +1.0],
                'HBD': [-1, +1],
                'HBA': [-1, +1]
            }
            
            original_features = features[0].copy()
            
            for desc_name, modifications in feature_modifications.items():
                desc_idx = feature_names.index(desc_name)
                
                for mod in modifications:
                    modified_features = original_features.copy()
                    modified_features[desc_idx] += mod
                    
                    # Ensure non-negative values for counts
                    if desc_name in ['HBD', 'HBA'] and modified_features[desc_idx] < 0:
                        modified_features[desc_idx] = 0
                    
                    features_scaled = scaler.transform([modified_features])
                    features_pca = pca.transform(features_scaled)
                    
                    model = self.trained_models[model_name]
                    pred = model.predict(features_pca)[0]
                    proba = model.predict_proba(features_pca)[0] if hasattr(model, 'predict_proba') else None
                    
                    if pred == target_class:
                        counterfactuals.append({
                            'modification': f'{desc_name} {"+" if mod > 0 else ""}{mod:.1f}',
                            'prediction': int(pred),
                            'probability': proba.tolist() if proba is not None else None,
                            'activity': 'ACTIVE' if pred == 1 else 'INACTIVE',
                            'modified_features': modified_features[:4].tolist()  # Only descriptor values
                        })
            
            if counterfactuals:
                # Return the counterfactual with highest probability for target class
                best_cf = max(counterfactuals, key=lambda x: x['probability'][target_class] if x['probability'] else 0)
                return {
                    'original_prediction': original_pred,
                    'counterfactual': best_cf,
                    'counterfactual_found': True
                }
            else:
                return {
                    'message': 'No counterfactual found with simple modifications',
                    'original_prediction': original_pred,
                    'counterfactual_found': False
                }
                
        except Exception as e:
            return {
                'error': str(e),
                'counterfactual_found': False
            }

# Initialize QSAR model
qsar_model = QSARModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_data')
def load_data_route():
    try:
        file_path = 'Dataset/Alzheimers.xlsx'
        df = qsar_model.load_data(file_path)
        
        # Enhanced dataset information
        overview = {
            'total_records': int(len(df)),
            'total_columns': int(len(df.columns)),
            'shape': (int(df.shape[0]), int(df.shape[1])),
            'columns': df.columns.tolist(),
            'column_details': [
                {
                    'name': col,
                    'type': str(df[col].dtype),
                    'non_null_count': int(df[col].count()),
                    'null_count': int(df[col].isnull().sum())
                } for col in df.columns.tolist()
            ],
            'active_count': int(df['Activity'].sum()),
            'inactive_count': int(len(df) - df['Activity'].sum()),
            'ic50_stats': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in df['IC50 (µM)'].describe().to_dict().items()},
            'sample_data': df.head(10).to_dict('records')
        }
        
        return jsonify({'success': True, 'overview': overview})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/preprocess')
def preprocess_data():
    try:
        file_path = 'Dataset/Alzheimers.xlsx'
        df = qsar_model.load_data(file_path)
        X_train_pca, X_test_pca, y_train, y_test = qsar_model.preprocess_data(df)
        
        preprocessing_info = {
            'total_compounds': len(df),
            'valid_compounds': len(X_train_pca) + len(X_test_pca),
            'training_samples': len(X_train_pca),
            'testing_samples': len(X_test_pca),
            'feature_count': X_train_pca.shape[1],
            'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
            'activity_distribution': {
                'active': int(np.sum(y_train) + np.sum(y_test)),
                'inactive': int(len(y_train) + len(y_test) - np.sum(y_train) - np.sum(y_test))
            }
        }
        
        return jsonify({'success': True, 'info': preprocessing_info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/feature_engineering')
def feature_engineering():
    try:
        file_path = 'Dataset/Alzheimers.xlsx'
        df = qsar_model.load_data(file_path)
        X_train_pca, X_test_pca, y_train, y_test = qsar_model.preprocess_data(df)
        
        # Calculate descriptor statistics
        X, valid_smiles = qsar_model.extract_features(df['SMILES Notation'].tolist())
        descriptor_data = X[:, :4]  # First 4 are descriptors
        
        descriptor_stats = {}
        for i, name in enumerate(['MolWt', 'MolLogP', 'HBD', 'HBA']):
            descriptor_stats[name] = {
                'mean': float(np.mean(descriptor_data[:, i])),
                'std': float(np.std(descriptor_data[:, i])),
                'min': float(np.min(descriptor_data[:, i])),
                'max': float(np.max(descriptor_data[:, i]))
            }
        
        engineering_info = {
            'descriptors_used': ['Molecular Weight', 'LogP', 'Hydrogen Bond Donors', 'Hydrogen Bond Acceptors'],
            'fingerprint_config': {
                'type': 'Morgan Fingerprints',
                'radius': 2,
                'bit_size': 128
            },
            'pca_components': 20,
            'descriptor_statistics': descriptor_stats,
            'total_features': X.shape[1],
            'pca_variance_explained': float(np.sum(pca.explained_variance_ratio_))
        }
        
        return jsonify({'success': True, 'info': engineering_info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/train_models')
def train_models():
    try:
        file_path = 'Dataset/Alzheimers.xlsx'
        df = qsar_model.load_data(file_path)
        X_train_pca, X_test_pca, y_train, y_test = qsar_model.preprocess_data(df)
        qsar_model.train_models(X_train_pca, y_train)
        
        training_results = {}
        for name, perf in qsar_model.performance.items():
            training_results[name] = {
                'training_accuracy': perf['train_accuracy']
            }
        
        return jsonify({'success': True, 'results': training_results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/evaluate_models')
def evaluate_models():
    try:
        file_path = 'Dataset/Alzheimers.xlsx'
        df = qsar_model.load_data(file_path)
        X_train_pca, X_test_pca, y_train, y_test = qsar_model.preprocess_data(df)
        qsar_model.train_models(X_train_pca, y_train)
        results = qsar_model.evaluate_models(X_test_pca, y_test)
        
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict_compound():
    try:
        data = request.get_json()
        smiles = data.get('smiles')
        model_name = data.get('model')
        
        if not smiles or not model_name:
            return jsonify({'success': False, 'error': 'SMILES and model name required'})
        
        # Ensure models are trained
        if not qsar_model.trained_models:
            file_path = 'Dataset/Alzheimers.xlsx'
            df = qsar_model.load_data(file_path)
            X_train_pca, X_test_pca, y_train, y_test = qsar_model.preprocess_data(df)
            qsar_model.train_models(X_train_pca, y_train)
        
        result = qsar_model.predict_compound(smiles, model_name)
        
        if result is None:
            return jsonify({'success': False, 'error': 'Invalid SMILES or model'})
        
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/counterfactual', methods=['POST'])
def generate_counterfactual():
    try:
        data = request.get_json()
        smiles = data.get('smiles')
        model_name = data.get('model')
        target_class = data.get('target_class', 1)
        
        if not smiles or not model_name:
            return jsonify({'success': False, 'error': 'SMILES and model name required'})
        
        # Ensure models are trained
        if not qsar_model.trained_models:
            file_path = 'Dataset/Alzheimers.xlsx'
            df = qsar_model.load_data(file_path)
            X_train_pca, X_test_pca, y_train, y_test = qsar_model.preprocess_data(df)
            qsar_model.train_models(X_train_pca, y_train)
        
        result = qsar_model.generate_counterfactual(smiles, model_name, target_class)
        
        if result is None:
            return jsonify({'success': False, 'error': 'Invalid SMILES or model'})
        
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/plot_confusion_matrix/<model_name>')
def plot_confusion_matrix(model_name):
    try:
        if not qsar_model.performance.get(model_name):
            return jsonify({'success': False, 'error': 'Model not evaluated'})
        
        cm = np.array(qsar_model.performance[model_name]['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['INACTIVE', 'ACTIVE'],
                   yticklabels=['INACTIVE', 'ACTIVE'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return jsonify({'success': True, 'plot_url': plot_url})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/compare_models')
def compare_models():
    try:
        if not qsar_model.performance:
            return jsonify({'success': False, 'error': 'Models not evaluated'})
        
        comparison = []
        for name, perf in qsar_model.performance.items():
            if 'test_accuracy' in perf:
                comparison.append({
                    'model': name,
                    'accuracy': perf['test_accuracy'],
                    'train_accuracy': perf.get('train_accuracy', 0)
                })
        
        comparison.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return jsonify({'success': True, 'comparison': comparison})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
   app.run(host='0.0.0.0', debug=True)
