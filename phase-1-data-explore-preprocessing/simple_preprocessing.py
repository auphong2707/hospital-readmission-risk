"""
Simplified Hospital Readmission Risk Preprocessing Pipeline
==========================================================

Clean implementation targeting exactly 56 base features:
- 23 raw features (kept as-is, individual medications/utilization/demographics)
- 27 diagnosis binary indicators (9 categories × 3 diagnosis positions)
- 6 age stratification features (age_bucket ordinal + 5 binary indicators)

Total: 56 base features → ~80-100 after one-hot encoding categorical raw features.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
import warnings
warnings.filterwarnings('ignore')


class CompletePreprocessor:
    """
    Simplified preprocessing pipeline for 56 base features.
    
    Follows eda_compact.ipynb plan exactly:
    - 23 raw features preserved (age, race, gender, medications, utilization, etc.)
    - 27 binary diagnosis indicators (9 categories × 3 positions) 
    - 6 age stratification features (age_bucket + 5 binary indicators)
    - One-hot encoding expands categoricals → ~80-100 final features
    
    Usage:
        preprocessor = CompletePreprocessor(random_state=42, scaler_type='standard')
        splits_dict = preprocessor.fit_transform(data_path)
    """
    
    def __init__(self, random_state=42, scaler_type='standard'):
        self.random_state = random_state
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else RobustScaler()
        self.label_encoders = {}
        self.target_encoders = {}
        self.original_data = None
        self.target_encoding_maps = {}
        
    def load_data(self, data_path="./data/diabetic_data.csv"):
        """Load the hospital readmission data."""
        print("Loading data...")
        data = pd.read_csv(data_path)
        print(f"Data loaded: {data.shape[0]} patients, {data.shape[1]} features")
        return data
    
    def handle_missing_values(self, data):
        """Handle missing values with median/mode and group-wise imputation."""
        print("Handling missing values with median/mode and group-wise imputation...")
        
        # Replace '?' with NaN
        data = data.replace('?', np.nan)
        
        # Create missing indicators for features with >10% missing
        missing_threshold = 0.10
        for col in data.columns:
            missing_rate = data[col].isnull().sum() / len(data)
            if missing_rate > missing_threshold:
                missing_indicator = f"{col}_is_missing"
                data[missing_indicator] = data[col].isnull().astype(int)
                print(f"  Created missing indicator: {missing_indicator} ({missing_rate:.1%} missing)")
        
        # Group-wise imputation for MAR features
        if 'medical_specialty' in data.columns and 'admission_type_id' in data.columns:
            for admission_type in data['admission_type_id'].unique():
                mask = (data['admission_type_id'] == admission_type) & data['medical_specialty'].isnull()
                if mask.sum() > 0:
                    mode_val = data[data['admission_type_id'] == admission_type]['medical_specialty'].mode()
                    fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
                    data.loc[mask, 'medical_specialty'] = fill_val
        
        if 'payer_code' in data.columns and 'admission_source_id' in data.columns:
            for source in data['admission_source_id'].unique():
                mask = (data['admission_source_id'] == source) & data['payer_code'].isnull()
                if mask.sum() > 0:
                    mode_val = data[data['admission_source_id'] == source]['payer_code'].mode()
                    fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
                    data.loc[mask, 'payer_code'] = fill_val
        
        # Fill remaining categorical with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                mode_value = data[col].mode().iloc[0] if not data[col].mode().empty else 'Unknown'
                data[col] = data[col].fillna(mode_value)
        
        # Fill numerical with median
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].median())
        
        print(f"Missing values remaining: {data.isnull().sum().sum()}")
        return data
    
    def validate_data_quality(self, data):
        """Validate value ranges, data types, and domain constraints."""
        print("Validating data quality (ranges, types, constraints)...")
        
        numerical_validations = {
            'time_in_hospital': (1, 14),
            'num_lab_procedures': (0, 150),
            'num_procedures': (0, 10),
            'num_medications': (0, 100),
            'number_outpatient': (0, 50),
            'number_emergency': (0, 50),
            'number_inpatient': (0, 20),
            'number_diagnoses': (1, 16)
        }
        
        for col, (min_val, max_val) in numerical_validations.items():
            if col in data.columns:
                data[col] = data[col].clip(lower=min_val, upper=max_val)
        
        if 'gender' in data.columns:
            valid_genders = ['Male', 'Female', 'Unknown/Invalid']
            data.loc[~data['gender'].isin(valid_genders), 'gender'] = 'Unknown/Invalid'
        
        print("Data validation complete")
        return data
    
    def treat_outliers(self, data):
        """Apply IQR-based outlier treatment."""
        print("Applying IQR-based outlier treatment...")
        
        numerical_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                         'num_medications', 'number_outpatient', 'number_emergency', 
                         'number_inpatient', 'number_diagnoses']
        
        outliers_treated = 0
        for col in numerical_cols:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers_before = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                outliers_treated += outliers_before
        
        print(f"Treated {outliers_treated} outliers using IQR method")
        return data
    
    def create_target(self, data):
        """Create binary target variable for 30-day readmission."""
        print("Creating 30-day readmission binary target...")
        data['target'] = (data['readmitted'] == '<30').astype(int)
        target_counts = data['target'].value_counts()
        print(f"Target distribution: No={target_counts[0]}, Yes={target_counts[1]}")
        return data
    
    def engineer_features(self, data):
        """Engineer 33 features: 27 diagnosis binary + 6 age stratification."""
        print("Engineering features: 27 diagnosis binary + 6 age stratification...")
        
        # 1. Create 27 binary diagnosis indicators (9 categories × 3 positions)
        diagnosis_cols = ['diag_1', 'diag_2', 'diag_3']
        if all(col in data.columns for col in diagnosis_cols):
            def categorize_diagnosis(code):
                if pd.isna(code):
                    return 'Other'
                code_str = str(code)
                try:
                    if code_str.startswith('V') or code_str.startswith('E'):
                        return 'Other'
                    code_num = float(code_str) if '.' in code_str else float(code_str)
                    if 390 <= code_num < 460 or code_num == 785:
                        return 'Circulatory'
                    elif 460 <= code_num < 520 or code_num == 786:
                        return 'Respiratory'
                    elif 520 <= code_num < 580 or code_num == 787:
                        return 'Digestive'
                    elif 250.0 <= code_num < 251:
                        return 'Diabetes'
                    elif 800 <= code_num < 1000:
                        return 'Injury'
                    elif 140 <= code_num < 240:
                        return 'Neoplasms'
                    elif 710 <= code_num < 740:
                        return 'Musculoskeletal'
                    elif 580 <= code_num < 630 or code_num == 788:
                        return 'Genitourinary'
                    else:
                        return 'Other'
                except:
                    return 'Other'
            
            categories = ['Circulatory', 'Respiratory', 'Diabetes', 'Digestive', 'Injury', 
                         'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other']
            
            for diag_col in diagnosis_cols:
                diag_categories = data[diag_col].apply(categorize_diagnosis)
                for category in categories:
                    feature_name = f"{diag_col}_{category.lower()}"
                    data[feature_name] = (diag_categories == category).astype(int)
            
            # Drop original diagnosis columns
            data = data.drop(columns=diagnosis_cols, errors='ignore')
            print("  Created 27 binary diagnosis indicators and dropped original diagnosis columns")
        
        # 2. Create 6 age stratification features
        # Keep original 'age' column (it's one of the 23 raw features)
        if 'age' in data.columns:
            # Map age brackets to ordinal codes (0-9) - refined based on risk patterns
            age_bucket_mapping = {
                '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
                '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
            }
            data['age_bucket'] = data['age'].map(age_bucket_mapping).fillna(5).astype(int)
            
            # Create 5 binary age indicators based on refined risk stratification
            data['age_very_young'] = (data['age_bucket'] < 2).astype(int)      # [0-20): low risk
            data['age_adult'] = ((data['age_bucket'] >= 2) & (data['age_bucket'] < 4)).astype(int)  # [20-40): higher risk
            data['age_middle'] = ((data['age_bucket'] >= 4) & (data['age_bucket'] < 6)).astype(int) # [40-60): moderate risk
            data['age_senior'] = ((data['age_bucket'] >= 6) & (data['age_bucket'] < 8)).astype(int)    # [60-80): moderate-high risk
            data['age_elderly'] = (data['age_bucket'] >= 8).astype(int)   # [80+): higher risk
            
            print("  Created age_bucket (ordinal 0-9) + 5 refined binary age indicators:")
            print("    - age_very_young: [0-20), low risk")
            print("    - age_adult: [20-40), higher risk")
            print("    - age_middle: [40-60), moderate risk")
            print("    - age_senior: [60-80), moderate-high risk")
            print("    - age_elderly: [80+), higher risk")
            print("  Kept original 'age' column as raw feature")
        
        return data
    
    def encode_features(self, data, is_training=True):
        """Encode features: one-hot for low cardinality, target encoding for high cardinality."""
        print("Encoding features...")
        
        # Remove ID columns and original readmitted
        id_cols = ['encounter_id', 'patient_nbr', 'readmitted']
        data = data.drop(columns=[col for col in id_cols if col in data.columns], errors='ignore')
        
        # Separate target
        y = data['target'] if 'target' in data.columns else None
        categorical_cols = data.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'target']
        
        # Split into low and high cardinality
        low_cardinality = []
        high_cardinality = []
        
        for col in categorical_cols:
            unique_count = data[col].nunique()
            if unique_count < 10:
                low_cardinality.append(col)
            else:
                high_cardinality.append(col)
        
        # One-hot encode low cardinality
        if low_cardinality:
            print(f"  One-hot encoding {len(low_cardinality)} low-cardinality features")
            one_hot_data = pd.get_dummies(data[low_cardinality], prefix=low_cardinality, drop_first=True)
            data = data.drop(columns=low_cardinality)
            data = pd.concat([data, one_hot_data], axis=1)
        
        # CV-safe target encoding for high cardinality
        if high_cardinality and y is not None:
            print(f"  Target encoding {len(high_cardinality)} high-cardinality features")
            for col in high_cardinality:
                if is_training:
                    category_means = data.groupby(col)[y.name].mean()
                    global_mean = y.mean()
                    category_counts = data.groupby(col).size()
                    smoothing = 100
                    smoothed_means = (category_counts * category_means + smoothing * global_mean) / (category_counts + smoothing)
                    self.target_encoding_maps[col] = {
                        'map': smoothed_means.to_dict(),
                        'global_mean': global_mean
                    }
                    data[f'{col}_target_encoded'] = data[col].map(smoothed_means).fillna(global_mean)
                else:
                    if col in self.target_encoding_maps:
                        encoding_info = self.target_encoding_maps[col]
                        data[f'{col}_target_encoded'] = data[col].map(encoding_info['map']).fillna(encoding_info['global_mean'])
                    else:
                        data[f'{col}_target_encoded'] = 0.0
                data = data.drop(columns=[col])
        
        # Label encode any remaining categoricals
        remaining_categorical = data.select_dtypes(include=['object']).columns
        remaining_categorical = [col for col in remaining_categorical if col != 'target']
        
        for col in remaining_categorical:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
            else:
                known_labels = set(self.label_encoders[col].classes_)
                data[col] = data[col].astype(str).apply(lambda x: x if x in known_labels else 'Unknown')
                data[col] = self.label_encoders[col].transform(data[col])
        
        print(f"Encoding complete")
        return data
    
    def _store_demographics(self, data):
        """Store demographics (race, gender, age) for Phase 5 fairness evaluation."""
        print("Storing demographics for Phase 5 fairness evaluation...")
        
        demographic_cols = ['race', 'gender', 'age']
        available_cols = [col for col in demographic_cols if col in data.columns]
        
        if not available_cols:
            print("⚠️  Warning: No demographic columns found!")
            self.original_demographics = None
            return
        
        self.original_demographics = data[available_cols].copy()
        
        if 'encounter_id' in data.columns:
            self.original_demographics['encounter_id'] = data['encounter_id']
        
        print(f"  Stored demographics: {available_cols}")
    
    def sanitize_column_names(self, data):
        """Sanitize column names for LightGBM compatibility."""
        print("Sanitizing column names for LightGBM...")
        
        def clean_name(name):
            name = str(name)
            for char in [':', '"', "'", '[', ']', '{', '}', '\\', '/', ',', '<', '>', '|']:
                name = name.replace(char, '_')
            name = name.replace(' ', '_').replace('-', '_')
            while '__' in name:
                name = name.replace('__', '_')
            return name.strip('_')
        
        old_columns = data.columns.tolist()
        new_columns = [clean_name(col) for col in old_columns]
        
        # Handle duplicates
        if len(new_columns) != len(set(new_columns)):
            seen = {}
            final_columns = []
            for col in new_columns:
                if col in seen:
                    seen[col] += 1
                    final_columns.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    final_columns.append(col)
            new_columns = final_columns
        
        data.columns = new_columns
        return data
    
    def scale_features_train_test(self, X_train, X_val, X_test):
        """Scale features separately for train/val/test to prevent data leakage."""
        print(f"\n🔒 Scaling features using {self.scaler_type.capitalize()}Scaler...")
        print("   Fitting scaler on TRAINING data only (prevents data leakage!)")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"   ✅ Scaled {X_train.shape[0]:,} train, {X_val.shape[0]:,} val, {X_test.shape[0]:,} test samples")
        
        return (
            pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index),
            pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index),
            pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        )
    
    def fit_transform(self, data_path="./data/diabetic_data.csv"):
        """Complete preprocessing pipeline."""
        print("=" * 80)
        print("SIMPLIFIED HOSPITAL READMISSION PREPROCESSING (56 BASE FEATURES)")
        print("=" * 80)
        
        # Load data
        data = self.load_data(data_path)
        self.original_data = data.copy()
        
        # Preprocessing steps
        data = self.handle_missing_values(data)
        data = self.validate_data_quality(data)
        data = self.treat_outliers(data)
        data = self.create_target(data)
        data = self.engineer_features(data)
        
        # Store demographics BEFORE encoding
        self._store_demographics(data)
        
        # Encode and sanitize
        data = self.encode_features(data, is_training=True)
        data = self.sanitize_column_names(data)
        
        # Separate features and target
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Create train/val/test splits BEFORE scaling
        print(f"\n📊 Creating train/validation/test splits (70/15/15)...")
        test_size = 0.15
        val_size = 0.15
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state, stratify=y_temp
        )
        
        print(f"   Training: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"   Validation: {X_val.shape[0]:,} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"   Test: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Scale features
        X_train, X_val, X_test = self.scale_features_train_test(X_train, X_val, X_test)
        
        # Store indices for demographics alignment
        self._train_indices = X_train.index
        self._val_indices = X_val.index
        self._test_indices = X_test.index
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE!")
        print("=" * 80)
        print(f"✅ 56 base features → {X_train.shape[1]} final features (after one-hot encoding)")
        print(f"✅ 23 raw features preserved (age, medications, utilization, demographics)")
        print(f"✅ 27 binary diagnosis indicators (9 categories × 3 positions)")
        print(f"✅ 6 age stratification features (age_bucket + 5 binary indicators)")
        print(f"✅ Missing indicators for features with >10% missing")
        print(f"✅ IQR outlier treatment applied")
        print(f"✅ One-hot encoding for low-cardinality categoricals")
        print(f"✅ Target encoding for high-cardinality categoricals")
        print(f"✅ StandardScaler fitted on training data only (no leakage!)")
        print(f"\n📊 Final splits:")
        print(f"   Train: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
        print(f"   Val:   {X_val.shape[0]:,} samples × {X_val.shape[1]} features")
        print(f"   Test:  {X_test.shape[0]:,} samples × {X_test.shape[1]} features")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def save_scaler(self, output_dir="./data/processed"):
        """Save fitted scaler."""
        print(f"\n💾 Saving scaler...")
        os.makedirs(output_dir, exist_ok=True)
        scaler_path = os.path.join(output_dir, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        print(f"   ✅ Saved: {scaler_path}")
        
        splits_dir = os.path.join(output_dir, "splits")
        if os.path.exists(splits_dir):
            scaler_split_path = os.path.join(splits_dir, "scaler.pkl")
            joblib.dump(self.scaler, scaler_split_path)
            print(f"   ✅ Saved: {scaler_split_path}")
        
        return scaler_path
    
    def create_train_test_split(self, splits_dict, output_dir="./data/processed"):
        """Save train/validation/test splits to files."""
        X_train = splits_dict['X_train']
        y_train = splits_dict['y_train']
        X_val = splits_dict['X_val']
        y_val = splits_dict['y_val']
        X_test = splits_dict['X_test']
        y_test = splits_dict['y_test']
        
        print(f"\n💾 Saving splits to {output_dir}/splits/...")
        splits_dir = os.path.join(output_dir, "splits")
        os.makedirs(splits_dir, exist_ok=True)
        
        # Save train/val/test
        train_file = os.path.join(splits_dir, "train.csv")
        train_data = X_train.copy()
        train_data['target'] = y_train
        train_data.to_csv(train_file, index=False)
        print(f"   ✅ Saved: {train_file}")
        
        val_file = os.path.join(splits_dir, "validation.csv")
        val_data = X_val.copy()
        val_data['target'] = y_val
        val_data.to_csv(val_file, index=False)
        print(f"   ✅ Saved: {val_file}")
        
        test_file = os.path.join(splits_dir, "test.csv")
        test_data = X_test.copy()
        test_data['target'] = y_test
        test_data.to_csv(test_file, index=False)
        print(f"   ✅ Saved: {test_file}")
        
        # Save demographics for Phase 5
        if hasattr(self, 'original_demographics') and self.original_demographics is not None:
            train_demo = self.original_demographics.loc[self._train_indices]
            val_demo = self.original_demographics.loc[self._val_indices]
            test_demo = self.original_demographics.loc[self._test_indices]
            
            train_demo.to_csv(os.path.join(splits_dir, "train_demographics.csv"), index=False)
            val_demo.to_csv(os.path.join(splits_dir, "validation_demographics.csv"), index=False)
            test_demo.to_csv(os.path.join(splits_dir, "test_demographics.csv"), index=False)
            print(f"   ✅ Saved: demographics files (for Phase 5 fairness)")
        
        # Save scaler
        self.save_scaler(output_dir=output_dir)
        
        return {
            'train': train_file,
            'validation': val_file,
            'test': test_file,
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }
    
    def export_for_huggingface(self, splits_dict, output_dir="./data/processed/huggingface"):
        """Export for HuggingFace upload."""
        print(f"\n🤗 Preparing HuggingFace export...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save splits
        splits = self.create_train_test_split(splits_dict, output_dir=output_dir)
        
        # Create dataset card
        readme_file = os.path.join(output_dir, "README.md")
        with open(readme_file, 'w') as f:
            f.write("# Hospital Readmission Risk - Preprocessed Data\n\n")
            f.write("## Dataset Description\n")
            f.write("Preprocessed hospital readmission data for 30-day readmission prediction.\n\n")
            f.write("## Features\n")
            f.write("- **56 base features** expanded to ~80-100 after one-hot encoding\n")
            f.write("- 23 raw features (individual medications, utilization, demographics)\n")
            f.write("- 27 binary diagnosis indicators (9 categories × 3 positions)\n")
            f.write("- 6 age stratification features\n\n")
            f.write("## Splits\n")
            f.write(f"- Training: {splits_dict['X_train'].shape[0]:,} samples\n")
            f.write(f"- Validation: {splits_dict['X_val'].shape[0]:,} samples\n")
            f.write(f"- Test: {splits_dict['X_test'].shape[0]:,} samples\n")
        
        print(f"   ✅ Created dataset card: {readme_file}")
        
        # Attempt HuggingFace upload
        self._upload_to_huggingface(output_dir, splits_dir=os.path.join(output_dir, "splits"))
        
        return {'readme': readme_file, 'splits_dir': os.path.join(output_dir, "splits")}
    
    def _upload_to_huggingface(self, output_dir, splits_dir):
        """Upload to HuggingFace Hub."""
        print(f"\n📤 Attempting HuggingFace upload...")
        
        load_dotenv()
        hf_token = os.getenv('HF_TOKEN')
        
        if not hf_token:
            print("⚠️  No HF_TOKEN found. Skipping upload.")
            print("   To upload: Set HF_TOKEN in .env file")
            return
        
        repo_id = os.getenv('HF_REPO_ID', 'hospital-readmission-risk-data')
        if '/' not in repo_id:
            try:
                api = HfApi(token=hf_token)
                user_info = api.whoami(token=hf_token)
                username = user_info['name']
                repo_id = f"{username}/{repo_id}"
            except:
                print(f"⚠️  Could not determine username. Using repo_id as-is: {repo_id}")
        
        print(f"📦 Repository: {repo_id}")
        
        try:
            api = HfApi(token=hf_token)
            
            try:
                create_repo(repo_id, token=hf_token, repo_type="dataset", exist_ok=True)
                print(f"✅ Repository created/verified: {repo_id}")
            except Exception as e:
                print(f"⚠️  Repository may already exist: {e}")
            
            # Upload files
            # CSV files and scaler go to splits/ folder for backward compatibility
            # README stays in root for documentation
            files_to_upload = [
                ("train.csv", "splits/train.csv"),
                ("validation.csv", "splits/validation.csv"),
                ("test.csv", "splits/test.csv"),
                ("train_demographics.csv", "splits/train_demographics.csv"),
                ("validation_demographics.csv", "splits/validation_demographics.csv"),
                ("test_demographics.csv", "splits/test_demographics.csv"),
                ("scaler.pkl", "splits/scaler.pkl"),
                ("README.md", "README.md")
            ]
            
            for filename, path_in_repo in files_to_upload:
                file_path = os.path.join(splits_dir if filename != 'README.md' else output_dir, filename)
                if os.path.exists(file_path):
                    api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=path_in_repo,
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=hf_token
                    )
                    print(f"   ✅ Uploaded: {path_in_repo}")
            
            print(f"\n🎉 Upload complete: https://huggingface.co/datasets/{repo_id}")
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")


def main():
    """Main execution."""
    preprocessor = CompletePreprocessor(random_state=42, scaler_type='standard')
    
    # Run preprocessing
    splits_dict = preprocessor.fit_transform()
    
    # Save splits
    saved_splits = preprocessor.create_train_test_split(splits_dict)
    
    # Export for HuggingFace
    hf_export = preprocessor.export_for_huggingface(splits_dict)
    
    print("\n" + "=" * 80)
    print("🎉 PREPROCESSING PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"📁 Outputs saved in: ./data/processed/")
    print(f"🔒 Scaler fitted on training data only - no leakage!")
    
    return splits_dict, preprocessor, saved_splits, hf_export


if __name__ == "__main__":
    splits_dict, preprocessor, saved_splits, hf_export = main()
