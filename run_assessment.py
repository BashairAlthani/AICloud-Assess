import os

from setuptools import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class AICloudAssess:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.feature_importance = None
        self.validation_metrics = {}
        print(f"Loading data from: {data_dir}")
        self.datasets = self._load_datasets()

    def _load_datasets(self):
        """Load the most recent datasets from the data directory"""
        try:
            # Find most recent files
            legacy_files = glob.glob(os.path.join(self.data_dir, 'legacy_systems*.csv'))
            migration_files = glob.glob(os.path.join(self.data_dir, 'migration_history*.csv'))
            validation_files = glob.glob(os.path.join(self.data_dir, 'validation_metrics*.csv'))

            if not (legacy_files and migration_files and validation_files):
                raise FileNotFoundError("Required dataset files not found")

            legacy_file = max(legacy_files)
            migration_file = max(migration_files)
            validation_file = max(validation_files)

            print(f"Loading files:\n{legacy_file}\n{migration_file}\n{validation_file}")

            datasets = {
                'legacy_data': pd.read_csv(legacy_file),
                'migration_data': pd.read_csv(migration_file),
                'validation_data': pd.read_csv(validation_file)
            }

            # Merge datasets
            merged_data = datasets['legacy_data'].merge(
                datasets['migration_data'],
                on='system_id',
                how='left'
            ).merge(
                datasets['validation_data'],
                on='system_id',
                how='left'
            )

            datasets['merged_data'] = merged_data
            print(f"Successfully loaded {len(merged_data)} records")

            return datasets

        except Exception as e:
            print(f"Error loading datasets: {str(e)}")
            raise

    def prepare_training_data(self):
        """Prepare data for model training"""
        try:
            data = self.datasets['merged_data'].copy()

            # Select features for training
            feature_columns = [
                'system_age', 'lines_of_code', 'cyclomatic_complexity',
                'function_points', 'external_dependencies', 'api_connections',
                'database_links', 'response_time_ms', 'throughput_rps',
                'resource_utilization', 'data_volume_gb', 'peak_load_users',
                'annual_maintenance_cost', 'institution_size', 'system_type',
                'critical_level', 'compliance_requirements'
            ]

            # Handle categorical variables
            cat_columns = ['institution_size', 'system_type', 'critical_level',
                           'compliance_requirements']

            # Create features and target
            X = pd.get_dummies(data[feature_columns], columns=cat_columns)
            y = data['migration_success']

            # Scale numerical features
            num_columns = X.select_dtypes(include=['float64', 'int64']).columns
            X[num_columns] = self.scaler.fit_transform(X[num_columns])

            return X, y

        except Exception as e:
            print(f"Error preparing training data: {str(e)}")
            raise

    def train_model(self):
        """Train the Random Forest model"""
        try:
            print("Preparing training data...")
            X, y = self.prepare_training_data()

            print("Splitting data into train and test sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            print("Training model...")
            self.model.fit(X_train, y_train)

            # Calculate feature importance
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            # Evaluate model
            print("Evaluating model performance...")
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]

            self.validation_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_prob)
            }

            print("\nModel Training Results:")
            for metric, value in self.validation_metrics.items():
                print(f"{metric.upper()}: {value:.4f}")

            return self.validation_metrics

        except Exception as e:
            print(f"Error training model: {str(e)}")
            raise

    def generate_assessment_report(self, system_id: str, output_dir: str):
        """Generate assessment report for a specific system"""
        try:
            # Check if system exists in dataset
            if system_id not in self.datasets['merged_data']['system_id'].values:
                raise ValueError(f"System ID {system_id} not found in dataset")

            print(f"Generating assessment report for system {system_id}...")
            # Add report generation logic here

        except Exception as e:
            print(f"Error generating assessment report: {str(e)}")
            raise


# Main execution
def main():
    try:
        # Step 1: Generate synthetic data
        print("Generating synthetic data...")
        file_paths = generate_data()
        print("Data generated successfully.")

        # Step 2: Initialize the framework
        print("\nInitializing AICloud-Assess framework...")
        assessor = AICloudAssess(data_dir)

        # Step 3: Train the model
        print("\nTraining model...")
        assessor.train_model()

        # Step 4: Generate assessment for a specific system
        system_id = "SYS_0001"
        print(f"\nGenerating assessment for system {system_id}...")
        assessor.generate_assessment_report(system_id, results_dir)

        print(f"\nAssessment completed. Results saved in: {results_dir}")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please ensure all required files and directories are present.")


if __name__ == "__main__":
    main()