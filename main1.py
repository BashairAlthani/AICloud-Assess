import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Tuple
import warnings
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

warnings.filterwarnings('ignore')


class ModelSelector:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=20,
                random_state=42
            ),
            'xgboost': XGBClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42
            )
        }

    def evaluate_all_models(self, X_train, X_test, y_train, y_test):
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            results[name] = {
                'accuracy': accuracy_score(y_test, predictions),
                'precision': precision_score(y_test, predictions),
                'recall': recall_score(y_test, predictions),
                'f1': f1_score(y_test, predictions),
                'auc_roc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            }
        return results


class DocumentAnalyzer:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=50,  # Reduced number of features
            stop_words='english'
        )
        self.is_fitted = False

    def extract_features(self, documentation):
        # Convert all documentation to strings and handle NaN values
        documentation = documentation.fillna('').astype(str)

        # Extract technical terms
        tech_features = pd.DataFrame([{
            'api_count': str(doc).lower().count('api'),
            'integration_count': str(doc).lower().count('integration'),
            'security_mentions': str(doc).lower().count('security'),
            'performance_mentions': str(doc).lower().count('performance'),
            'issue_count': str(doc).lower().count('issue')
        } for doc in documentation])

        # Get TF-IDF features
        try:
            if not self.is_fitted:
                tfidf_features = self.vectorizer.fit_transform(documentation)
                self.is_fitted = True
            else:
                tfidf_features = self.vectorizer.transform(documentation)
        except ValueError:
            # If vectorization fails, create empty features
            if not self.is_fitted:
                tfidf_features = self.vectorizer.fit_transform([''] * len(documentation))
                self.is_fitted = True
            else:
                tfidf_features = self.vectorizer.transform([''] * len(documentation))

        return tech_features, tfidf_features
class AICloudAssess:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.model_selector = ModelSelector()
        self.doc_analyzer = DocumentAnalyzer()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_importance = None
        self.validation_metrics = {}
        self.feature_names = None
        self.categorical_columns = None
        self.numerical_columns = None
        self.datasets = self._load_datasets()

    def _load_datasets(self) -> Dict[str, pd.DataFrame]:
        try:
            latest_files = {
                'legacy_data': max(glob.glob(os.path.join(self.data_dir, 'legacy_systems*.csv'))),
                'migration_data': max(glob.glob(os.path.join(self.data_dir, 'migration_history*.csv'))),
                'validation_data': max(glob.glob(os.path.join(self.data_dir, 'validation_metrics*.csv')))
            }

            datasets = {
                'legacy_data': pd.read_csv(latest_files['legacy_data']),
                'migration_data': pd.read_csv(latest_files['migration_data']),
                'validation_data': pd.read_csv(latest_files['validation_data'])
            }

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
            raise Exception(f"Error loading datasets: {str(e)}")

    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = self.datasets['merged_data'].copy()

        # Extract NLP features from documentation if documentation exists
        if 'documentation' in data.columns:
            tech_features, tfidf_features = self.doc_analyzer.extract_features(
                data['documentation']
            )
        else:
            # Create dummy NLP features if documentation doesn't exist
            tech_features = pd.DataFrame(0, index=data.index,
                                         columns=['api_count', 'integration_count',
                                                  'security_mentions', 'performance_mentions',
                                                  'issue_count'])
            tfidf_features = np.zeros((len(data), 1))

        # Original feature processing
        self.category_mappings = {
            'institution_size': ['Small', 'Medium', 'Large'],
            'system_type': ['Student Information', 'Learning Management', 'Finance', 'HR', 'Research Management'],
            'critical_level': ['Low', 'Medium', 'High'],
            'compliance_requirements': ['None', 'FERPA', 'GDPR', 'Both']
        }

        feature_columns = [
            'system_age', 'lines_of_code', 'cyclomatic_complexity',
            'function_points', 'external_dependencies', 'api_connections',
            'database_links', 'response_time_ms', 'throughput_rps',
            'resource_utilization', 'data_volume_gb', 'peak_load_users',
            'annual_maintenance_cost', 'institution_size', 'system_type',
            'critical_level', 'compliance_requirements'
        ]

        self.categorical_columns = list(self.category_mappings.keys())
        self.numerical_columns = [col for col in feature_columns if col not in self.categorical_columns]

        X = data[feature_columns].copy()
        y = data['migration_success']

        # Create dummy variables
        X_encoded = X[self.numerical_columns].copy()
        for col, categories in self.category_mappings.items():
            dummies = pd.get_dummies(X[col], prefix=col)
            for category in categories:
                dummy_col = f"{col}_{category}"
                if dummy_col not in dummies.columns:
                    dummies[dummy_col] = 0
            X_encoded = pd.concat([X_encoded, dummies], axis=1)

        # Scale numerical features
        X_encoded[self.numerical_columns] = self.scaler.fit_transform(X_encoded[self.numerical_columns])

        # Convert tfidf_features to DataFrame if it's sparse
        if hasattr(tfidf_features, 'toarray'):
            tfidf_df = pd.DataFrame(
                tfidf_features.toarray(),
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            )
        else:
            tfidf_df = pd.DataFrame(
                tfidf_features,
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            )

        # Combine all features
        X_final = pd.concat([
            X_encoded.reset_index(drop=True),
            tech_features.reset_index(drop=True),
            tfidf_df.reset_index(drop=True)
        ], axis=1)

        self.feature_names = X_final.columns.tolist()
        return X_final, y

    def train_model(self):
        """Train and select the best performing model"""
        print("Preparing training data...")
        # Store X and y as class attributes
        self.X, self.y = self.prepare_training_data()

        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        print("Evaluating models...")
        model_results = self.model_selector.evaluate_all_models(
            X_train, X_test, y_train, y_test
        )

        # Select best model based on AUC-ROC
        best_model_name = max(
            model_results.items(),
            key=lambda x: x[1]['auc_roc']
        )[0]

        print(f"\nSelected best model: {best_model_name}")

        # Use the best model
        self.model = self.model_selector.models[best_model_name]
        self.model.fit(X_train, y_train)

        # Store validation metrics
        self.validation_metrics = model_results[best_model_name]

        # Calculate feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

        print("\nModel Training Results:")
        for metric, value in self.validation_metrics.items():
            print(f"{metric.upper()}: {value:.4f}")

    def prepare_system_data(self, system_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare a single system's data for prediction"""
        try:
            # Extract NLP features
            tech_features, tfidf_features = self.doc_analyzer.extract_features(
                system_data['documentation']
            )

            # Create DataFrame with numerical features
            data_encoded = system_data[self.numerical_columns].copy()

            # Add dummy variables for each category
            for col, categories in self.category_mappings.items():
                # Get the current category value
                current_value = system_data[col].iloc[0]
                # Create columns for all possible categories
                for category in categories:
                    col_name = f"{col}_{category}"
                    data_encoded[col_name] = 1 if current_value == category else 0

            # Scale numerical features
            data_encoded[self.numerical_columns] = self.scaler.transform(
                data_encoded[self.numerical_columns]
            )

            # Add NLP features
            for col in tech_features.columns:
                data_encoded[col] = tech_features[col].values

            # Convert tfidf_features to DataFrame with consistent feature names
            tfidf_df = pd.DataFrame(
                tfidf_features.toarray(),
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            )

            # Add TF-IDF features
            for col in tfidf_df.columns:
                data_encoded[col] = tfidf_df[col].values

            # Ensure we only use columns that were present during training
            missing_cols = set(self.feature_names) - set(data_encoded.columns)
            for col in missing_cols:
                data_encoded[col] = 0

            # Select only the columns used in training, in the same order
            data_encoded = data_encoded[self.feature_names]

            return data_encoded

        except Exception as e:
            print(f"Error preparing system data: {str(e)}")
            raise
    def _plot_confusion_matrix(self, output_dir: str):
        """Plot confusion matrix for the best model"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        plt.figure(figsize=(8, 6))
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

    def _plot_system_type_analysis(self, output_dir: str):
        """Plot system type distribution and success rates"""
        plt.figure(figsize=(12, 6))

        # Get system type distribution
        system_types = self.datasets['merged_data']['system_type'].value_counts()
        success_rates = self.datasets['merged_data'].groupby('system_type')['migration_success'].mean()

        # Create bar plot
        ax = plt.gca()
        x = np.arange(len(system_types))
        width = 0.35

        plt.bar(x - width / 2, system_types, width, label='Total Systems')
        plt.bar(x + width / 2, success_rates * max(system_types), width, label='Success Rate')

        plt.xlabel('System Type')
        plt.ylabel('Count / Success Rate')
        plt.title('System Type Distribution and Success Rates')
        plt.xticks(x, system_types.index, rotation=45)
        plt.legend()

        # Add second y-axis for success rate percentage
        ax2 = ax.twinx()
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'system_type_analysis.png'))
        plt.close()

    def _plot_risk_distribution(self, output_dir: str):
        """Plot risk distribution analysis"""
        plt.figure(figsize=(10, 6))

        # Calculate risk metrics for all systems
        risk_scores = []
        for _, system in self.datasets['merged_data'].iterrows():
            risk = self._assess_risk(system.to_frame().T)
            risk_scores.append([
                risk['security_risk'],
                risk['technical_risk'],
                risk['operational_risk'],
                risk['financial_risk']
            ])

        risk_df = pd.DataFrame(
            risk_scores,
            columns=['Security Risk', 'Technical Risk', 'Operational Risk', 'Financial Risk']
        )

        # Create box plot
        sns.boxplot(data=risk_df)
        plt.title('Risk Distribution Analysis')
        plt.ylabel('Risk Score')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'risk_distribution.png'))
        plt.close()
    def _assess_risk(self, system_data: pd.DataFrame) -> Dict:
        """Perform detailed risk assessment"""
        # Calculate security risk
        security_risk = self._calculate_security_risk(system_data)

        # Calculate technical risk
        technical_risk = self._calculate_technical_risk(system_data)

        # Calculate operational risk
        operational_risk = self._calculate_operational_risk(system_data)

        # Calculate financial risk
        financial_risk = self._calculate_financial_risk(system_data)

        # Calculate overall risk score
        overall_risk = np.mean([security_risk, technical_risk, operational_risk, financial_risk])

        return {
            'overall_risk': overall_risk,
            'security_risk': security_risk,
            'technical_risk': technical_risk,
            'operational_risk': operational_risk,
            'financial_risk': financial_risk
        }

    def _calculate_security_risk(self, system_data: pd.DataFrame) -> float:
        """Calculate security risk score"""
        # Extract relevant features
        compliance_req = system_data['compliance_requirements'].iloc[0]
        critical_level = system_data['critical_level'].iloc[0]
        data_volume = system_data['data_volume_gb'].iloc[0]

        # Calculate base risk score
        base_score = 0

        # Adjust for compliance requirements
        if compliance_req == 'Both':
            base_score += 30
        elif compliance_req in ['FERPA', 'GDPR']:
            base_score += 20

        # Adjust for critical level
        if critical_level == 'High':
            base_score += 30
        elif critical_level == 'Medium':
            base_score += 20

        # Adjust for data volume
        base_score += min(20, data_volume / 1000)  # Max 20 points for data volume

        return min(100, base_score)  # Cap at 100

    def _calculate_technical_risk(self, system_data: pd.DataFrame) -> float:
        """Calculate technical risk score"""
        # Extract relevant features
        system_age = system_data['system_age'].iloc[0]
        complexity = system_data['cyclomatic_complexity'].iloc[0]
        dependencies = system_data['external_dependencies'].iloc[0]

        # Calculate base risk score
        base_score = 0

        # Adjust for system age
        base_score += min(30, system_age * 2)  # Max 30 points for age

        # Adjust for complexity
        base_score += min(40, complexity / 2)  # Max 40 points for complexity

        # Adjust for dependencies
        base_score += min(30, dependencies)  # Max 30 points for dependencies

        return min(100, base_score)  # Cap at 100

    def _calculate_operational_risk(self, system_data: pd.DataFrame) -> float:
        """Calculate operational risk score"""
        # Extract relevant features
        peak_load = system_data['peak_load_users'].iloc[0]
        response_time = system_data['response_time_ms'].iloc[0]
        resource_util = system_data['resource_utilization'].iloc[0]

        # Calculate base risk score
        base_score = 0

        # Adjust for peak load
        base_score += min(30, peak_load / 1000)  # Max 30 points

        # Adjust for response time
        base_score += min(40, response_time / 10)  # Max 40 points

        # Adjust for resource utilization
        base_score += min(30, resource_util * 100)  # Max 30 points

        return min(100, base_score)  # Cap at 100

    def _calculate_financial_risk(self, system_data: pd.DataFrame) -> float:
        """Calculate financial risk score"""
        # Extract relevant features
        maintenance_cost = system_data['annual_maintenance_cost'].iloc[0]

        # Calculate base risk score
        base_score = min(100, maintenance_cost / 1000)  # Scale maintenance cost

        return base_score



    def _estimate_resources(self, system_data: pd.DataFrame) -> Dict:
        """Estimate required resources for migration"""
        # Calculate timeline
        timeline = self._generate_timeline(system_data)

        # Calculate costs
        costs = self._calculate_costs(system_data)

        # Calculate resource requirements
        resources = self._calculate_resource_requirements(system_data)

        return {
            'timeline': timeline,
            'costs': costs,
            'resources': resources
        }

    def _generate_timeline(self, system_data: pd.DataFrame) -> Dict:
        """Generate migration timeline"""
        # Base timeline calculations
        complexity_factor = (system_data['cyclomatic_complexity'].iloc[0] / 100)
        size_factor = (system_data['data_volume_gb'].iloc[0] / 1000)

        # Calculate phase durations (in days)
        planning_duration = max(14, int(30 * complexity_factor))
        migration_duration = max(30, int(60 * (complexity_factor + size_factor)))
        testing_duration = max(14, int(30 * complexity_factor))

        # Generate timeline dates
        start_date = datetime.now()
        planning_start = start_date
        planning_end = planning_start + timedelta(days=planning_duration)
        migration_start = planning_end
        migration_end = migration_start + timedelta(days=migration_duration)
        testing_start = migration_end
        testing_end = testing_start + timedelta(days=testing_duration)

        return {
            'planning_start': planning_start.strftime('%Y-%m-%d'),
            'planning_end': planning_end.strftime('%Y-%m-%d'),
            'migration_start': migration_start.strftime('%Y-%m-%d'),
            'migration_end': migration_end.strftime('%Y-%m-%d'),
            'testing_start': testing_start.strftime('%Y-%m-%d'),
            'testing_end': testing_end.strftime('%Y-%m-%d'),
            'total_duration': planning_duration + migration_duration + testing_duration
        }

    def _calculate_costs(self, system_data: pd.DataFrame) -> Dict:
        """Calculate migration costs"""
        # Base cost calculations
        base_infrastructure_cost = 50000
        base_personnel_cost = 100000
        base_training_cost = 20000
        base_maintenance_cost = 30000

        # Adjustment factors
        complexity_factor = (system_data['cyclomatic_complexity'].iloc[0] / 100)
        size_factor = (system_data['data_volume_gb'].iloc[0] / 1000)
        criticality_factor = 1.5 if system_data['critical_level'].iloc[0] == 'High' else 1.0

        # Calculate adjusted costs
        infrastructure_cost = base_infrastructure_cost * (1 + size_factor)
        personnel_cost = base_personnel_cost * (1 + complexity_factor) * criticality_factor
        training_cost = base_training_cost * (1 + complexity_factor)
        maintenance_cost = base_maintenance_cost * (1 + size_factor)

        return {
            'infrastructure': round(infrastructure_cost, 2),
            'personnel': round(personnel_cost, 2),
            'training': round(training_cost, 2),
            'maintenance': round(maintenance_cost, 2),
            'total': round(infrastructure_cost + personnel_cost + training_cost + maintenance_cost, 2)
        }

    def _calculate_resource_requirements(self, system_data: pd.DataFrame) -> Dict:
        """Calculate required resources"""
        # Base resource calculations
        complexity_factor = (system_data['cyclomatic_complexity'].iloc[0] / 100)
        size_factor = (system_data['data_volume_gb'].iloc[0] / 1000)

        # Calculate team size
        developers = max(2, int(4 * complexity_factor))
        system_engineers = max(1, int(2 * size_factor))
        testers = max(1, int(2 * complexity_factor))

        return {
            'team_size': {
                'developers': developers,
                'system_engineers': system_engineers,
                'testers': testers,
                'total': developers + system_engineers + testers
            },
            'estimated_effort_hours': round(160 * (developers + system_engineers + testers), 0)
        }

    def _generate_risk_visualization(self, assessment: Dict, output_dir: str):
        """Generate risk visualization"""
        risk_data = assessment['risk_assessment']

        # Create radar chart
        categories = ['Security Risk', 'Technical Risk', 'Operational Risk', 'Financial Risk']
        values = [
            risk_data['security_risk'],
            risk_data['technical_risk'],
            risk_data['operational_risk'],
            risk_data['financial_risk']
        ]

        # Create figure
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Risk Assessment'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            title="Risk Assessment Radar Chart"
        )

        fig.write_html(os.path.join(output_dir, 'risk_assessment.html'))

    def _generate_resource_visualization(self, assessment: Dict, output_dir: str):
        """Generate resource visualization"""
        resource_data = assessment['resource_estimates']

        # Create Gantt chart for timeline
        fig = px.timeline(
            pd.DataFrame([
                dict(Task="Planning",
                     Start=resource_data['timeline']['planning_start'],
                     Finish=resource_data['timeline']['planning_end']),
                dict(Task="Migration",
                     Start=resource_data['timeline']['migration_start'],
                     Finish=resource_data['timeline']['migration_end']),
                dict(Task="Testing",
                     Start=resource_data['timeline']['testing_start'],
                     Finish=resource_data['timeline']['testing_end'])
            ]),
            x_start="Start",
            x_end="Finish",
            y="Task",
            title="Project Timeline"
        )

        fig.write_html(os.path.join(output_dir, 'resource_timeline.html'))

    def _generate_performance_visualization(self, assessment: Dict, output_dir: str):
        """Generate performance visualization"""
        if 'performance_analysis' in assessment:
            perf_data = assessment['performance_analysis']

            # Create performance comparison chart
            fig = go.Figure(data=[
                go.Bar(name='Current',
                       x=['Response Time', 'Throughput', 'Availability'],
                       y=[perf_data['current']['response_time'],
                          perf_data['current']['throughput'],
                          perf_data['current']['availability']]),
                go.Bar(name='Predicted',
                       x=['Response Time', 'Throughput', 'Availability'],
                       y=[perf_data['predicted']['response_time'],
                          perf_data['predicted']['throughput'],
                          perf_data['predicted']['availability']])
            ])

            fig.update_layout(
                barmode='group',
                title="Performance Comparison"
            )

            fig.write_html(os.path.join(output_dir, 'performance_comparison.html'))
    def generate_assessment_report(self, system_id: str, output_dir: str):
        """Generate comprehensive assessment report with visualizations"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Generate analysis visualizations first
            print("Generating analysis visualizations...")
            self.generate_analysis_visualizations(output_dir)

            # Get assessment results
            print("Generating system assessment...")
            assessment = self.assess_system(system_id)

            # Generate system-specific visualizations
            print("Generating system-specific visualizations...")
            self._generate_risk_visualization(assessment, output_dir)
            self._generate_resource_visualization(assessment, output_dir)
            self._generate_performance_visualization(assessment, output_dir)

            # Generate HTML report
            report_path = os.path.join(output_dir, f'assessment_report_{system_id}.html')
            self._generate_html_report(assessment, report_path)

            print(f"\nAssessment report generated: {report_path}")

        except Exception as e:
            print(f"Error generating assessment report: {str(e)}")
            raise

    def _generate_html_report(self, assessment: Dict, output_path: str):
        """Generate HTML assessment report"""
        html_content = f"""
        <html>
        <head>
            <title>Cloud Migration Assessment Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
                .metric {{ margin: 10px 0; }}
                .high-risk {{ color: red; }}
                .medium-risk {{ color: orange; }}
                .low-risk {{ color: green; }}
            </style>
        </head>
        <body>
            <h1>Cloud Migration Assessment Report</h1>
            <div class="section">
                <h2>System Information</h2>
                <div class="metric">System ID: {assessment['system_id']}</div>
                <div class="metric">Migration Probability: {assessment['migration_probability']:.2%}</div>
            </div>

            <div class="section">
                <h2>Risk Assessment</h2>
                <div class="metric">Overall Risk: {self._format_risk_level(assessment['risk_assessment']['overall_risk'])}</div>
                <div class="metric">Security Risk: {assessment['risk_assessment']['security_risk']:.2f}%</div>
                <div class="metric">Technical Risk: {assessment['risk_assessment']['technical_risk']:.2f}%</div>
                <div class="metric">Operational Risk: {assessment['risk_assessment']['operational_risk']:.2f}%</div>
                <div class="metric">Financial Risk: {assessment['risk_assessment']['financial_risk']:.2f}%</div>
            </div>

            <div class="section">
                <h2>Resource Estimates</h2>
                <h3>Timeline</h3>
                <div class="metric">Total Duration: {assessment['resource_estimates']['timeline']['total_duration']} days</div>

                <h3>Costs</h3>
                <div class="metric">Total Cost: ${assessment['resource_estimates']['costs']['total']:,.2f}</div>
                <div class="metric">Infrastructure: ${assessment['resource_estimates']['costs']['infrastructure']:,.2f}</div>
                <div class="metric">Personnel: ${assessment['resource_estimates']['costs']['personnel']:,.2f}</div>
                <div class="metric">Training: ${assessment['resource_estimates']['costs']['training']:,.2f}</div>
                <div class="metric">Maintenance: ${assessment['resource_estimates']['costs']['maintenance']:,.2f}</div>

                <h3>Team Requirements</h3>
                <div class="metric">Total Team Size: {assessment['resource_estimates']['resources']['team_size']['total']}</div>
                <div class="metric">Estimated Effort: {assessment['resource_estimates']['resources']['estimated_effort_hours']} hours</div>
            </div>

            <div class="section">
                <h2>NLP Analysis</h2>
                <h3>Technical Complexity Indicators</h3>
                <div class="metric">API References: {assessment['nlp_analysis']['technical_complexity']['api_count']}</div>
                <div class="metric">Integration Points: {assessment['nlp_analysis']['technical_complexity']['integration_count']}</div>
                <div class="metric">Security Concerns: {assessment['nlp_analysis']['technical_complexity']['security_mentions']}</div>
            </div>

            <div class="section">
                <h2>Model Selection Results</h2>
                <div class="metric">Selected Model: Gradient Boosting</div>
                <div class="metric">Model Accuracy: {self.validation_metrics['accuracy']:.2f}</div>
                <div class="metric">Model Precision: {self.validation_metrics['precision']:.2f}</div>
                <div class="metric">Model Recall: {self.validation_metrics['recall']:.2f}</div>
            </div>
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)

    @staticmethod
    def _format_risk_level(risk_score: float) -> str:
        """Format risk level with color coding"""
        if risk_score >= 70:
            return f'<span class="high-risk">High ({risk_score:.2f}%)</span>'
        elif risk_score >= 30:
            return f'<span class="medium-risk">Medium ({risk_score:.2f}%)</span>'
        else:
            return f'<span class="low-risk">Low ({risk_score:.2f}%)</span>'

    def generate_analysis_visualizations(self, output_dir: str):
        """Generate and save comprehensive analysis visualizations"""
        try:
            os.makedirs(output_dir, exist_ok=True)

            # 1. Model Selection Performance Comparison
            self._plot_model_comparison(output_dir)

            # 2. Feature Importance Analysis
            self._plot_feature_importance(output_dir)

            # 3. ROC Curves Comparison
            self._plot_roc_curves(output_dir)

            # 4. Confusion Matrix for Best Model
            self._plot_confusion_matrix(output_dir)

            # 5. Learning Curves
            self._plot_learning_curves(output_dir)

            # 6. Feature Correlation Matrix
            self._plot_correlation_matrix(output_dir)

            # 7. System Type Distribution Analysis
            self._plot_system_type_analysis(output_dir)

            # 8. NLP Feature Impact
            self._plot_nlp_feature_impact(output_dir)

            # 9. Risk Distribution Analysis
            self._plot_risk_distribution(output_dir)

        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            raise

    def _plot_model_comparison(self, output_dir: str):
        """Create comparative visualization of model performances"""
        # Prepare data
        models = list(self.model_selector.models.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        performances = []

        for model_name in models:
            model = self.model_selector.models[model_name]
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            performances.append({
                'model': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_prob)
            })

        # Create visualization
        performance_df = pd.DataFrame(performances)
        plt.figure(figsize=(12, 6))

        # Plot grouped bar chart
        x = np.arange(len(models))
        width = 0.15

        for i, metric in enumerate(metrics):
            plt.bar(x + i * width, performance_df[metric], width, label=metric.upper())

        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width * 2, models, rotation=45)
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
        plt.close()

    def _plot_feature_importance(self, output_dir: str):
        """Visualize feature importance analysis"""
        plt.figure(figsize=(12, 8))

        # Sort features by importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True)

        # Plot horizontal bar chart for top 20 features
        top_20 = importance_df.tail(20)
        sns.barplot(data=top_20, y='feature', x='importance')

        plt.title('Top 20 Most Important Features')
        plt.xlabel('Importance Score')
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()

    def _plot_roc_curves(self, output_dir: str):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))

        for model_name, model in self.model_selector.models.items():
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)

            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
        plt.close()

    def _plot_learning_curves(self, output_dir: str):
        """Generate learning curves visualization"""
        plt.figure(figsize=(10, 6))

        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.X, self.y,
            cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.plot(train_sizes, train_mean, label='Training score')
        plt.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.1
        )
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        plt.fill_between(
            train_sizes,
            test_mean - test_std,
            test_mean + test_std,
            alpha=0.1
        )

        plt.xlabel('Training Size')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True)

        plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
        plt.close()

    def _plot_correlation_matrix(self, output_dir: str):
        """Visualize feature correlation matrix"""
        plt.figure(figsize=(15, 12))

        # Calculate correlation matrix for numerical features
        correlation_matrix = self.X[self.numerical_columns].corr()

        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix), k=1)
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            square=True
        )

        plt.title('Feature Correlation Matrix')
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()

    def _plot_nlp_feature_impact(self, output_dir: str):
        """Visualize impact of NLP features"""
        plt.figure(figsize=(10, 6))

        # Compare model performance with and without NLP features
        performance_comparison = {
            'With NLP': self.validation_metrics,
            'Without NLP': self._get_performance_without_nlp()
        }

        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        x = np.arange(len(metrics))
        width = 0.35

        plt.bar(x - width / 2, [performance_comparison['Without NLP'][m] for m in metrics],
                width, label='Without NLP')
        plt.bar(x + width / 2, [performance_comparison['With NLP'][m] for m in metrics],
                width, label='With NLP')

        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Impact of NLP Features on Model Performance')
        plt.xticks(x, metrics)
        plt.legend()

        plt.savefig(os.path.join(output_dir, 'nlp_impact.png'))
        plt.close()

    def _get_performance_without_nlp(self) -> Dict:
        """Calculate model performance without NLP features"""
        # Prepare data without NLP features
        X_without_nlp = self.X[self.numerical_columns +
                               [col for col in self.X.columns
                                if any(prefix in col for prefix in self.category_mappings.keys())]]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_without_nlp, self.y, test_size=0.2, random_state=42
        )

        # Train model
        model = self.model.__class__(**self.model.get_params())
        model.fit(X_train, y_train)

        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_prob)
        }

    def assess_system(self, system_id: str) -> Dict:
        try:
            system_data = self.datasets['merged_data'][
                self.datasets['merged_data']['system_id'] == system_id
                ]

            if len(system_data) == 0:
                raise ValueError(f"System ID {system_id} not found in dataset")

            # Extract features including NLP features
            tech_features, tfidf_features = self.doc_analyzer.extract_features(
                system_data['documentation']
            )

            # Prepare system data
            X = self.prepare_system_data(system_data)

            # Get migration probability
            migration_probability = self.model.predict_proba(X)[0][1]

            # Generate detailed assessment
            assessment = {
                'system_id': system_id,
                'migration_probability': migration_probability,
                'risk_assessment': self._assess_risk(system_data),
                'resource_estimates': self._estimate_resources(system_data),
                'performance_analysis': self._analyze_performance(system_data),
                'nlp_analysis': {
                    'technical_complexity': tech_features.iloc[0].to_dict(),
                    'key_terms': self._extract_key_terms(tfidf_features)
                }
            }

            return assessment

        except Exception as e:
            print(f"Error in assessment: {str(e)}")
            raise

    def _analyze_performance(self, system_data: pd.DataFrame) -> Dict:
        """Analyze current and predicted performance metrics"""
        try:
            # Get current performance metrics
            current_performance = {
                'response_time': system_data['response_time_ms'].iloc[0],
                'throughput': system_data['throughput_rps'].iloc[0],
                'availability': 99.0  # Assumed current availability
            }

            # Calculate predicted performance in cloud
            # Assuming improvements based on typical cloud migration benefits
            predicted_performance = {
                'response_time': current_performance['response_time'] * 0.7,  # 30% improvement
                'throughput': current_performance['throughput'] * 1.5,  # 50% improvement
                'availability': 99.95  # Expected cloud availability
            }

            # Calculate improvement percentages
            improvements = {
                'response_time_improvement': round((1 - predicted_performance['response_time'] /
                                                    current_performance['response_time']) * 100, 2),
                'throughput_improvement': round((predicted_performance['throughput'] /
                                                 current_performance['throughput'] - 1) * 100, 2),
                'availability_improvement': round(predicted_performance['availability'] -
                                                  current_performance['availability'], 2)
            }

            # Generate performance recommendations
            recommendations = self._generate_performance_recommendations(current_performance)

            return {
                'current': current_performance,
                'predicted': predicted_performance,
                'improvements': improvements,
                'recommendations': recommendations
            }

        except Exception as e:
            print(f"Error analyzing performance: {str(e)}")
            raise

    def _generate_performance_recommendations(self, current_performance: Dict) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []

        # Response time recommendations
        if current_performance['response_time'] > 500:  # If response time > 500ms
            recommendations.append(
                "High response time detected. Cloud migration can help through:\n"
                "- Global content delivery networks (CDN)\n"
                "- Auto-scaling capabilities\n"
                "- Optimized infrastructure"
            )

        # Throughput recommendations
        if current_performance['throughput'] < 100:  # If throughput < 100 requests/second
            recommendations.append(
                "Low throughput identified. Cloud benefits include:\n"
                "- Elastic load balancing\n"
                "- Horizontal scaling\n"
                "- Distributed processing"
            )

        # Availability recommendations
        if current_performance['availability'] < 99.9:
            recommendations.append(
                "Availability can be improved through:\n"
                "- Multi-region deployment\n"
                "- Automated failover\n"
                "- Built-in redundancy"
            )

        # Add general recommendations if none specific
        if not recommendations:
            recommendations.append(
                "Current performance metrics are within acceptable ranges.\n"
                "Cloud migration can still provide:\n"
                "- Enhanced scalability\n"
                "- Improved resource utilization\n"
                "- Better cost optimization"
            )

        return recommendations
    def _extract_key_terms(self, tfidf_features):
        """Extract most important terms from TF-IDF features"""
        feature_names = self.vectorizer.get_feature_names_out()
        important_terms = []

        if tfidf_features.shape[1] > 0:
            # Get top 10 terms by TF-IDF score
            scores = tfidf_features.toarray()[0]
            top_indices = scores.argsort()[-10:][::-1]
            important_terms = [(feature_names[i], scores[i]) for i in top_indices]

        return important_terms

    # Keep all other existing methods from your original AICloudAssess class...

def main():
    try:
        # Initialize framework
        assessor = AICloudAssess('cloud_migration_data')

        # Train the model
        print("Training model...")
        assessor.train_model()

        # Generate assessment for a specific system
        system_id = "SYS_0001"
        print(f"\nGenerating assessment for system {system_id}...")
        assessor.generate_assessment_report(system_id, "assessment_results")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()