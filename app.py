import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

def load_data():
    """Load and prepare the dataset """
    try:
        # Add file uploader to sidebar
        st.sidebar.header('Select feature data file')
        use_default = st.sidebar.checkbox('Use default features file', value=True)
        
        if use_default:
            # default file for the project
            df = pd.read_csv('result/masked_out_images/extracted_features.csv')
            st.sidebar.success('Loaded default features file')
        else:
            # upload csv if not
            uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f'Loaded {uploaded_file.name}')
            else:
                st.sidebar.warning('Please upload a CSV file')
                return None
        
        # Load lables
        try:
            labels_df = pd.read_csv('data/labels.csv')
            df['label'] = labels_df['label']
        except FileNotFoundError: #create lables if needed
            if 'label' not in df.columns:
                st.warning("No labels found. Creating synthetic labels for demonstration...")
                df['label'] = ((df['asymmetry'] > df['asymmetry'].mean()) & 
                            (df['irregular_pigmentation'] > df['irregular_pigmentation'].mean())).astype(int)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

#sample model traning v.1
def train_model(X, y, test_size=0.3):
    """Train and validate the model using cross-validation with dynamic splits and handle missing values"""
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    # sanity check - whether we haev both classes
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        st.error("Dataset must contain both melanoma and non-melanoma cases")
        return None, 0, np.zeros((2,2)), None, None, None, None
    
    # Count samples in each class
    class_counts = np.bincount(y)
    min_class_count = np.min(class_counts)
    
    # determine optimal number of splits for cross valdidation
    n_splits = min(5, max(2, min_class_count // 2))
    
    # create preprocessing and model pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
        ('scaler', StandardScaler()),  # Scale features
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    
    # split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # cross validationg the training dataset
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='balanced_accuracy')
    
    # Traning the final model on full training data
    pipeline.fit(X_train, y_train)
    
    # make predictions on test set
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # detailed report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # class distribution info print
    st.info(f"Class distribution - Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
    st.info(f"Using {n_splits}-fold cross-validation based on class sizes")
    
    # data quality information
    n_missing = X.isna().sum().sum()
    if n_missing > 0:
        st.warning(f"Found {n_missing} missing values. These have been imputed using mean strategy.")
    
    return pipeline, test_accuracy, conf_matrix, X_test, y_test, y_pred, {
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'class_report': class_report
    }

def plot_feature_importance(model, feature_names):
    """Plot feature importance from the pipeline's classifier"""
    importance = abs(model.named_steps['classifier'].coef_[0])
    fig = px.bar(
        x=feature_names,
        y=importance,
        title='Feature Importance',
        labels={'x': 'Features', 'y': 'Importance'}
    )
    return fig

def plot_confusion_matrix(conf_matrix):
    """Plot confusion matrix"""
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        colorscale='RdBu'
    ))
    fig.update_layout(title='Confusion Matrix')
    return fig

def main():
    st.title('Melanoma Model Test - Penguins')
    st.write('Select features and click train model')

    # Initialize session state for model
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = None

    # Load data 1
    df = load_data()
    if df is None:
        return

    # Sidebar
    st.sidebar.header('Feature Selection')
    all_features = ['pigment_network', 'blue_veil', 'vascular', 'globules', 
                    'streaks', 'irregular_pigmentation', 'regression',
                    'asymmetry', 'compactness', 'convexity']
    
    selected_features = st.sidebar.multiselect(
        'Select features for analysis',
        all_features,
        default=['pigment_network', 'asymmetry', 'compactness']
    )

    if not selected_features:
        st.warning('Please select at least one feature.')
        return

    # LOAD DATA
    X = df[selected_features]
    y = df['label'] if 'label' in df.columns else np.zeros(len(df))

    # MODEL TRAINIG !! 
    if st.sidebar.button('Train Model'):
        with st.spinner('Training model...'):
            model, accuracy, conf_matrix, X_test, y_test, y_pred, metrics = train_model(X, y)
            
            if model is not None:
                # save model and features in session
                st.session_state.model = model
                st.session_state.selected_features = selected_features
                
                # display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader('Model Performance')
                    st.write(f'Test Accuracy: {accuracy:.2f}')
                    st.write(f'Cross-validation Accuracy: {metrics["cv_mean"]:.2f} (±{metrics["cv_std"]:.2f})')
                    
                    # feature importance plot
                    fig_importance = plot_feature_importance(model, selected_features)
                    st.plotly_chart(fig_importance)
                    
                    # classification report
                    st.subheader('Classification Report')
                    report = pd.DataFrame(metrics['class_report']).transpose()
                    st.dataframe(report.style.format("{:.2f}"))

                with col2:
                    st.subheader('Confusion Matrix')
                    fig_conf = plot_confusion_matrix(conf_matrix)
                    st.plotly_chart(fig_conf)
                    
                    # ROC curve
                    st.subheader('ROC Curve')
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fig_roc = plot_roc_curve(y_test, y_prob)
                    st.plotly_chart(fig_roc)

                # feature distributions
                st.subheader('Feature Distributions')
                for feature in selected_features:
                    fig = px.histogram(df, x=feature, 
                                     color=df['label'] if 'label' in df.columns else None,
                                     title=f'{feature} Distribution by Class',
                                     barmode='overlay')
                    st.plotly_chart(fig)

#    # pridiction section - still wonkey !!!
#     st.sidebar.header('Make Prediction - Still Wonkey!!')
#     if st.session_state.model is None:
#         st.sidebar.warning('Please train the model first before making predictions.')
#     else:
#         make_prediction = st.sidebar.button('Predict New Case')
#         if make_prediction:
#             st.subheader('Enter Feature Values')
#             user_input = {}
#             col1, col2 = st.columns(2)
            
#             # Use the features from the trained model
#             for i, feature in enumerate(st.session_state.selected_features):
#                 with col1 if i % 2 == 0 else col2:
#                     user_input[feature] = st.number_input(
#                         f'Enter {feature}',
#                         value=float(df[feature].mean()),
#                         format='%f',
#                         key=f'input_{feature}'  # Unique key for each input
#                     )

#             if st.button('Submit Prediction'):
#                 # Create DataFrame with the same features used in training
#                 input_data = pd.DataFrame([user_input])
                
#                 # Make prediction
#                 prediction = st.session_state.model.predict(input_data)[0]
#                 probability = st.session_state.model.predict_proba(input_data)[0]
                
#                 # Display results
#                 st.subheader('Prediction Result')
                
#                 # Result container with styling
#                 result_container = st.container()
#                 with result_container:
#                     prediction_text = "Melanoma" if prediction == 1 else "Non-Melanoma"
#                     confidence = max(probability)
                    
#                     # Style based on prediction
#                     if prediction == 1:
#                         st.error(f"⚠️ Prediction: {prediction_text}")
#                     else:
#                         st.success(f"✅ Prediction: {prediction_text}")
                    
#                     st.write(f"Confidence: {confidence:.2f}")
                    
#                     # probability gauge chart
#                     fig = go.Figure(go.Indicator(
#                         mode="gauge+number",
#                         value=probability[1],
#                         title={'text': "Melanoma Probability"},
#                         gauge={
#                             'axis': {'range': [0, 1]},
#                             'steps': [
#                                 {'range': [0, 0.4], 'color': "lightgreen"},
#                                 {'range': [0.4, 0.7], 'color': "yellow"},
#                                 {'range': [0.7, 1], 'color': "red"}
#                             ],
#                             'threshold': {
#                                 'line': {'color': "red", 'width': 4},
#                                 'thickness': 0.75,
#                                 'value': 0.7
#                             }
#                         }))
#                     st.plotly_chart(fig)
                    
#                     # Feature values table
#                     st.subheader("Input Feature Values")
#                     st.dataframe(input_data)

def plot_roc_curve(y_true, y_prob):
    """Plot ROC curve using plotly"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                            name=f'ROC curve (AUC = {roc_auc:.2f})',
                            mode='lines'))
    
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                            name='Random Classifier',
                            mode='lines',
                            line=dict(dash='dash')))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=600,
        height=600
    )
    
    return fig
    


if __name__ == '__main__':
    main()