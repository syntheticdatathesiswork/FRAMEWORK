import datetime
import json
import numpy as np
import pandas as pd

def numpy_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def create_model_card(model_name, overview, preproc_file, random_state, train_test_split, features,
                      target, train_set_name, test_set_name, evaluation_metrics, intended_use, ethical_bias_concerns, output_filename):
    """
    Generates a Markdown model card for an ML model and writes it to a file.
    
    Parameters:
      - model_name (str): Name of the model.
      - overview (str): Brief overview of the model.
      - preproc_file (str): Name of the data pre-processing file.
      - random_state (int): The random state used for reproducibility.
      - train_test_split (float): Train/test split ratio.
      - features (list): List of feature names.
      - target (str): Target variable name.
      - train_set_name (str): Name of the training set file.
      - test_set_name (str): Name of the test set file.
      - evaluation_metrics (dict): Dictionary of evaluation metrics.
      - intended_use (str): Description of the intended use of the model.
      - ethical_bias_concerns (str): Information about ethical considerations or bias concerns.
      - output_filename (str): The filename (with path if needed) for the generated Markdown model card.
      
    Returns:
      None
    """
    # Get the current date.
    creation_date = datetime.date.today().strftime("%Y-%m-%d")
    
    # Create a copy of the evaluation metrics and remove the "y_test" key if present.
    metrics_to_export = evaluation_metrics.copy()
    metrics_to_export.pop("y_test", None)
    metrics_to_export.pop("y_pred", None)
    metrics_to_export.pop("y_proba", None)
    
    # Convert the evaluation_metrics dictionary to a nicely formatted JSON string,
    # using the numpy_encoder to handle any numpy types or pandas Series.
    metrics_md = "```json\n" + json.dumps(metrics_to_export, indent=4, default=numpy_encoder) + "\n```"
    
    # Build the Markdown content.
    model_card = f"""
# Model Card: {model_name}

**Overview:**  
{overview}

**Dataset Information:**  
- **Features:** {", ".join(features)}
- **Target:** {target}

**Pre-processing Details:**  
- **Data Pre-processing File:** {preproc_file}
- **Random State:** {random_state}
- **Train/Test Split Ratio:** {train_test_split}
- **Train Set Name:** {train_set_name}
- **Test Set Name:** {test_set_name}

**Evaluation Metrics:**  
{metrics_md}

**Intended Use:**  
{intended_use}

**Ethical and Bias Concerns:**  
{ethical_bias_concerns}

**Date of Creation:** {creation_date}
"""
    # Write the model card to the specified output file.
    with open(output_filename, "w") as f:
        f.write(model_card.strip())
    
    print(f"Model card saved as {output_filename}")
