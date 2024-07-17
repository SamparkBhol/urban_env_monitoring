# Urban Environment Monitoring

## Objective
Develop a CNN to segment satellite images to monitor urban growth and environmental changes over time.

## Project Structure
- `data/`: Contains raw and processed data.
- `notebooks/`: Jupyter notebooks for data preprocessing, model training, and evaluation.
- `src/`: Source code for data loading, model definition, training, and evaluation.
- `scripts/`: Shell scripts for preprocessing, training, and evaluating the model.
- `results/`: Contains segmentation and change detection maps.
- `README.md`: Project documentation.
- `requirements.txt`: Python dependencies.

## Implementation Steps
1. **Data Collection and Preprocessing**
2. **Model Architecture**
3. **Training the Model**
4. **Post-Processing**
5. **Visualization and Analysis**
6. **Deployment**

## How to Run
1. Preprocess data:
   ```bash
   bash scripts/preprocess_data.sh
   ```
2. Train the model:
   ```bash
   bash scripts/train_model.sh
   ```
3. Evaluate the model:
   ```bash
   bash scripts/evaluate_model.sh
   ```
