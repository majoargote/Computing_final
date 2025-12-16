import pandas as pd
from pathlib import Path

#define into three classes, per the data in the csv

def load_wine_data(data_path=None):
    base = Path(__file__).resolve().parent  # folder containing data_loader.py
    csv_path = base / "data" / "WineQT.csv" if data_path is None else Path(data_path)

    df = pd.read_csv(csv_path)
    
    # Drop non-features
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    # create three quality classes
    def bin_quality(score):
        if score <= 4:
            return 'poor'
        elif score <= 6:
            return 'average'
        else:  # 7-8
            return 'good'
    
    df['quality_class'] = df['quality'].apply(bin_quality)
    
    # Separate features and target
    X = df.drop(['quality', 'quality_class'], axis=1)
    y = df['quality_class']
    
    # Print info
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"\nClass distribution:")
    for c in ['poor', 'average', 'good']:
        if c in y.values:
            count = (y == c).sum()
            pct = count / len(y) * 100
            print(f"  {c:10s}: {count:4d} samples ({pct:5.1f}%)")
    
    return X, y


def get_feature_names():

    return [
        'fixed_acidity',
        'volatile_acidity',
        'citric_acid',
        'residual_sugar',
        'chlorides',
        'free_sulfur_dioxide',
        'total_sulfur_dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol'
    ]