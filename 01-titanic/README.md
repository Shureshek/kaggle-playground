CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    cat_features=cat_features,
    random_seed=42,
    verbose=100,
    eval_metric='Accuracy'
)
Validation Accuracy: 0.8324
Score на Kaggle: 0.79186