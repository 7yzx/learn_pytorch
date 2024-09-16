import pandas as pd
import numpy as np

# 创建示例数据框
data = {
    'color': ['red', 'green', np.nan, 'blue'],
    'size': ['S', 'M', 'L', 'M'],
    'price': [10.5, 20.0, np.nan, 15.0]
}

all_features = pd.DataFrame(data)
new_features = pd.get_dummies(all_features,dummy_na=True)