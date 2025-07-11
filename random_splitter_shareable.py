import pandas as pd
from sklearn.utils import shuffle
import analyze_data

split_train = pd.read_csv('split_train.csv', delimiter=",") # read the train split
split_test = pd.read_csv('split_test.csv', delimiter=",") # read the test split

df = pd.concat([split_train, split_test])


#42

for i in range(5):
    train_rows = []
    test_rows = []
    seed = 41+i
    df = shuffle(df, random_state=seed).reset_index(drop=True)
    # Define the categories to split stratified (DM and SYM only)
    for (disease, category), group in df[df['category'].isin(['DM', 'SYM'])].groupby(['disease', 'category']):
        group = shuffle(group, random_state=42)
        n = len(group)

        if n == 1:
            train_rows.append(group.iloc[0])
        elif n == 2:
            train_rows.append(group.iloc[0])
            test_rows.append(group.iloc[1])
        elif n == 3:
            train_rows.extend([group.iloc[0], group.iloc[1]])
            test_rows.append(group.iloc[2])
        elif n == 4:
            train_rows.extend([group.iloc[0], group.iloc[1]])
            test_rows.extend([group.iloc[2], group.iloc[3]])
        else:
            # General case for >4: 70/30 split
            split_idx = int(0.7 * n)
            train_rows.extend(group.iloc[:split_idx].to_dict(orient='records'))
            test_rows.extend(group.iloc[split_idx:].to_dict(orient='records'))

    # Now include all NOT rows in the train set
    train_rows.extend(df[df['category'] == 'NOT'].to_dict(orient='records'))

    # Convert to DataFrames
    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    testdf = analyze_data.run_test(train_df,test_df, 'rulebased3', str(seed))
