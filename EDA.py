import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.font_manager
from googletrans import Translator

matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  # Ensure minus sign is shown correctly

def translate_columns(input_path, output_dir_data, output_dir_mapping):
    # Read all sheets
    xls = pd.ExcelFile(input_path)
    translator = Translator()
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        mapping = []
        new_columns = []
        for col in df.columns:
            translation = translator.translate(col, src='zh-cn', dest='en').text
            eng_col = translation.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('-', '_')
            eng_col = ''.join([c for c in eng_col if c.isalnum() or c == '_'])
            mapping.append({'chinese': col, 'english': eng_col})
            new_columns.append(eng_col)
        df.columns = new_columns
        # Translate sheet name for file naming
        eng_sheet = translator.translate(sheet_name, src='zh-cn', dest='en').text
        eng_sheet = eng_sheet.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('-', '_')
        eng_sheet = ''.join([c for c in eng_sheet if c.isalnum() or c == '_'])
        # Save translated data and mapping
        data_path = os.path.join(output_dir_data, f'{eng_sheet}.csv')
        mapping_path = os.path.join(output_dir_mapping, f'column_name_mapping_{eng_sheet}.csv')
        df.to_csv(data_path, index=False)
        pd.DataFrame(mapping).to_csv(mapping_path, index=False)
        print(f"Column mapping for sheet '{sheet_name}' saved to {mapping_path}")
        print(f"Translated data for sheet '{sheet_name}' saved to {data_path}")
    return

def EDA(df, outdir, col_map=None):
    os.makedirs(outdir, exist_ok=True)
    # Save head as CSV
    df.head().to_csv(os.path.join(outdir, 'head.csv'), index=False)
    # Save info as TXT
    with open(os.path.join(outdir, 'info.txt'), 'w', encoding='utf-8') as f:
        df.info(buf=f)
    # Save describe as CSV
    df.describe(include='all').to_csv(os.path.join(outdir, 'describe.csv'))
    # Save missing values as CSV
    df.isnull().sum().to_csv(os.path.join(outdir, 'missing_values.csv'))
    # Value counts for categorical columns (use all object columns if col_map is None)
    if col_map is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    else:
        categorical_cols = col_map['categorical']
    for col in categorical_cols:
        if col in df.columns:
            vc = df[col].value_counts()
            vc.to_csv(os.path.join(outdir, f'value_counts_{col}.csv'))
    # Visualizations for numeric columns
    if col_map is None:
        numeric_cols = df.select_dtypes(include=['number']).columns
    else:
        numeric_cols = col_map['numeric']
    for col in numeric_cols:
        if col in df.columns:
            plt.figure()
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            plt.savefig(os.path.join(outdir, f'hist_{col}.png'))
            plt.close()

def get_col_map(sheet_name):
    # You can customize this mapping if you want to specify which columns are categorical/numeric for each sheet
    # For now, use all object columns as categorical, all number columns as numeric
    df = pd.read_csv(f'Data/ccRCC_Survival_Analysis_Dataset_english/{sheet_name}.csv')
    return {
        'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'numeric': df.select_dtypes(include=['number']).columns.tolist()
    }

def transform():
    input_path = 'Data/ccRCC_Survival_Analysis_Dataset.xlsx'
    output_dir_data = 'Data/ccRCC_Survival_Analysis_Dataset_english'
    output_dir_mapping = 'EDA/column_name_mapping'
    os.makedirs(output_dir_data, exist_ok=True)
    os.makedirs(output_dir_mapping, exist_ok=True)
    translate_columns(input_path, output_dir_data, output_dir_mapping)

if __name__ == '__main__':
    # transform()
    # EDA for each dataset
    datasets = {
        'training': 'training_set_603_cases',
        'internal_test': 'internal_test_set_259_cases',
        'external_test': 'external_verification_set_308_cases'
    }
    for key, fname in datasets.items():
        df = pd.read_csv(f'Data/ccRCC_Survival_Analysis_Dataset_english/{fname}.csv')
        outdir = f'EDA/{key}'
        col_map = get_col_map(fname)
        EDA(df, outdir, col_map)
