import os.path

import pandas as pd


def save_acc_to_excel(excel_file_name, head_acc_rates, client_acc_rates):
    excel_file_path = '../results/' + excel_file_name + '.xlsx'
    data_dict = {'Head Acc': head_acc_rates}
    for client_id, client_acc_rate in client_acc_rates.items():
        data_dict[f'Client {client_id} Acc'] = client_acc_rate
    df = pd.DataFrame.from_dict(data_dict)
    with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a' if os.path.exists(excel_file_path) else 'w') as writer:
        df.to_excel(writer, sheet_name='Accuracies', index=False)


def read_acc_from_excel(excel_file_name):
    df = pd.read_excel(excel_file_name)
    acc_dict = {}
    for column in df.columns:
        acc_dict[column] = df[column].tolist()
    return acc_dict
