import pandas as pd

file_names = ['final - log_cifar_stl_run']

for filename in file_names:
    with open(f'./colab_results_sup/{filename}.txt', 'r') as f:
        res_list = []
        res_dict = {}
        Lines = f.readlines()
        for line in Lines:
            if line.startswith('outer CV'):
                res_dict['params'] = line[line.rindex('{'):]
            elif line.startswith('tgt_scores_dict'):
                res_dict.update(eval(line[line.rindex('{'):]))
            elif line.startswith('training_time'):
                res_dict['training_time'] = float(line[line.rindex(':') + 2:-2])
            elif line.startswith('inference'):
                res_dict['inference_time'] = float(line[line.rindex(':') + 2:-2])
                res_list.append(res_dict)
                res_dict = {}

    df = pd.DataFrame(res_list)
    df = df.round(4)
    df['acc'] = df['acc'].round(2)

    df.to_csv(f'./colab_results_sup/{filename}.csv')