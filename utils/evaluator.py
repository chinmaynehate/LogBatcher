import os
import scipy.special
import pandas as pd
from nltk.metrics.distance import edit_distance
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

def evaluate(output_file, groundtruth_file, dataset, mismatch=False):

    df1 = pd.read_csv(output_file)
    df2 = pd.read_csv(groundtruth_file)

    # Remove invalid groundtruth event Ids
    null_logids = df2[~df2['EventTemplate'].isnull()].index
    df1 = df1.loc[null_logids]
    df2 = df2.loc[null_logids]


    accuracy_exact_string_matching = accuracy_score(np.array(df1['EventTemplate'].values, dtype='str'),np.array(df2['EventTemplate'], dtype='str'))
    print("MLA: ", accuracy_exact_string_matching)
    # find the mismatch values
    if mismatch:
        head,_,_ = output_file.rpartition('/')
        os.makedirs(f'{head}/mismatch', exist_ok=True)
        df_mismatch = df2[df1.EventTemplate != df2.EventTemplate]
        df_mismatch.to_csv(f'{head}/mismatch/{dataset}.csv', index=False)


    edit_distance_result = []
    normalized_ed_result = []
    for i, j in tqdm(zip(np.array(df1.EventTemplate.values, dtype='str'),
                    np.array(df2.EventTemplate.values, dtype='str'))):
        if i == j:
            pass
        else:
            ed = edit_distance(i, j)
            normalized_ed = 1 - ed / max(len(i), len(j))
            edit_distance_result.append(ed)
            normalized_ed_result.append(normalized_ed)

    # edit_distance_result_mean = np.mean(edit_distance_result)
    # edit_distance_result_std = np.std(edit_distance_result)
    # normalized_ed_result_mean = np.mean(normalized_ed_result)

    length_logs = len(df1['EventTemplate'].values)
    edit_distance_result_mean = sum(edit_distance_result) / length_logs
    normalized_ed_result_mean = (sum(normalized_ed_result)+ length_logs - len(normalized_ed_result)) / length_logs

    print("ED: ", edit_distance_result_mean)
    print("NED: ", normalized_ed_result_mean)
    

    (precision, recall, f_measure, accuracy_PA) = get_accuracy(df1['EventTemplate'],
                                                               df2['EventTemplate'])
    print("PA: ", accuracy_PA)

    # print(
    #     'Precision: %.4f, Recall: %.4f, F1_measure: %.4f, Group Accuracy: %.4f, Message-Level Accuracy: %.4f, Edit Distance: %.4f' % (
    #         precision, recall, f_measure, accuracy_PA, accuracy_exact_string_matching, edit_distance_result_mean))
    dataset = ' ' * (12 - len(dataset)) + dataset 
    print('%s: group Accuracy: %.4f, Message-Level Accuracy: %.4f, Edit Distance: %.4f, Normalized Edit Distance: %.6f' % (dataset, accuracy_PA, accuracy_exact_string_matching, edit_distance_result_mean, normalized_ed_result_mean))
    return accuracy_PA, accuracy_exact_string_matching, edit_distance_result_mean, normalized_ed_result_mean



def get_accuracy(series_groundtruth, series_parsedlog, debug=False):

    # series_groundtruth_valuecounts = series_groundtruth.value_counts()
    # real_pairs = 0
    # for count in series_groundtruth_valuecounts:
    #     if count > 1:
    #         real_pairs += scipy.special.comb(count, 2)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    # parsed_pairs = 0
    # for count in series_parsedlog_valuecounts:
    #     if count > 1:
    #         parsed_pairs += scipy.special.comb(count, 2)

    # accurate_pairs = 0
    accurate_events = 0  # determine how many lines are correctly parsed
    for parsed_eventId in tqdm(series_parsedlog_valuecounts.index):
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts(
        )
        # error_eventIds = (
        #     parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
        # error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                error = False
        # if error and debug:
        #     print('(parsed_eventId, groundtruth_eventId) =',
        #           error_eventIds, 'failed', logIds.size, 'messages')
        # for count in series_groundtruth_logId_valuecounts:
        #     if count > 1:
        #         accurate_pairs += scipy.special.comb(count, 2)

    # precision = float(accurate_pairs) / parsed_pairs
    # recall = float(accurate_pairs) / real_pairs
    # f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_groundtruth.size
    # return precision, recall, f_measure, accuracy
    return 0, 0, 0, accuracy