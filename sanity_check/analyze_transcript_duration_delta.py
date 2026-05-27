import os
import ast
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils_plot import (
    find_overlapping_interval_groups_pair,
    and_operation,
    or_operation,
    get_transcript_with_duration,
)


def build_overlap_groups(df):
    task_dataset_overlap = {}

    for _, row in df.iterrows():
        anots = []
        annotations_A = ast.literal_eval(row['Annotations_A'])
        annotations_B = ast.literal_eval(row['Annotations_B'])
        annotations_C = ast.literal_eval(row['Annotations_C'])

        if len(annotations_A) > 0:
            anots.append(annotations_A)
        if len(annotations_B) > 0:
            anots.append(annotations_B)
        if len(annotations_C) > 0:
            anots.append(annotations_C)

        if len(anots) < 2:
            continue

        intervals1 = anots[0]
        intervals2 = anots[1]

        intervals1 = [
            current_interval for current_interval in intervals1
            if current_interval['timestamp']['start'] is not None
            and current_interval['timestamp']['end'] is not None
        ]
        intervals2 = [
            current_interval for current_interval in intervals2
            if current_interval['timestamp']['start'] is not None
            and current_interval['timestamp']['end'] is not None
        ]

        overlapping_groups = find_overlapping_interval_groups_pair(intervals1, intervals2)

        k = row['file_name']
        task_dataset_overlap[k] = {}

        count = 0
        for sublist in overlapping_groups:
            task_dataset_overlap[k][count] = [{**sub, 'video_id': k} for sub in sublist]
            count += 1

    return task_dataset_overlap


def build_agreed_overlap(task_dataset_overlap, task_type):
    task_dataset_agreed_overlap = {}

    for k, v in task_dataset_overlap.items():
        count = 0
        for _, group_dict in v.items():
            if len(group_dict) == 1:
                continue

            if len(group_dict) != 2:
                continue

            agree = False
            agreed_dict = None

            if task_type in ['debug', 'detection', 'rationale', 'correction', 'context']:
                agree = group_dict[0]['error'] == group_dict[1]['error']

            if task_type in ['detection_error_only']:
                agree = group_dict[0]['error'] == group_dict[1]['error']
                if agree:
                    if group_dict[0]['error'] is True and group_dict[1]['error'] is True:
                        agreed_dict = or_operation(group_dict[0]['attribute'], group_dict[1]['attribute'])
                        agree = True
                    else:
                        agree = False

            if task_type in ['attribute', 'attribute_disagree', 'attribute_agreed_multiple', 'attribute_agreed_multiple_subj']:
                agree = group_dict[0]['error'] == group_dict[1]['error']
                if agree:
                    if task_type == 'attribute':
                        agreed_dict = or_operation(group_dict[0]['attribute'], group_dict[1]['attribute'])

                    if task_type == 'attribute_agreed_multiple':
                        agreed_dict = and_operation(group_dict[0]['attribute'], group_dict[1]['attribute'])
                        if sum(list(agreed_dict.values())) >= 2:
                            agree = True
                        else:
                            agree = False

                    if task_type == 'attribute_disagree':
                        agreed_dict_or = or_operation(group_dict[0]['attribute'], group_dict[1]['attribute'])
                        agreed_dict_and = and_operation(group_dict[0]['attribute'], group_dict[1]['attribute'])
                        intersect_dict = and_operation(agreed_dict_or, agreed_dict_and)
                        if sum(list(intersect_dict.values())) == 0:
                            agree = True
                            agreed_dict = agreed_dict_or
                        else:
                            agree = False

            if agree:
                if task_type in ['attribute', 'attribute_disagree', 'attribute_agreed_multiple', 'detection_error_only']:
                    group_dict[0]['tier2'] = agreed_dict
                    group_dict[1]['tier2'] = agreed_dict

                if not task_dataset_agreed_overlap.get(k):
                    task_dataset_agreed_overlap[k] = {}
                task_dataset_agreed_overlap[k][count] = group_dict
                count += 1

    return task_dataset_agreed_overlap


def select_source_groups(task_dataset_overlap, task_dataset_agreed_overlap, task_type):
    if task_type in [
        'debug',
        'detection',
        'detection_error_only',
        'attribute',
        'attribute_disagree',
        'attribute_agreed_multiple',
        'attribute_agreed_multiple_subj',
    ]:
        return task_dataset_agreed_overlap

    if task_type in ['rationale', 'context', 'correction', 'pre', 'post']:
        return task_dataset_overlap

    return task_dataset_agreed_overlap


def should_keep_sample(sample, task_type):
    if task_type == 'rationale':
        return len(sample.get('rationale', '')) > 0

    if task_type == 'correction':
        return sample.get('error') is True and len(sample.get('correction', '')) > 0

    if task_type == 'context':
        return True

    if task_type in ['pre', 'post']:
        return sample.get('error') is False

    return True


def collect_duration_delta_rows(df, task_type):
    task_dataset_overlap = build_overlap_groups(df)
    task_dataset_agreed_overlap = build_agreed_overlap(task_dataset_overlap, task_type)
    source_groups = select_source_groups(task_dataset_overlap, task_dataset_agreed_overlap, task_type)

    rows = []

    for video_id, groups in tqdm(source_groups.items()):
        transcriptions = df[df['file_name'] == video_id]['transcript'].item()

        for group_id, group_dict in groups.items():
            sample = group_dict[0]

            if not should_keep_sample(sample, task_type):
                continue

            timestamp_dict = sample['timestamp']
            transcript_info = get_transcript_with_duration(timestamp_dict, transcriptions)

            if transcript_info['transcript_duration'] is None:
                continue

            conversation = transcript_info['conversation']

            if task_type in ['pre', 'post']:
                split_convo = conversation.split("\n")
                if not (len(split_convo) == 2 and 'User' in split_convo[0] and 'Agent' in split_convo[1]):
                    continue

            interval_start = timestamp_dict['start']
            interval_end = timestamp_dict['end']
            interval_duration = interval_end - interval_start

            transcript_start = transcript_info['transcript_start']
            transcript_end = transcript_info['transcript_end']
            transcript_duration = transcript_info['transcript_duration']

            delta_duration = transcript_duration - interval_duration

            row = {
                'video_id': video_id,
                'group_id': group_id,
                'task_type': task_type,
                'error': sample.get('error'),
                'interval_start': interval_start,
                'interval_end': interval_end,
                'interval_duration': interval_duration,
                'transcript_start': transcript_start,
                'transcript_end': transcript_end,
                'transcript_duration': transcript_duration,
                'delta_duration': delta_duration,
                'transcript': conversation,
            }

            if task_type in ['pre', 'post']:
                split_convo = conversation.split("\n")
                row['transcript_user'] = split_convo[0]
                row['transcript_agent'] = split_convo[1]

            rows.append(row)

    return pd.DataFrame(rows)


def save_scatter_plot(result_df, out_path, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(
        result_df['interval_duration'],
        result_df['delta_duration'],
        alpha=0.6,
        s=20
    )
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Groundtruth interval duration (sec)')
    plt.ylabel('Delta duration = transcript duration - interval duration (sec)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_bin_summary(result_df, bin_size=10):
    if len(result_df) == 0:
        return pd.DataFrame(columns=[
            'bin_start',
            'bin_end',
            'bin_label',
            'count',
        ])

    min_val = math.floor(result_df['delta_duration'].min() / bin_size) * bin_size
    max_val = math.ceil(result_df['delta_duration'].max() / bin_size) * bin_size

    if min_val == max_val:
        max_val = min_val + bin_size

    bins = list(range(int(min_val), int(max_val) + bin_size, bin_size))

    binned = pd.cut(
        result_df['delta_duration'],
        bins=bins,
        right=False,
        include_lowest=True
    )

    summary = (
        result_df.assign(delta_bin=binned)
        .groupby('delta_bin')
        .size()
        .reset_index(name='count')
    )

    summary['bin_start'] = summary['delta_bin'].apply(lambda x: x.left)
    summary['bin_end'] = summary['delta_bin'].apply(lambda x: x.right)
    summary['bin_label'] = summary['delta_bin'].apply(lambda x: f"[{int(x.left)}, {int(x.right)})")

    summary = summary[['bin_start', 'bin_end', 'bin_label', 'count']]
    summary = summary.sort_values('bin_start').reset_index(drop=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze transcript vs annotation duration deltas")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--plot_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.plot_path, exist_ok=True)
    os.makedirs(args.csv_path, exist_ok=True)

    df = pd.read_csv(args.data_path)
    result_df = collect_duration_delta_rows(df, args.task_type)

    base_name = f"{args.data_name}_{args.task_type}"

    raw_csv_path = os.path.join(args.csv_path, f"{base_name}_delta_duration.csv")
    bin_csv_path = os.path.join(args.csv_path, f"{base_name}_delta_duration_bins_10s.csv")
    plot_file_path = os.path.join(args.plot_path, f"{base_name}_delta_duration_scatter.png")

    result_df.to_csv(raw_csv_path, index=False)

    bin_summary_df = build_bin_summary(result_df, bin_size=10)
    bin_summary_df.to_csv(bin_csv_path, index=False)

    if len(result_df) > 0:
        save_scatter_plot(
            result_df,
            plot_file_path,
            f"{args.data_name} | {args.task_type} | transcript vs annotation duration delta"
        )
        print(f"Saved plot: {plot_file_path}")
    else:
        print("No valid samples found; skipping plot.")

    print(f"Saved raw csv: {raw_csv_path}")
    print(f"Saved binned csv: {bin_csv_path}")


if __name__ == "__main__":
    main()