import argparse
import pandas as pd
from collect_results_across_percents import name_map


def make_table(args):
    df = pd.read_csv(args.results_path, header=0, index_col=0)
    df["Type"] = df["Type"].apply(lambda x: name_map[x])
    df.Score = df.Score.round(3) * 100
    df = df[df.percent.isin([1, 2, 3, 4, 5, 10, 20, 40, 50, 100])]
    for data_type, group_df, in df.groupby("\nData Type"):
        print(data_type)
        table_df = group_df.pivot_table("Score", ["\nData Type", "Type"], "percent").reset_index( drop=False)
        table_df["\nData Type"] = [""] * len(table_df)
        print(table_df.to_latex(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results_path', help='path to file containing results from FiD predictions', type=str, required=True)
    args = parser.parse_args()
    make_table(args)

    # python plotting/make_table_of_percents.py -r results_nq_test/ATLAS/article/data.csv
