import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="将股票数据切分为 train.csv 和 test.csv；默认使用最后5个交易日作为测试集"
	)
	parser.add_argument(
		"--input",
		type=str,
		default="data/stock_data.csv",
		help="原始数据文件路径，默认 data/stock_data.csv",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="data",
		help="输出目录，默认 data",
	)
	parser.add_argument(
		"--train-start",
		type=str,
		default=None,
		help="训练集开始日期；若提供手动日期模式，需要同时提供 train-end/test-start/test-end",
	)
	parser.add_argument(
		"--train-end",
		type=str,
		default=None,
		help="训练集结束日期；若提供手动日期模式，需要同时提供 train-start/test-start/test-end",
	)
	parser.add_argument(
		"--test-start",
		type=str,
		default=None,
		help="测试集开始日期；若提供手动日期模式，需要同时提供 train-start/train-end/test-end",
	)
	parser.add_argument(
		"--test-end",
		type=str,
		default=None,
		help="测试集结束日期；若提供手动日期模式，需要同时提供 train-start/train-end/test-start",
	)
	parser.add_argument(
		"--test-days",
		type=int,
		default=5,
		help="自动切分模式下，测试集使用最后几个交易日，默认 5",
	)
	return parser.parse_args()


def _to_timestamp(date_str: str, name: str) -> pd.Timestamp:
	ts = pd.to_datetime(date_str, errors="coerce")
	if pd.isna(ts):
		raise ValueError(f"参数 {name} 的日期格式无效: {date_str}")
	return ts.normalize()


def _validate_columns(df: pd.DataFrame) -> None:
	required = {"股票代码", "日期"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"输入文件缺少必要列: {sorted(missing)}")


def _filter_by_date(
	df: pd.DataFrame,
	start_date: pd.Timestamp,
	end_date: pd.Timestamp,
) -> pd.DataFrame:
	if start_date > end_date:
		raise ValueError(f"开始日期晚于结束日期: {start_date.date()} > {end_date.date()}")

	mask = (df["日期"] >= start_date) & (df["日期"] <= end_date)
	out = df.loc[mask].copy()
	out = out.sort_values(["股票代码", "日期"]).reset_index(drop=True)
	out["日期"] = out["日期"].dt.strftime("%Y-%m-%d")
	return out


def _filter_by_dates(df: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
	out = df[df["日期"].isin(dates)].copy()
	out = out.sort_values(["股票代码", "日期"]).reset_index(drop=True)
	out["日期"] = out["日期"].dt.strftime("%Y-%m-%d")
	return out


def main() -> None:
	args = parse_args()

	input_path = Path(args.input)
	if not input_path.exists():
		raise FileNotFoundError(f"输入文件不存在: {input_path}")

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(input_path, dtype={"股票代码": str})
	_validate_columns(df)

	df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
	df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
	if df["日期"].isna().any():
		bad_rows = int(df["日期"].isna().sum())
		raise ValueError(f"原始数据中存在无法解析的日期，共 {bad_rows} 行")

	df["日期"] = df["日期"].dt.normalize()
	source_min_date = df["日期"].min().date()
	source_max_date = df["日期"].max().date()

	manual_args = [args.train_start, args.train_end, args.test_start, args.test_end]
	has_any_manual = any(x is not None for x in manual_args)
	has_all_manual = all(x is not None for x in manual_args)

	if has_any_manual and not has_all_manual:
		raise ValueError(
			"若使用手动日期模式，必须同时提供 --train-start --train-end --test-start --test-end"
		)

	if has_all_manual:
		train_start = _to_timestamp(args.train_start, "--train-start")
		train_end = _to_timestamp(args.train_end, "--train-end")
		test_start = _to_timestamp(args.test_start, "--test-start")
		test_end = _to_timestamp(args.test_end, "--test-end")

		if train_end >= test_start:
			raise ValueError(
				f"训练集与测试集日期区间重叠或相接不合法: "
				f"train_end={train_end.date()} >= test_start={test_start.date()}"
			)

		train_df = _filter_by_date(df, train_start, train_end)
		test_df = _filter_by_date(df, test_start, test_end)

		print("切分模式: 手动日期区间")
		print(
			f"训练集日期范围: {train_start.date()} ~ {train_end.date()} | "
			f"测试集日期范围: {test_start.date()} ~ {test_end.date()}"
		)
	else:
		if args.test_days <= 0:
			raise ValueError(f"--test-days 必须为正整数，当前为 {args.test_days}")

		trade_dates = pd.Index(sorted(df["日期"].dropna().unique()))
		if len(trade_dates) <= args.test_days:
			raise ValueError(
				f"交易日数量不足，无法切分。当前共有 {len(trade_dates)} 个交易日，"
				f"但 test_days={args.test_days}，至少需要大于 test_days。"
			)

		test_dates = trade_dates[-args.test_days:]
		train_dates = trade_dates[:-args.test_days]

		train_df = _filter_by_dates(df, train_dates)
		test_df = _filter_by_dates(df, test_dates)

		print("切分模式: 自动按最后 N 个交易日切分")
		print(
			f"训练集日期范围: {pd.Timestamp(train_dates.min()).date()} ~ {pd.Timestamp(train_dates.max()).date()} | "
			f"测试集日期范围: {pd.Timestamp(test_dates.min()).date()} ~ {pd.Timestamp(test_dates.max()).date()} | "
			f"测试集交易日数: {args.test_days}"
		)

	train_path = output_dir / "train.csv"
	test_path = output_dir / "test.csv"

	train_df.to_csv(train_path, index=False)
	test_df.to_csv(test_path, index=False)

	print(f"训练集: {train_path}，共 {len(train_df)} 行，股票数 {train_df['股票代码'].nunique()}")
	print(f"测试集: {test_path}，共 {len(test_df)} 行，股票数 {test_df['股票代码'].nunique()}")

	if train_df.empty or test_df.empty:
		print("警告: 训练集或测试集为空，请检查日期范围是否与原始数据重叠。")
		print(f"原始数据日期范围: {source_min_date} ~ {source_max_date}")


if __name__ == "__main__":
	main()