#!/usr/bin/env python3
"""
获取沪深300指数成分股历史数据

功能：
- 获取沪深300成分股列表；
- 抓取每只股票指定时间范围内的历史量价数据；
- 使用 baostock 平台；
- 支持已有 stock_data.csv 的增量更新；
- 登录、成分股查询、单股历史数据查询均带重试；
- 可选择清除代理环境变量，避免 baostock 连接异常；
- 保存格式:
  股票代码, 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌额, 换手率, 涨跌幅
"""

import argparse
import os
import time
from datetime import datetime

import baostock as bs
import pandas as pd


PROXY_ENV_KEYS = [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
]


def parse_args():
    parser = argparse.ArgumentParser(description="下载/增量更新沪深300成分股历史行情数据")
    parser.add_argument("--start-date", type=str, default="2024-01-01", help="开始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default="2026-03-15", help="结束日期，格式 YYYY-MM-DD")
    parser.add_argument("--save-dir", type=str, default="./data", help="保存目录，默认 ./data")
    parser.add_argument("--login-retries", type=int, default=10, help="baostock 登录最大重试次数")
    parser.add_argument("--query-retries", type=int, default=5, help="查询接口最大重试次数")
    parser.add_argument("--retry-sleep", type=float, default=6.0, help="每次失败后的等待秒数")
    parser.add_argument(
        "--clear-proxy",
        action="store_true",
        help="运行前清除 http_proxy/https_proxy 等代理环境变量；baostock 网络异常时建议打开",
    )
    return parser.parse_args()


def clear_proxy_env():
    """清除代理环境变量，避免 baostock 连接被本地代理影响。"""
    removed = []
    for key in PROXY_ENV_KEYS:
        if key in os.environ:
            removed.append((key, os.environ.pop(key)))
    if removed:
        print("已清除代理环境变量:")
        for key, value in removed:
            print(f"  {key}={value}")
    else:
        print("未发现需要清除的代理环境变量")


def validate_date_range(start_date, end_date):
    start_dt = pd.to_datetime(start_date, errors="coerce")
    end_dt = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start_dt) or pd.isna(end_dt):
        raise ValueError(f"日期格式无效: start_date={start_date}, end_date={end_date}")
    if start_dt > end_dt:
        raise ValueError(f"开始日期不能晚于结束日期: {start_date} > {end_date}")


def login(max_retries=10, sleep_seconds=8):
    """登录 baostock，失败自动重试。"""
    last_msg = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"正在登录 baostock... 第 {attempt}/{max_retries} 次")
            lg = bs.login()
            last_msg = getattr(lg, "error_msg", "")

            if getattr(lg, "error_code", None) == "0":
                print("baostock登录成功")
                return lg

            print(f"baostock登录失败: {lg.error_msg}")

        except Exception as e:
            last_msg = str(e)
            print(f"baostock登录异常: {e}")

        if attempt < max_retries:
            print(f"等待 {sleep_seconds} 秒后重试登录...")
            time.sleep(sleep_seconds)

    raise Exception(f"登录失败，已重试 {max_retries} 次，最后错误: {last_msg}")


def logout():
    """登出 baostock。"""
    try:
        bs.logout()
        print("baostock已登出")
    except Exception as e:
        print(f"baostock登出异常，可忽略: {e}")


def relogin(login_retries=3, sleep_seconds=3):
    """尝试重连 baostock。"""
    try:
        bs.logout()
    except Exception:
        pass
    return login(max_retries=login_retries, sleep_seconds=sleep_seconds)


def get_hs300_stocks():
    """获取沪深300成分股列表。"""
    print("正在获取沪深300成分股列表...")

    rs = bs.query_hs300_stocks()
    if rs.error_code != "0":
        raise Exception(f"获取成分股失败: {rs.error_msg}")

    stocks = []
    while (rs.error_code == "0") and rs.next():
        stocks.append(rs.get_row_data())

    df = pd.DataFrame(stocks, columns=rs.fields)
    print(f"获取到 {len(df)} 只沪深300成分股")
    return df


def get_hs300_stocks_with_retry(max_retries=5, sleep_seconds=5):
    """获取沪深300成分股列表，失败自动重试并尝试重连。"""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return get_hs300_stocks()
        except Exception as e:
            last_error = e
            print(f"获取沪深300成分股失败，第 {attempt}/{max_retries} 次: {e}")

            if attempt < max_retries:
                print(f"等待 {sleep_seconds} 秒后重试...")
                time.sleep(sleep_seconds)
                try:
                    relogin(login_retries=3, sleep_seconds=3)
                except Exception as login_error:
                    print(f"重连 baostock 失败: {login_error}")

    raise Exception(f"获取沪深300成分股失败，已重试 {max_retries} 次，最后错误: {last_error}")


def _format_date_series(dt_series):
    """将日期统一格式化为 YYYY/M/D，兼容 Windows/Linux/macOS。"""
    dt_series = pd.to_datetime(dt_series, errors="coerce")
    return (
        dt_series.dt.year.astype(str)
        + "/"
        + dt_series.dt.month.astype(str)
        + "/"
        + dt_series.dt.day.astype(str)
    )


def get_stock_history(bs_code, start_date, end_date):
    """获取单只股票历史数据。"""
    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="1",  # 后复权
    )

    if rs.error_code != "0":
        raise Exception(f"查询失败: {rs.error_msg}")

    data_list = []
    while (rs.error_code == "0") and rs.next():
        data_list.append(rs.get_row_data())

    if not data_list:
        return None

    df = pd.DataFrame(data_list, columns=rs.fields)

    numeric_cols = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["振幅"] = ((df["high"] - df["low"]) / (df["preclose"] + 1e-12) * 100).round(2)
    df["涨跌额"] = (df["close"] - df["preclose"]).round(2)

    df["date"] = _format_date_series(df["date"])

    df["code"] = df["code"].str.replace("sh.", "", regex=False).str.replace("sz.", "", regex=False)
    df["code"] = df["code"].str.zfill(6)

    df = df.rename(
        columns={
            "code": "股票代码",
            "date": "日期",
            "open": "开盘",
            "close": "收盘",
            "high": "最高",
            "low": "最低",
            "volume": "成交量",
            "amount": "成交额",
            "turn": "换手率",
            "pctChg": "涨跌幅",
        }
    )

    columns = [
        "股票代码",
        "日期",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "振幅",
        "涨跌额",
        "换手率",
        "涨跌幅",
    ]
    df = df[columns]
    return df


def get_stock_history_with_retry(
    bs_code,
    start_date,
    end_date,
    max_retries=5,
    sleep_seconds=5,
):
    """单只股票下载失败自动重试，并在失败后尝试重连 baostock。"""
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return get_stock_history(bs_code, start_date, end_date)
        except Exception as e:
            last_error = e
            print(
                f"  第 {attempt}/{max_retries} 次下载失败: "
                f"{bs_code}, {start_date} 至 {end_date}, 错误: {e}"
            )

            if attempt < max_retries:
                print(f"  等待 {sleep_seconds} 秒后重试...")
                time.sleep(sleep_seconds)
                try:
                    relogin(login_retries=3, sleep_seconds=3)
                except Exception as login_error:
                    print(f"  重连 baostock 失败: {login_error}")

    raise Exception(f"{bs_code} 下载失败，已重试 {max_retries} 次，最后错误: {last_error}")


def get_existing_stocks(output_path):
    """获取已经保存的股票代码列表。"""
    if not os.path.exists(output_path):
        return set()
    try:
        df = pd.read_csv(output_path, dtype={"股票代码": str})
        if "股票代码" in df.columns and len(df) > 0:
            return set(df["股票代码"].astype(str).str.zfill(6).unique())
    except Exception:
        pass
    return set()


def get_stock_date_range(output_path, stock_code, start_date=None, end_date=None):
    """获取某只股票在现有数据中的日期范围，可限定目标时间窗。"""
    if not os.path.exists(output_path):
        return None, None

    try:
        df = pd.read_csv(output_path, dtype={"股票代码": str})
        if "股票代码" not in df.columns or "日期" not in df.columns:
            return None, None

        stock_df = df[df["股票代码"].astype(str).str.zfill(6) == stock_code].copy()
        if len(stock_df) == 0:
            return None, None

        stock_df.loc[:, "日期_dt"] = pd.to_datetime(stock_df["日期"], format="%Y/%m/%d", errors="coerce")
        stock_df = stock_df.dropna(subset=["日期_dt"])
        if len(stock_df) == 0:
            return None, None

        if start_date is not None:
            stock_df = stock_df[stock_df["日期_dt"] >= pd.to_datetime(start_date)]
        if end_date is not None:
            stock_df = stock_df[stock_df["日期_dt"] <= pd.to_datetime(end_date)]
        if len(stock_df) == 0:
            return None, None

        return stock_df["日期_dt"].min().strftime("%Y-%m-%d"), stock_df["日期_dt"].max().strftime("%Y-%m-%d")

    except Exception as e:
        print(f"  警告: 读取股票 {stock_code} 现有日期范围失败: {e}")
        return None, None


def filter_data_by_date_range(df, start_date, end_date):
    """过滤 DataFrame，仅保留目标时间窗内的数据。"""
    if df is None or df.empty or "日期" not in df.columns:
        return df

    filtered = df.copy()
    filtered.loc[:, "日期_dt"] = pd.to_datetime(filtered["日期"], format="%Y/%m/%d", errors="coerce")
    filtered = filtered.dropna(subset=["日期_dt"])

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    filtered = filtered[(filtered["日期_dt"] >= start_dt) & (filtered["日期_dt"] <= end_dt)].copy()
    filtered = filtered.drop(columns=["日期_dt"])
    return filtered


def merge_stock_data(existing_df, new_df, stock_code):
    """合并现有数据和新数据，保持同一股票数据相邻。"""
    if new_df is None or new_df.empty:
        return existing_df

    if existing_df is None or existing_df.empty:
        return new_df.copy()

    existing_df = existing_df.copy()
    existing_df["股票代码_str"] = existing_df["股票代码"].astype(str).str.zfill(6)

    other_df = existing_df[existing_df["股票代码_str"] != stock_code].drop(columns=["股票代码_str"])

    if stock_code in existing_df["股票代码_str"].values:
        stock_existing = existing_df[existing_df["股票代码_str"] == stock_code].drop(columns=["股票代码_str"])
    else:
        stock_existing = pd.DataFrame()

    if not stock_existing.empty:
        stock_existing_copy = stock_existing.copy()
        new_df_copy = new_df.copy()

        stock_existing_copy["日期_dt"] = pd.to_datetime(stock_existing_copy["日期"], format="%Y/%m/%d", errors="coerce")
        new_df_copy["日期_dt"] = pd.to_datetime(new_df_copy["日期"], format="%Y/%m/%d", errors="coerce")

        combined = pd.concat([stock_existing_copy, new_df_copy], ignore_index=True)
        combined = combined.dropna(subset=["日期_dt"])
        combined = combined.drop_duplicates(subset=["日期_dt"], keep="last")
        combined = combined.sort_values("日期_dt")
        combined = combined.drop(columns=["日期_dt"])
    else:
        combined = new_df.copy()

    result = pd.concat([other_df, combined], ignore_index=True)
    return result


def add_fetch_range(fetch_ranges, start_date, end_date, label):
    """仅当 start_date <= end_date 时加入需要抓取的区间。"""
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    if start_dt <= end_dt:
        fetch_ranges.append((start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), label))


def main():
    args = parse_args()

    if args.clear_proxy:
        clear_proxy_env()

    start_date = args.start_date
    end_date = args.end_date
    validate_date_range(start_date, end_date)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    output_path = os.path.join(save_dir, "stock_data.csv")

    print(f"目标数据时间范围: {start_date} 至 {end_date}")
    print(f"输出文件: {output_path}")
    print("=" * 60)

    existing_stocks = get_existing_stocks(output_path)
    if existing_stocks:
        print(f"发现已有数据，包含 {len(existing_stocks)} 只股票，将检查每只股票是否需要增量更新")

    login(max_retries=args.login_retries, sleep_seconds=args.retry_sleep)

    try:
        hs300_df = get_hs300_stocks_with_retry(
            max_retries=args.query_retries,
            sleep_seconds=args.retry_sleep,
        )

        hs300_list_path = os.path.join(save_dir, "hs300_stock_list.csv")
        hs300_df.to_csv(hs300_list_path, index=False, encoding="utf-8-sig")

        existing_df = None
        if os.path.exists(output_path) and len(existing_stocks) > 0:
            try:
                existing_df = pd.read_csv(output_path, dtype={"股票代码": str})
                raw_len = len(existing_df)
                existing_df = filter_data_by_date_range(existing_df, start_date, end_date)
                filtered_len = len(existing_df)
                print(f"  已加载现有数据: {len(existing_df)} 条记录")
                if filtered_len != raw_len:
                    print(f"  已按目标区间过滤旧数据: {raw_len} -> {filtered_len}")
            except Exception as e:
                print(f"  警告: 读取现有数据失败: {e}")

        hs300_df["纯代码"] = (
            hs300_df["code"].astype(str).str.replace("sh.", "", regex=False).str.replace("sz.", "", regex=False).str.zfill(6)
        )

        failed_stocks = []
        total = len(hs300_df)
        success_count = 0
        new_stock_count = 0
        incremental_count = 0
        total_new_records = 0

        for idx, row in hs300_df.iterrows():
            bs_code = row.get("code", "")
            stock_name = row.get("code_name", "")
            pure_code = row.get("纯代码", "")

            existing_min_date, existing_max_date = get_stock_date_range(
                output_path,
                pure_code,
                start_date,
                end_date,
            )

            fetch_ranges = []

            if existing_min_date and existing_max_date:
                need_early = existing_min_date > start_date
                need_late = existing_max_date < end_date

                if not need_early and not need_late:
                    print(
                        f"\n[{idx + 1}/{total}] {bs_code} {stock_name} "
                        f"- 数据已完整 ({existing_min_date} 至 {existing_max_date})，跳过"
                    )
                    continue

                print(f"\n[{idx + 1}/{total}] {bs_code} {stock_name} - 增量更新")
                print(f"  现有数据范围: {existing_min_date} 至 {existing_max_date}")

                if need_early:
                    early_end = (pd.to_datetime(existing_min_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    add_fetch_range(fetch_ranges, start_date, early_end, "早期")

                if need_late:
                    late_start = (pd.to_datetime(existing_max_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    fetch_start = max(pd.to_datetime(start_date), pd.to_datetime(late_start)).strftime("%Y-%m-%d")
                    add_fetch_range(fetch_ranges, fetch_start, end_date, "近期")

            else:
                print(f"\n[{idx + 1}/{total}] {bs_code} {stock_name} - 全新获取")
                add_fetch_range(fetch_ranges, start_date, end_date, "全量")

            if not fetch_ranges:
                print("  ✗ 没有有效的待获取区间，跳过")
                continue

            try:
                all_new_data = []

                for fetch_start, fetch_end, period_name in fetch_ranges:
                    print(f"  获取{period_name}数据: {fetch_start} 至 {fetch_end}")
                    stock_data = get_stock_history_with_retry(
                        bs_code,
                        fetch_start,
                        fetch_end,
                        max_retries=args.query_retries,
                        sleep_seconds=args.retry_sleep,
                    )
                    if stock_data is not None and not stock_data.empty:
                        all_new_data.append(stock_data)

                if all_new_data:
                    new_data = pd.concat(all_new_data, ignore_index=True)

                    if existing_df is not None and len(existing_df) > 0:
                        existing_df = merge_stock_data(existing_df, new_data, pure_code)
                        existing_df.to_csv(output_path, index=False, encoding="utf-8-sig")
                        incremental_count += 1
                    else:
                        new_data.to_csv(output_path, index=False, encoding="utf-8-sig")
                        existing_df = new_data
                        new_stock_count += 1

                    total_new_records += len(new_data)
                    success_count += 1
                    print(f"  ✓ 获取成功，新增 {len(new_data)} 条记录")
                else:
                    print("  ✗ 无新数据")

            except Exception as e:
                print(f"  ✗ 失败: {e}")
                failed_stocks.append((bs_code, stock_name))

            if success_count > 0 and success_count % 10 == 0:
                print(f"\n  --- 已成功处理 {success_count} 只，暂停2秒 ---")
                time.sleep(2)

        print("\n" + "=" * 60)
        print("本次运行完成!")
        print(f"  - 全新获取: {new_stock_count} 只股票")
        print(f"  - 增量更新: {incremental_count} 只股票")
        print(f"  - 失败: {len(failed_stocks)} 只股票")
        print(f"  - 新增记录: {total_new_records}")

        if os.path.exists(output_path):
            df = pd.read_csv(output_path, dtype={"股票代码": str})
            print("\n文件总览:")
            print(f"  - 文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
            print(f"  - 总行数: {len(df)}")
            print(f"  - 股票数量: {df['股票代码'].astype(str).str.zfill(6).nunique()}")

            if len(df) > 0:
                df_preview = df.copy()
                df_preview["日期_dt"] = pd.to_datetime(df_preview["日期"], format="%Y/%m/%d", errors="coerce")
                print(f"  - 时间范围: {df_preview['日期_dt'].min().date()} 至 {df_preview['日期_dt'].max().date()}")

                stock_blocks = df.groupby("股票代码").apply(lambda x: x.index.max() - x.index.min() + 1).sum()
                if stock_blocks == len(df):
                    print("  - 数据组织: ✓ 同一股票数据相邻")
                else:
                    print(f"  - 数据组织: 警告，股票数据块总长度({stock_blocks})与总行数({len(df)})不一致")

                print("\n前3行数据预览:")
                print(df.head(3).to_string(index=False))
                print("\n最后3行数据预览:")
                print(df.tail(3).to_string(index=False))

        if failed_stocks:
            failed_df = pd.DataFrame(failed_stocks, columns=["股票代码", "股票名称"])
            failed_path = os.path.join(save_dir, "failed_stocks.csv")
            failed_df.to_csv(failed_path, index=False, encoding="utf-8-sig")
            print(f"\n失败股票列表已保存至: {failed_path}")

    finally:
        logout()


if __name__ == "__main__":
    main()
