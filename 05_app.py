"""銅箔製程異常監控 Streamlit 應用程式。

啟動方式：streamlit run 05_app.py
"""

from datetime import date, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# 以程式所在目錄作為根目錄，方便本機與 Streamlit Cloud 部署。
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "random_forest_anomaly_model.joblib"
CLEANED_DATA_PATH = PROJECT_DIR / "copper_foil_1000_cleaned.csv"
TRANSFORMED_DATA_PATH = PROJECT_DIR / "copper_foil_1000_transformed.csv"

st.set_page_config(
    page_title="銅箔製程異常監控",
    page_icon=":material/monitor_heart:",
    layout="wide",
)


@st.cache_resource
def load_model_bundle(path: str) -> dict:
    """快取模型，避免每次 widget rerun 都重新讀取 joblib。"""
    return joblib.load(path)


@st.cache_data
def load_cleaned_data(path: str) -> pd.DataFrame:
    """載入視覺化資料並解析時間欄位。"""
    data = pd.read_csv(path)
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    return data


@st.cache_data
def load_transformed_data(path: str) -> pd.DataFrame:
    """載入轉換資料，作為表單特徵轉換的參考分布。"""
    return pd.read_csv(path)


def build_model_features(
    values: dict,
    timestamp: pd.Timestamp,
    shift: str,
    feature_columns: list[str],
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """重現訓練 notebook 的特徵工程，輸出模型需要的 40 個欄位。"""
    row = dict(values)
    row["Year"] = timestamp.year
    row["Month"] = timestamp.month
    row["Day"] = timestamp.day
    row["Hour"] = timestamp.hour
    row["DayOfWeek"] = timestamp.dayofweek
    row["Is_Weekend"] = int(timestamp.dayofweek >= 5)
    row["Hour_sin"] = np.sin(2 * np.pi * timestamp.hour / 24)
    row["Hour_cos"] = np.cos(2 * np.pi * timestamp.hour / 24)

    thickness = [row["Thick_Left_um"], row["Thick_Center_um"], row["Thick_Right_um"]]
    row["Thick_Mean_um"] = float(np.mean(thickness))
    row["Thick_Range_um"] = float(np.max(thickness) - np.min(thickness))
    row["Thick_Max_um"] = float(np.max(thickness))
    row["Thick_Min_um"] = float(np.min(thickness))
    row["Thick_Deviation_from_Target_um"] = row["Thick_Mean_um"] - 18.0
    row["Thick_CV"] = row["Thick_Range_um"] / abs(row["Thick_Mean_um"])

    # 依模型保存的 Shift_* 欄位建立 one-hot 特徵。
    for column in (c for c in feature_columns if c.startswith("Shift_")):
        row[column] = int(column == f"Shift_{shift}")

    # 03 notebook 使用 StandardScaler 的 population standard deviation（ddof=0）。
    for column in feature_columns:
        if column.endswith("_z"):
            base_column = column[:-2]
            mean = reference[base_column].mean()
            std = reference[base_column].std(ddof=0)
            row[column] = 0.0 if std == 0 else (row[base_column] - mean) / std

    features = pd.DataFrame([row])
    missing = [c for c in feature_columns if c not in features.columns]
    if missing:
        raise ValueError(f"模型特徵轉換缺少欄位：{missing}")
    return features.loc[:, feature_columns].astype(float)


def render_visualization_page(data: pd.DataFrame) -> None:
    """第 1 頁：資料概況與主要製程特徵視覺化。"""
    st.title("資料視覺化")
    st.caption("探索銅箔製程參數、厚度量測與異常標記的分布。")

    with st.sidebar:
        st.subheader("視覺化篩選")
        shifts = sorted(data["Shift"].dropna().unique().tolist())
        selected_shifts = st.multiselect("班別", shifts, default=shifts)
        plot_features = st.multiselect(
            "時間序列特徵",
            [
                "Line_Speed_mpm", "Current_ASD", "Bath_Temp_C", "Cu_Conc_gL",
                "Additive_ppm", "Jet_Pressure_bar", "Thick_Mean_um", "Thick_Range_um",
            ],
            default=["Thick_Mean_um", "Thick_Range_um"],
        )

    filtered = data[data["Shift"].isin(selected_shifts)].copy()
    if filtered.empty:
        st.warning("目前篩選條件沒有資料。", icon=":material/warning:")
        return

    with st.container(horizontal=True):
        st.metric("樣本數", f"{len(filtered):,}", border=True)
        st.metric("異常率", f"{filtered['Is_Anomaly'].mean():.2%}", border=True)
        st.metric("超規筆數", f"{filtered['Is_Out_of_Spec'].sum():,}", border=True)
        st.metric("平均厚度", f"{filtered['Thick_Mean_um'].mean():.3f} μm", border=True)

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("製程特徵時間序列")
            if plot_features:
                chart_data = filtered.sort_values("Timestamp").set_index("Timestamp")[plot_features]
                st.line_chart(chart_data, alt="Selected process features over time")
            else:
                st.info("請在左側至少選擇一個時間序列特徵。")

    with right:
        with st.container(border=True):
            st.subheader("主要特徵分布")
            distribution_feature = st.selectbox(
                "選擇特徵",
                [
                    "Thick_Mean_um", "Thick_Range_um", "Line_Speed_mpm", "Current_ASD",
                    "Bath_Temp_C", "Cu_Conc_gL", "Additive_ppm", "Jet_Pressure_bar",
                ],
                key="distribution_feature",
            )
            bins = pd.cut(filtered[distribution_feature], bins=12)
            distribution = bins.value_counts().sort_index().rename("count").to_frame()
            distribution.index = distribution.index.astype(str)
            st.bar_chart(distribution, y="count", alt=f"Distribution of {distribution_feature}")

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("厚度均值與厚度範圍")
            st.scatter_chart(
                filtered,
                x="Thick_Mean_um",
                y="Thick_Range_um",
                color="Is_Anomaly",
                alt="Thickness mean versus thickness range colored by anomaly flag",
            )

    with right:
        with st.container(border=True):
            st.subheader("各班別異常率")
            shift_summary = (
                filtered.groupby("Shift", as_index=True)["Is_Anomaly"]
                .mean().mul(100).rename("anomaly_rate_percent").to_frame()
            )
            st.bar_chart(shift_summary, y="anomaly_rate_percent", alt="Anomaly rate by shift")

    with st.container(border=True):
        st.subheader("數值特徵摘要")
        summary_cols = [
            "Line_Speed_mpm", "Current_ASD", "Bath_Temp_C", "Cu_Conc_gL",
            "Additive_ppm", "Jet_Pressure_bar", "Thick_Mean_um", "Thick_Range_um",
        ]
        st.dataframe(filtered[summary_cols].describe().T.round(3), hide_index=False)


def render_prediction_page(bundle: dict, reference: pd.DataFrame) -> None:
    """第 2 頁：原始製程欄位表單與異常預測。"""
    st.title("異常預測")
    st.caption("輸入一筆製程資料，模型會回傳異常機率與分類結果。")

    feature_columns = bundle["feature_columns"]
    model = bundle["model"]
    shift_columns = [c for c in feature_columns if c.startswith("Shift_")]
    shift_options = [c.removeprefix("Shift_") for c in shift_columns]
    defaults = reference.median(numeric_only=True)

    # 以訓練資料 P5-P95 作為「參考正常範圍」；這是資料分布提示，不是硬性規格。
    reference_ranges = {
        column: (reference[column].quantile(0.05), reference[column].quantile(0.95))
        for column in [
            "Line_Speed_mpm", "Current_ASD", "Bath_Temp_C", "Cu_Conc_gL",
            "Additive_ppm", "Jet_Pressure_bar", "Edge_Current_Bias",
            "Thick_Left_um", "Thick_Center_um", "Thick_Right_um",
        ]
    }
    # 厚度規格沿用資料生成流程中的 17-19 μm 目標規格，另提供範圍變異的參考值。
    reference_ranges.update({
        "Thick_Left_um": (17.0, 19.0),
        "Thick_Center_um": (17.0, 19.0),
        "Thick_Right_um": (17.0, 19.0),
    })

    def range_text(column: str) -> str:
        lower, upper = reference_ranges[column]
        return f"{lower:.2f}–{upper:.2f}"

    with st.form("prediction_form", border=True):
        st.subheader("製程輸入")
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_date = st.date_input("量測日期", value=date(2026, 10, 1))
            selected_time = st.time_input("量測時間", value=time(8, 0))
            shift = st.selectbox("班別", shift_options, index=0)
            line_speed = st.number_input(
                f"線速 (mpm)｜正常範圍 {range_text('Line_Speed_mpm')}",
                value=float(defaults["Line_Speed_mpm"]), step=0.01,
                help="參考正常範圍為訓練資料的 P5-P95；超出範圍仍可送出並交由模型判斷。",
            )
            current_asd = st.number_input(
                f"電流 ASD｜正常範圍 {range_text('Current_ASD')}",
                value=float(defaults["Current_ASD"]), step=0.01,
                help="參考正常範圍為訓練資料的 P5-P95。",
            )
        with col2:
            bath_temp = st.number_input(
                f"槽液溫度 (°C)｜正常範圍 {range_text('Bath_Temp_C')}",
                value=float(defaults["Bath_Temp_C"]), step=0.01,
                help="參考正常範圍為訓練資料的 P5-P95。",
            )
            cu_conc = st.number_input(
                f"銅濃度 (g/L)｜正常範圍 {range_text('Cu_Conc_gL')}",
                value=float(defaults["Cu_Conc_gL"]), step=0.01,
                help="參考正常範圍為訓練資料的 P5-P95。",
            )
            additive = st.number_input(
                f"添加劑 (ppm)｜正常範圍 {range_text('Additive_ppm')}",
                value=float(defaults["Additive_ppm"]), step=0.01,
                help="參考正常範圍為訓練資料的 P5-P95。",
            )
            jet_pressure = st.number_input(
                f"噴流壓力 (bar)｜正常範圍 {range_text('Jet_Pressure_bar')}",
                value=float(defaults["Jet_Pressure_bar"]), step=0.01,
                help="參考正常範圍為訓練資料的 P5-P95。",
            )
            edge_bias = st.number_input(
                f"邊緣電流偏差｜正常範圍 {range_text('Edge_Current_Bias')}",
                value=float(defaults["Edge_Current_Bias"]), step=0.01,
                help="參考正常範圍為訓練資料的 P5-P95。",
            )
        with col3:
            thick_left = st.number_input(
                f"左側厚度 (μm)｜正常範圍 {range_text('Thick_Left_um')}",
                value=float(defaults["Thick_Left_um"]), step=0.01,
                help="參考厚度規格為 17.00–19.00 μm。",
            )
            thick_center = st.number_input(
                f"中央厚度 (μm)｜正常範圍 {range_text('Thick_Center_um')}",
                value=float(defaults["Thick_Center_um"]), step=0.01,
                help="參考厚度規格為 17.00–19.00 μm。",
            )
            thick_right = st.number_input(
                f"右側厚度 (μm)｜正常範圍 {range_text('Thick_Right_um')}",
                value=float(defaults["Thick_Right_um"]), step=0.01,
                help="參考厚度規格為 17.00–19.00 μm。",
            )
            st.info("模型會自動計算厚度均值、範圍、時間及標準化特徵。", icon=":material/info:")
        submitted = st.form_submit_button(
            "執行異常預測", type="primary", icon=":material/search:", width="stretch"
        )

    if not submitted:
        st.info("填寫表單後按下「執行異常預測」。", icon=":material/lightbulb:")
        return

    timestamp = pd.Timestamp.combine(selected_date, selected_time)
    values = {
        "Line_Speed_mpm": line_speed, "Current_ASD": current_asd,
        "Bath_Temp_C": bath_temp, "Cu_Conc_gL": cu_conc,
        "Additive_ppm": additive, "Jet_Pressure_bar": jet_pressure,
        "Edge_Current_Bias": edge_bias, "Thick_Left_um": thick_left,
        "Thick_Center_um": thick_center, "Thick_Right_um": thick_right,
    }
    try:
        model_features = build_model_features(values, timestamp, shift, feature_columns, reference)
        probability = float(model.predict_proba(model_features)[0, 1])
        prediction = int(model.predict(model_features)[0])
    except (KeyError, ValueError) as error:
        st.error(f"資料轉換失敗：{error}", icon=":material/error:")
        return

    st.divider()
    if prediction == 1:
        st.error("預測結果：異常", icon=":material/warning:")
    else:
        st.success("預測結果：正常", icon=":material/check_circle:")
    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("異常機率", f"{probability:.2%}", border=True)
    result_col2.metric("分類門檻", "50%", border=True)
    result_col3.metric("模型", "Random Forest", border=True)
    with st.container(border=True):
        st.subheader("模型實際收到的特徵")
        st.dataframe(model_features.T.rename(columns={0: "value"}), hide_index=False)


# 左側功能分頁選擇器。
with st.sidebar:
    st.title("銅箔製程監控")
    st.caption("Random Forest anomaly detection")
    selected_page = st.selectbox("功能分頁", ["資料視覺化", "異常預測"], key="selected_page")

if not MODEL_PATH.exists():
    st.error(f"找不到模型檔案：{MODEL_PATH}", icon=":material/error:")
    st.stop()
if not CLEANED_DATA_PATH.exists() or not TRANSFORMED_DATA_PATH.exists():
    st.error("找不到 cleaned 或 transformed CSV，請確認檔案在專案目錄。", icon=":material/error:")
    st.stop()

model_bundle = load_model_bundle(str(MODEL_PATH))
cleaned_data = load_cleaned_data(str(CLEANED_DATA_PATH))
transformed_data = load_transformed_data(str(TRANSFORMED_DATA_PATH))

if selected_page == "資料視覺化":
    render_visualization_page(cleaned_data)
else:
    render_prediction_page(model_bundle, transformed_data)
