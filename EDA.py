import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="EDA Dashboard", layout="wide")
st.title("📊 EDA Dashboard")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

def load_data(file):
    file_name = file.name.lower()

    if file_name.endswith(".csv"):
        data = pd.read_csv(file)
        st.success("CSV file loaded successfully!")
    elif file_name.endswith((".xlsx", ".xls")):
        data = pd.read_excel(file)
        st.success("Excel file loaded successfully!")
    else:
        st.error("Unsupported file format")
        st.stop()

    return data

if uploaded_file is not None:
    try:
        data = load_data(uploaded_file)

        st.subheader("1. Data Preview")
        st.dataframe(data.head())

        st.subheader("2. Dataset Information")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", data.shape[0])
        c2.metric("Columns", data.shape[1])
        c3.metric("Missing Values", int(data.isnull().sum().sum()))

        info_df = pd.DataFrame({
            "Column": data.columns,
            "Data Type": data.dtypes.astype(str),
            "Missing Values": data.isnull().sum().values,
            "Unique Values": data.nunique().values
        })
        st.dataframe(info_df)

        st.subheader("3. Missing Values")
        missing_df = pd.DataFrame({
            "Column": data.columns,
            "Missing Count": data.isnull().sum(),
            "Missing %": (data.isnull().sum() / len(data)) * 100
        }).reset_index(drop=True)

        missing_only = missing_df[missing_df["Missing Count"] > 0]

        if not missing_only.empty:
            st.dataframe(missing_only.sort_values("Missing %", ascending=False))

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(missing_only["Column"], missing_only["Missing %"])
            ax.set_title("Missing Value Percentage by Column")
            ax.set_xlabel("Column")
            ax.set_ylabel("Missing %")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("No missing values found.")

        numeric_data = data.select_dtypes(include=["number"])
        non_numeric_data = data.select_dtypes(include=["bool", "object", "category"])

        st.subheader("4. Numerical Summary")
        if not numeric_data.empty:
            st.dataframe(numeric_data.describe().T)
        else:
            st.info("No numerical features found.")

        st.subheader("5. Non-Numerical Summary")
        if not non_numeric_data.empty:
            st.dataframe(non_numeric_data.describe().T)
        else:
            st.info("No non-numerical features found.")

        st.subheader("6. Duplicate Rows")
        duplicate_count = data.duplicated().sum()
        st.write(f"Duplicate rows: **{duplicate_count}**")

        st.subheader("7. Histograms")
        if not numeric_data.empty:
            hist_col = st.selectbox(
                "Select a numerical column for histogram",
                numeric_data.columns,
                key="hist_col"
            )

            bins = st.slider("Number of bins", 5, 50, 20)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(data[hist_col].dropna(), bins=bins)
            ax.set_title(f"Histogram of {hist_col}")
            ax.set_xlabel(hist_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.info("No numerical columns available for histograms.")

        st.subheader("8. Box Plot")
        if not numeric_data.empty:
            box_col = st.selectbox(
                "Select a numerical column for box plot",
                numeric_data.columns,
                key="box_col"
            )

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.boxplot(data[box_col].dropna(), vert=False)
            ax.set_title(f"Box Plot of {box_col}")
            st.pyplot(fig)
        else:
            st.info("No numerical columns available for box plot.")

        st.subheader("9. Scatter Plot")
        if numeric_data.shape[1] >= 2:
            x_col = st.selectbox("X-axis", numeric_data.columns, key="scatter_x")
            y_col = st.selectbox("Y-axis", numeric_data.columns, key="scatter_y")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(data[x_col], data[y_col])
            ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            st.pyplot(fig)
        else:
            st.info("Need at least 2 numerical columns for scatter plot.")

        st.subheader("10. Line Chart")
        if not numeric_data.empty:
            line_cols = st.multiselect(
                "Select numerical columns for line chart",
                numeric_data.columns,
                default=list(numeric_data.columns[:2])
            )

            if line_cols:
                st.line_chart(data[line_cols])
            else:
                st.info("Select at least one column.")
        else:
            st.info("No numerical columns available for line chart.")

        st.subheader("11. Correlation Heatmap")
        if numeric_data.shape[1] >= 2:
            corr = numeric_data.corr()

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(corr, aspect="auto")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)
            ax.set_title("Correlation Heatmap")
            fig.colorbar(im)
            st.pyplot(fig)

            st.dataframe(corr)
        else:
            st.info("Need at least 2 numerical columns for correlation heatmap.")

        st.subheader("12. Bar Chart for Categorical Features")
        if not non_numeric_data.empty:
            cat_col = st.selectbox(
                "Select a categorical column",
                non_numeric_data.columns,
                key="cat_bar"
            )

            top_n = st.slider("Top categories to show", 3, 20, 10)

            value_counts = data[cat_col].astype(str).value_counts().head(top_n)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(value_counts.index, value_counts.values)
            ax.set_title(f"Top {top_n} Categories in {cat_col}")
            ax.set_xlabel(cat_col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.dataframe(value_counts.reset_index().rename(
                columns={"index": cat_col, cat_col: "Count"}
            ))
        else:
            st.info("No categorical columns available for bar chart.")

        st.subheader("13. Value Counts")
        selected_col = st.selectbox("Select any column to inspect", data.columns, key="inspect")
        value_counts_df = data[selected_col].astype(str).value_counts().reset_index()
        value_counts_df.columns = [selected_col, "Count"]
        st.dataframe(value_counts_df.head(20))

        st.subheader("14. Filter Data")
        filter_col = st.selectbox("Select column to filter", data.columns, key="filter")

        if np.issubdtype(data[filter_col].dtype, np.number):
            min_val = float(data[filter_col].min())
            max_val = float(data[filter_col].max())

            selected_range = st.slider(
                f"Select range for {filter_col}",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )

            filtered_data = data[
                (data[filter_col] >= selected_range[0]) &
                (data[filter_col] <= selected_range[1])
            ]
        else:
            options = data[filter_col].dropna().astype(str).unique().tolist()
            selected_values = st.multiselect(
                f"Select values for {filter_col}",
                options,
                default=options[:5] if len(options) > 5 else options
            )

            if selected_values:
                filtered_data = data[data[filter_col].astype(str).isin(selected_values)]
            else:
                filtered_data = data.copy()

        st.write("Filtered Data Preview")
        st.dataframe(filtered_data.head())

        csv = filtered_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered data as CSV",
            data=csv,
            file_name="filtered_data.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Please upload a CSV or Excel file.")