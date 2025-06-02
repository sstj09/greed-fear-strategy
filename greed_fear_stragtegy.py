import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(page_title="贪婪恐惧与比特币分析", layout="wide")
st.title("📊 贪婪恐惧指数与比特币价格关系分析")

# 数据加载函数
@st.cache_data
def load_data():
    try:
        fng_df = pd.read_excel("2.Fear_and_Greed_Index.xlsx")
        fng_df = fng_df[['timestamp', 'value']].rename(columns={'value': 'greed'})
        fng_df['date'] = pd.to_datetime(fng_df['timestamp'])
        fng_df['greed'] = fng_df['greed'].astype(int)
        fng_df = fng_df.sort_values('date')

        btc_df = pd.read_csv("3.Bitcoin_2024_6_1-2025_6_1_historical_data_coinmarketcap.csv")
        btc_df = btc_df[['date', 'close']].rename(columns={'close': 'price'})
        btc_df['date'] = pd.to_datetime(btc_df['date'])
        btc_df['price'] = btc_df['price'].astype(str).str.replace(",", "").astype(float)

        df = pd.merge(fng_df, btc_df, on='date', how='inner')
        df = df[['date', 'greed', 'price']].sort_values('date').dropna().reset_index(drop=True)

        df['greed_level'] = pd.cut(df['greed'], 
                                   bins=[0, 24, 49, 74, 100],
                                   labels=['极度恐惧', '恐惧', '贪婪', '极度贪婪'])
        return df
    except Exception as e:
        st.error(f"❌ 数据加载失败：{e}")
        return pd.DataFrame()

# 加载数据
df = load_data()

if df.empty:
    st.warning("⚠️ 数据为空，请确认上传的数据文件是否完整。")
else:
    st.success(f"✅ 数据加载成功，共 {len(df)} 条记录，时间范围：{df['date'].min().date()} - {df['date'].max().date()}")

    # 显示数据样本
    with st.expander("📄 展示数据样本"):
        st.dataframe(df.head())

    # 折线图：贪婪恐惧指数 & 比特币价格
    st.subheader("📈 贪婪恐惧指数与比特币价格走势")

    colors = {'极度恐惧': '#1a5e1a', '恐惧': '#2e8b57', '贪婪': '#ffa500', '极度贪婪': '#ff4500'}

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    for level, group in df.groupby('greed_level'):
        ax1.plot(group['date'], group['greed'], 'o',
                 color=colors[level], alpha=0.6, label=str(level), markersize=4)

    ax2.plot(df['date'], df['price'], 'b-', linewidth=1.2, alpha=0.7, label='BTC价格')

    ax1.set_ylabel("贪婪恐惧指数", fontsize=12)
    ax2.set_ylabel("比特币价格", fontsize=12)
    ax1.set_xlabel("日期", fontsize=12)

    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    fig.tight_layout()

    st.pyplot(fig)

    # 相关性分析
    st.subheader("🔍 贪婪恐惧指数与比特币价格的相关性")

    corr = df['greed'].corr(df['price'])
    st.markdown(f"**相关系数（皮尔森）为： `{corr:.4f}`**")
    
    fig2, ax = plt.subplots()
    sns.regplot(x='greed', y='price', data=df, ax=ax, scatter_kws={'alpha':0.5})
    ax.set_title("贪婪恐惧指数 vs 比特币价格")
    st.pyplot(fig2)

    # 动态过滤器
    st.subheader("📅 按时间范围筛选数据")
    date_range = st.date_input("选择时间区间", [df['date'].min().date(), df['date'].max().date()])

    if len(date_range) == 2:
        filtered_df = df[(df['date'] >= pd.to_datetime(date_range[0])) & 
                         (df['date'] <= pd.to_datetime(date_range[1]))]
        st.write(f"筛选后数据量：{len(filtered_df)}")

        fig3, ax = plt.subplots(figsize=(12, 5))
        ax.plot(filtered_df['date'], filtered_df['price'], label="BTC 价格", color='blue')
        ax.set_ylabel("价格")
        ax.set_title("比特币价格走势（筛选）")
        st.pyplot(fig3)
