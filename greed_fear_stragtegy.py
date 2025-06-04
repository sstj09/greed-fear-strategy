import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests  # 确保导入 requests 库

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
                                   labels=['Extreme Fear', 'Fear', 'Greed', 'Extreme Greed'])
        return df
    except Exception as e:
        st.error(f"❌ Data loading failed: {e}")
        return pd.DataFrame()

# 加载数据
df = load_data()

# ============================= 用户调研问卷 =============================
st.subheader("📝 用户调研问卷")

with st.form("user_survey_form"):
    st.markdown("我们希望了解您的使用感受，以下问题完全匿名，仅用于改进产品体验。")

    experience = st.radio("您对本页面展示的数据和分析是否易于理解？", 
                          ["非常易懂", "基本理解", "有些难", "不太理解"], index=0)

    insight = st.radio("这些数据对您了解比特币市场有帮助吗？", 
                       ["非常有帮助", "一般", "帮助不大", "没有帮助"], index=0)

    expected_feature = st.text_input("您希望我们未来加入哪些功能？（如：价格预测、新闻热度分析等）",
                                    placeholder="请分享您的想法...")

    submit = st.form_submit_button("提交反馈")

    if submit:
        # 确保所有必填字段已填写
        if not experience or not insight:
            st.error("请回答所有必填问题")
            st.stop()
            
        # 写入 Notion 数据库
        notion_token = "ntn_T401856748914gT9Zu7PzfLJyPFFC0r0awF9pDiVWEV8SX"
        database_id = "2080ef86-7941-80d3-9a68-000cf75416b3"

        headers = {
            "Authorization": f"Bearer {notion_token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
        
        # 检查数据库ID格式是否正确
        if len(database_id) != 36 or database_id.count("-") != 4:
            st.error("数据库ID格式不正确，请检查")
            st.stop()

        # 准备Notion API请求数据
        notion_payload = {
            "parent": {"database_id": database_id},
            "properties": {
                "提交时间": {
                    "date": {"start": datetime.utcnow().isoformat() + "Z"}
                },
                "理解度": {
                    "select": {"name": experience}
                },
                "帮助程度": {
                    "select": {"name": insight}
                },
                "建议功能": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": expected_feature or "无"}
                    }]
                }
            }
        }

        try:
            response = requests.post(
                "https://api.notion.com/v1/pages",
                headers=headers,
                json=notion_payload
            )

            if response.status_code == 200:
                st.success("✅ 感谢您的反馈，我们会认真参考！")
            elif response.status_code == 404:
                st.error("❌ Notion 数据库未找到，请检查数据库ID和集成权限")
                st.json(response.json())
            else:
                st.error(f"❌ Notion 写入失败 (状态码: {response.status_code})")
                st.json(response.json())
                
        except Exception as e:
            st.error(f"❌ 连接Notion时出错: {str(e)}")

# ============================= 主内容区域 =============================
if df.empty:
    st.warning("⚠️ Data is empty. Please check if the uploaded files are complete.")
else:
    st.success(f"✅ Data loaded successfully. Total {len(df)} records from {df['date'].min().date()} to {df['date'].max().date()}")

    # 显示数据样本
    with st.expander("📄 展示数据样本"):
        st.dataframe(df.head())

    # 折线图：贪婪恐惧指数 & 比特币价格
    st.subheader("📈 贪婪恐惧指数与比特币价格走势")

    colors = {'Extreme Fear': '#1a5e1a', 'Fear': '#2e8b57', 'Greed': '#ffa500', 'Extreme Greed': '#ff4500'}

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    for level, group in df.groupby('greed_level'):
        ax1.plot(group['date'], group['greed'], 'o',
                 color=colors[level], alpha=0.6, label=str(level), markersize=4)

    ax2.plot(df['date'], df['price'], 'b-', linewidth=1.2, alpha=0.7, label='BTC_price')

    ax1.set_ylabel("Fear and Greed Index", fontsize=12)
    ax2.set_ylabel("Bitcoin Price", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)

    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    fig.tight_layout()

    st.pyplot(fig)

    # 相关性分析
    st.subheader("🔍 贪婪恐惧指数与比特币价格的相关性")

    corr = df['greed'].corr(df['price'])
    st.markdown(f"**Pearson correlation coefficient: `{corr:.4f}`**")
    
    fig2, ax = plt.subplots()
    sns.regplot(x='greed', y='price', data=df, ax=ax, scatter_kws={'alpha':0.5})
    ax.set_title("Fear and Greed Index vs Bitcoin Price")
    st.pyplot(fig2)

    # 动态过滤器
    st.subheader("📅 按时间范围筛选数据")
    date_range = st.date_input("选择时间区间", [df['date'].min().date(), df['date'].max().date()])

    if len(date_range) == 2:
        filtered_df = df[(df['date'] >= pd.to_datetime(date_range[0])) & 
                         (df['date'] <= pd.to_datetime(date_range[1]))]
        st.write(f"筛选后数据量：{len(filtered_df)}")

        fig3, ax = plt.subplots(figsize=(12, 5))
        ax.plot(filtered_df['date'], filtered_df['price'], label="BTC Price", color='blue')
        ax.set_ylabel("Price")
        ax.set_title("Bitcoin Price Trend (Filtered)")
        st.pyplot(fig3)
