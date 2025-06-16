
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="US Diabetes Strategic Dashboard", page_icon="🩺", layout="wide")

# --- DATA & MODEL LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv('brfss_2015_dashboard_data.csv')
    df['DIABETES_STATUS'] = df['DIABETE3'].map({1: 'Yes', 2: 'Yes', 3: 'No', 4: 'No'})
    df.dropna(subset=['DIABETES_STATUS'], inplace=True)
    
    df['AGE_BRACKET'] = pd.cut(df['_AGEG5YR'], bins=[0, 4, 8, 14], labels=['18-39', '40-59', '60+'], right=True)
    df['BMI_CATEGORY'] = df['_BMI5'].apply(lambda x: 'Obese' if x >= 3000 else ('Overweight' if x >= 2500 else 'Normal/Under'))
    df['INCOME_LEVEL'] = df['_INCOMG'].map({1:'<$15k', 2:'$15-25k', 3:'$25-35k', 4:'$35-50k', 5:'$50-75k', 7:'>$75k', 8:'>$75k'})
    df['EDUCATION_LEVEL'] = df['_EDUCAG'].map({1:'No HS Diploma', 2:'HS Diploma', 3:'Some College', 4:'College Grad'})
    df['RACE_ETHNICITY'] = df['_RACE'].map({1.0: 'White', 2.0: 'Black', 3.0: 'Am. Indian/Alaskan Native', 4.0: 'Asian', 5.0: 'Nat. Hawaiian/Pacific Isl.', 6.0: 'Other', 7.0: 'Multiracial', 8.0: 'Hispanic'})
    df['HAS_HIGH_BP'] = df['BPHIGH4'].map({1.0: 'Yes', 3.0: 'No'})
    df['HAS_HIGH_CHOL'] = df['TOLDHI2'].map({1.0: 'Yes', 2.0: 'No'})
    df['EXERCISED'] = df['EXERANY2'].map({1.0: 'Yes', 2.0: 'No'})
    df['SMOKING_STATUS'] = df['_SMOKER3'].map({1:'Current', 2:'Current', 3:'Former', 4:'Never'})

    state_map = {1: 'AL', 2: 'AK', 4: 'AZ', 5: 'AR', 6: 'CA', 8: 'CO', 9: 'CT', 10: 'DE', 11: 'DC', 12: 'FL', 13: 'GA', 15: 'HI', 16: 'ID', 17: 'IL', 18: 'IN', 19: 'IA', 20: 'KS', 21: 'KY', 22: 'LA', 23: 'ME', 24: 'MD', 25: 'MA', 26: 'MI', 27: 'MN', 28: 'MS', 29: 'MO', 30: 'MT', 31: 'NE', 32: 'NV', 33: 'NH', 34: 'NJ', 35: 'NM', 36: 'NY', 37: 'NC', 38: 'ND', 39: 'OH', 40: 'OK', 41: 'OR', 42: 'PA', 44: 'RI', 45: 'SC', 46: 'SD', 47: 'TN', 48: 'TX', 49: 'UT', 50: 'VT', 51: 'VA', 53: 'WA', 54: 'WV', 55: 'WI', 56: 'WY', 72: 'PR'}
    df['STATE_ABBR'] = df['_STATE'].map(state_map)
    
    poverty_data = {'STATE_ABBR': ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC', 'PR'], 'Poverty_Rate_2015': [17.1, 10.4, 16.4, 17.2, 15.3, 11.0, 9.8, 11.7, 15.7, 16.0, 9.3, 14.8, 13.0, 14.1, 11.8, 12.1, 18.5, 20.2, 12.5, 9.7, 10.4, 15.0, 9.9, 20.8, 14.0, 13.3, 11.4, 13.8, 7.3, 10.4, 19.8, 14.7, 15.4, 10.7, 14.6, 16.3, 13.3, 12.9, 12.8, 15.3, 13.5, 15.8, 15.6, 10.2, 11.5, 11.0, 11.3, 17.9, 11.8, 11.1, 17.3, 43.4]}
    state_poverty_df = pd.DataFrame(poverty_data)
    
    return df, state_poverty_df

@st.cache_resource
def load_model():
    try:
        model_data = joblib.load('diabetes_risk_model_v2.joblib')
        return model_data['model'], model_data['columns']
    except FileNotFoundError:
        return None, None

def create_highlighted_bar_chart(data, title, height=270, x_title="Prevalence (%)", y_order=None):
    data = data.dropna()
    if data.empty: return go.Figure().update_layout(title_text=f"{title}<br>No data for selection", height=height)
    max_val_index = data.idxmax()
    colors = ['#EF553B' if i == max_val_index else 'grey' for i in data.index]
    fig = px.bar(data, y=data.index, x=data.values, orientation='h', text_auto='.1f', labels={'x':x_title, 'y':''})
    fig.update_traces(marker_color=colors, textfont_size=12, textangle=0, textposition="outside")
    fig.update_layout(margin=dict(l=0, r=20, t=40, b=5), height=height, title_text=title, title_x=0.5)
    if y_order: fig.update_layout(yaxis={'categoryorder':'array', 'categoryarray': y_order})
    return fig

def get_risk_ratio(df, column, risk_group, baseline_group):
    risk_rate = df[df[column] == risk_group]['DIABETES_STATUS'].value_counts(normalize=True).get('Yes', 0)
    baseline_rate = df[df[column] == baseline_group]['DIABETES_STATUS'].value_counts(normalize=True).get('Yes', 0)
    return risk_rate / baseline_rate if baseline_rate > 0 else 0

df_full, df_poverty = load_data()
model, model_columns = load_model()

# --- HEADER AND STATIC KPI BAR ---
st.markdown("#### 🩺 Diabetes in the USA: A Strategic Analysis of Interconnected Risk")
st.markdown("---")
st.markdown("<h6>Key National Risk Multipliers</h6>", unsafe_allow_html=True)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
obesity_ratio = get_risk_ratio(df_full, 'BMI_CATEGORY', 'Obese', 'Normal/Under')
kpi1.metric("Obesity Risk", f"{obesity_ratio:.1f}x", "vs. Normal/Under BMI")
exercise_ratio = get_risk_ratio(df_full, 'EXERCISED', 'No', 'Yes')
kpi2.metric("No Exercise Risk", f"{exercise_ratio:.1f}x", "vs. Exercised")
chol_ratio = get_risk_ratio(df_full, 'HAS_HIGH_CHOL', 'Yes', 'No')
kpi3.metric("High Cholesterol Risk", f"{chol_ratio:.1f}x", "vs. No")
smoke_ratio = get_risk_ratio(df_full, 'SMOKING_STATUS', 'Former', 'Never')
kpi4.metric("Former Smoker Risk", f"{smoke_ratio:.1f}x", "vs. Never Smoked")
st.markdown("---")

# --- 3-COLUMN LAYOUT ---
col1, col2, col3 = st.columns([0.3, 0.4, 0.3])

# --- COLUMN 1: DISPARITY DRIVERS ---
with col1:
    st.markdown("<h6>The Disparity Story</h6>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["By Economic Status", "By Demographics"])
    with tab1:
        income_prevalence = df_full.groupby('INCOME_LEVEL')['DIABETES_STATUS'].value_counts(normalize=True).unstack().get('Yes', pd.Series()) * 100
        fig_income = create_highlighted_bar_chart(income_prevalence, "Income Disparities", y_order=['>$75k', '$50-75k', '$35-50k', '$25-35k', '$15-25k', '<$15k'], height=250)
        st.plotly_chart(fig_income, use_container_width=True)
        edu_prevalence = df_full.groupby('EDUCATION_LEVEL')['DIABETES_STATUS'].value_counts(normalize=True).unstack().get('Yes', pd.Series()) * 100
        fig_edu = create_highlighted_bar_chart(edu_prevalence, "Education Disparities", y_order=['College Grad', 'Some College', 'HS Diploma', 'No HS Diploma'], height=230)
        st.plotly_chart(fig_edu, use_container_width=True)
    with tab2:
        race_prevalence = df_full.groupby('RACE_ETHNICITY')['DIABETES_STATUS'].value_counts(normalize=True).unstack().get('Yes', pd.Series()) * 100
        fig_race = create_highlighted_bar_chart(race_prevalence, "Prevalence by Race/Ethnicity", height=480)
        st.plotly_chart(fig_race, use_container_width=True)

# --- COLUMN 2: GEOGRAPHY AND CORE RISK ---
with col2:
    st.markdown("<h6>Geographic & Core Risk Factors</h6>", unsafe_allow_html=True)
    state_prev_full = df_full.groupby('STATE_ABBR')['DIABETES_STATUS'].value_counts(normalize=True).unstack().get('Yes', pd.Series()) * 100
    merged_df = pd.merge(state_prev_full.reset_index(name='Prevalence_Rate_%'), df_poverty, on='STATE_ABBR')
    
    fig_map = px.choropleth(merged_df, locations='STATE_ABBR', locationmode="USA-states", color='Prevalence_Rate_%', scope="usa",
                          color_continuous_scale="Reds", hover_name='STATE_ABBR',
                          hover_data={'Poverty_Rate_2015': ':.1f', 'Prevalence_Rate_%':':.1f', 'STATE_ABBR':False})
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=250)
    st.plotly_chart(fig_map, use_container_width=True)

    interaction_data = df_full.groupby(['AGE_BRACKET', 'BMI_CATEGORY'])['DIABETES_STATUS'].value_counts(normalize=True).unstack().get('Yes', pd.Series(0)).reset_index(name='Prevalence')
    interaction_data['Prevalence'] *= 100
    color_map = {'Obese': '#EF553B', 'Overweight': 'grey', 'Normal/Under': 'grey'}
    
    fig_interaction = px.bar(interaction_data, x='AGE_BRACKET', y='Prevalence', color='BMI_CATEGORY', barmode='group',
                             labels={'Prevalence':'Prevalence (%)', 'AGE_BRACKET':''}, text_auto='.1f',
                             color_discrete_map=color_map, title="The Amplifying Effect of Age on BMI Risk")
    fig_interaction.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300, legend_title_text='BMI')
    st.plotly_chart(fig_interaction, use_container_width=True)

# --- COLUMN 3: CLINICAL INSIGHTS & PERSONAL RISK ---
with col3:
    st.markdown("<h6>Clinical Insights & Personal Risk</h6>", unsafe_allow_html=True)
    bp_data = df_full.dropna(subset=['HAS_HIGH_BP']).groupby('HAS_HIGH_BP')['DIABETES_STATUS'].value_counts(normalize=True).unstack().get('Yes', pd.Series()) * 100
    fig_bp = create_highlighted_bar_chart(bp_data, "Impact of High Blood Pressure", x_title="", height=140)
    st.plotly_chart(fig_bp, use_container_width=True)
    
    chol_data = df_full.dropna(subset=['HAS_HIGH_CHOL']).groupby('HAS_HIGH_CHOL')['DIABETES_STATUS'].value_counts(normalize=True).unstack().get('Yes', pd.Series()) * 100
    fig_chol = create_highlighted_bar_chart(chol_data, "Impact of High Cholesterol", x_title="", height=140)
    st.plotly_chart(fig_chol, use_container_width=True)
    
    st.markdown("<h6 style='margin-top:20px;'>Personal Risk Estimator</h6>", unsafe_allow_html=True)
    if model is not None:
        calc_col1, calc_col2 = st.columns(2)
        with calc_col1:
            age_input = st.selectbox("Age:", ['18-39', '40-59', '60+'], index=1)
            bmi_input = st.selectbox("BMI:", ['Normal/Under', 'Overweight', 'Obese'], index=1)
            bp_input = st.radio("High BP?", ('Yes', 'No'))
        with calc_col2:
            chol_input = st.radio("High Chol?", ('Yes', 'No'))
            race_input = st.selectbox("Race:", ['White', 'Black', 'Hispanic'], index=0)
            edu_input = st.selectbox("Education:", ['No College', 'College'], index=0)

        if st.button("Estimate My Risk", use_container_width=True, type="primary"):
            input_df = pd.DataFrame([np.zeros(len(model_columns))], columns=model_columns)
            input_map = {f'Age_{age_input}':1, f'BMI_{bmi_input}':1, f'High_BP_{bp_input}':1, f'High_Chol_{chol_input}':1, f'Education_{edu_input}':1, f'Race_{race_input}':1}
            for col, val in input_map.items():
                if col in input_df.columns: input_df[col] = val
            
            risk_prob = model.predict_proba(input_df)[0][1] * 100
            natl_avg = (df_full['DIABETES_STATUS'] == 'Yes').mean() * 100

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_prob,
                number={'suffix': '%', 'font': {'size': 36}},
                title={'text': "Predicted Diabetes Risk"},
                gauge={'axis': {'range': [None, 60]}, 'bar': {'color': "#EF553B"},
                       # The 'steps' argument is now removed for a cleaner look
                       'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': natl_avg}}
            ))
            fig_gauge.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=180)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            risk_level = "HIGH" if risk_prob > natl_avg * 1.5 else ("MODERATE" if risk_prob > natl_avg * 0.75 else "LOW")
            st.markdown(f"Your estimated risk of **{risk_prob:.1f}%** is considered **{risk_level}** compared to the national average of {natl_avg:.1f}%.")
    else:
        st.warning("Risk calculator model is unavailable.")
