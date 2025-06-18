
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="US Diabetes Strategic Dashboard", page_icon="🩺", layout="wide")

# --- EMBEDDED, PRE-CALCULATED DATA ---
# PASTE THE ENTIRE OUTPUT FROM YOUR COLAB "DATA EXPORT" SCRIPT HERE
# -----------------------------------------------------------------
MAP_DATA = [{'STATE_ABBR': 'AK', 'Prevalence_Rate_%': 10.550835845437106, 'Poverty_Rate_2015': 10.4}, {'STATE_ABBR': 'AL', 'Prevalence_Rate_%': 17.861191585842047, 'Poverty_Rate_2015': 17.1}, {'STATE_ABBR': 'AR', 'Prevalence_Rate_%': 18.916857360793287, 'Poverty_Rate_2015': 17.2}, {'STATE_ABBR': 'AZ', 'Prevalence_Rate_%': 14.948323670279809, 'Poverty_Rate_2015': 16.4}, {'STATE_ABBR': 'CA', 'Prevalence_Rate_%': 11.82453909726637, 'Poverty_Rate_2015': 15.3}, {'STATE_ABBR': 'CO', 'Prevalence_Rate_%': 9.73189157161902, 'Poverty_Rate_2015': 11.0}, {'STATE_ABBR': 'CT', 'Prevalence_Rate_%': 12.540003368704733, 'Poverty_Rate_2015': 9.8}, {'STATE_ABBR': 'DC', 'Prevalence_Rate_%': 14.486567913632939, 'Poverty_Rate_2015': 17.3}, {'STATE_ABBR': 'DE', 'Prevalence_Rate_%': 16.104407781334647, 'Poverty_Rate_2015': 11.7}, {'STATE_ABBR': 'FL', 'Prevalence_Rate_%': 14.95317484820418, 'Poverty_Rate_2015': 15.7}, {'STATE_ABBR': 'GA', 'Prevalence_Rate_%': 17.166738105443635, 'Poverty_Rate_2015': 16.0}, {'STATE_ABBR': 'HI', 'Prevalence_Rate_%': 10.932475884244374, 'Poverty_Rate_2015': 9.3}, {'STATE_ABBR': 'IA', 'Prevalence_Rate_%': 12.638687891944041, 'Poverty_Rate_2015': 11.8}, {'STATE_ABBR': 'ID', 'Prevalence_Rate_%': 12.5625539257981, 'Poverty_Rate_2015': 14.8}, {'STATE_ABBR': 'IL', 'Prevalence_Rate_%': 13.523737469264233, 'Poverty_Rate_2015': 13.0}, {'STATE_ABBR': 'IN', 'Prevalence_Rate_%': 15.20052813995709, 'Poverty_Rate_2015': 14.1}, {'STATE_ABBR': 'KS', 'Prevalence_Rate_%': 13.262359381061161, 'Poverty_Rate_2015': 12.1}, {'STATE_ABBR': 'KY', 'Prevalence_Rate_%': 17.776510832383124, 'Poverty_Rate_2015': 18.5}, {'STATE_ABBR': 'LA', 'Prevalence_Rate_%': 18.30926083262532, 'Poverty_Rate_2015': 20.2}, {'STATE_ABBR': 'MA', 'Prevalence_Rate_%': 11.613111925814104, 'Poverty_Rate_2015': 10.4}, {'STATE_ABBR': 'MD', 'Prevalence_Rate_%': 15.758925021865311, 'Poverty_Rate_2015': 9.7}, {'STATE_ABBR': 'ME', 'Prevalence_Rate_%': 12.673879443585781, 'Poverty_Rate_2015': 12.5}, {'STATE_ABBR': 'MI', 'Prevalence_Rate_%': 12.648575913882038, 'Poverty_Rate_2015': 15.0}, {'STATE_ABBR': 'MN', 'Prevalence_Rate_%': 11.052034171694844, 'Poverty_Rate_2015': 9.9}, {'STATE_ABBR': 'MO', 'Prevalence_Rate_%': 16.774016719199672, 'Poverty_Rate_2015': 14.0}, {'STATE_ABBR': 'MS', 'Prevalence_Rate_%': 19.774386197743862, 'Poverty_Rate_2015': 20.8}, {'STATE_ABBR': 'MT', 'Prevalence_Rate_%': 11.271102284011917, 'Poverty_Rate_2015': 13.3}, {'STATE_ABBR': 'NC', 'Prevalence_Rate_%': 13.450904199671202, 'Poverty_Rate_2015': 15.4}, {'STATE_ABBR': 'ND', 'Prevalence_Rate_%': 12.172511084240226, 'Poverty_Rate_2015': 10.7}, {'STATE_ABBR': 'NE', 'Prevalence_Rate_%': 12.57764860090044, 'Poverty_Rate_2015': 11.4}, {'STATE_ABBR': 'NH', 'Prevalence_Rate_%': 12.421383647798741, 'Poverty_Rate_2015': 7.3}, {'STATE_ABBR': 'NJ', 'Prevalence_Rate_%': 12.72369714847591, 'Poverty_Rate_2015': 10.4}, {'STATE_ABBR': 'NM', 'Prevalence_Rate_%': 14.517328573553472, 'Poverty_Rate_2015': 19.8}, {'STATE_ABBR': 'NV', 'Prevalence_Rate_%': 12.319109461966605, 'Poverty_Rate_2015': 13.8}, {'STATE_ABBR': 'NY', 'Prevalence_Rate_%': 12.629617627997408, 'Poverty_Rate_2015': 14.7}, {'STATE_ABBR': 'OH', 'Prevalence_Rate_%': 16.42167156656867, 'Poverty_Rate_2015': 14.6}, {'STATE_ABBR': 'OK', 'Prevalence_Rate_%': 16.50555475400375, 'Poverty_Rate_2015': 16.3}, {'STATE_ABBR': 'OR', 'Prevalence_Rate_%': 13.294689603590125, 'Poverty_Rate_2015': 13.3}, {'STATE_ABBR': 'PA', 'Prevalence_Rate_%': 13.522945384749608, 'Poverty_Rate_2015': 12.9}, {'STATE_ABBR': 'PR', 'Prevalence_Rate_%': 20.60459940652819, 'Poverty_Rate_2015': 43.4}, {'STATE_ABBR': 'RI', 'Prevalence_Rate_%': 13.192313902793476, 'Poverty_Rate_2015': 12.8}, {'STATE_ABBR': 'SC', 'Prevalence_Rate_%': 16.583131697200137, 'Poverty_Rate_2015': 15.3}, {'STATE_ABBR': 'SD', 'Prevalence_Rate_%': 12.546790517121863, 'Poverty_Rate_2015': 13.5}, {'STATE_ABBR': 'TN', 'Prevalence_Rate_%': 17.914707857622567, 'Poverty_Rate_2015': 15.8}, {'STATE_ABBR': 'TX', 'Prevalence_Rate_%': 16.30271598198444, 'Poverty_Rate_2015': 15.6}, {'STATE_ABBR': 'UT', 'Prevalence_Rate_%': 10.268798313422348, 'Poverty_Rate_2015': 10.2}, {'STATE_ABBR': 'VA', 'Prevalence_Rate_%': 14.366817286525317, 'Poverty_Rate_2015': 11.0}, {'STATE_ABBR': 'VT', 'Prevalence_Rate_%': 10.4613485573214, 'Poverty_Rate_2015': 11.5}, {'STATE_ABBR': 'WA', 'Prevalence_Rate_%': 12.132901941264311, 'Poverty_Rate_2015': 11.3}, {'STATE_ABBR': 'WI', 'Prevalence_Rate_%': 11.364740165128703, 'Poverty_Rate_2015': 11.8}, {'STATE_ABBR': 'WV', 'Prevalence_Rate_%': 16.77321668909825, 'Poverty_Rate_2015': 17.9}, {'STATE_ABBR': 'WY', 'Prevalence_Rate_%': 12.996718920889535, 'Poverty_Rate_2015': 11.1}]
INTERACTION_DATA = [{'AGE_BRACKET': '18-39', 'BMI_CATEGORY': 'Normal/Under', 'Prevalence': 2.5044477897906114}, {'AGE_BRACKET': '18-39', 'BMI_CATEGORY': 'Obese', 'Prevalence': 7.237862480612868}, {'AGE_BRACKET': '18-39', 'BMI_CATEGORY': 'Overweight', 'Prevalence': 2.647891635417464}, {'AGE_BRACKET': '40-59', 'BMI_CATEGORY': 'Normal/Under', 'Prevalence': 6.345000697252824}, {'AGE_BRACKET': '40-59', 'BMI_CATEGORY': 'Obese', 'Prevalence': 20.47926350467079}, {'AGE_BRACKET': '40-59', 'BMI_CATEGORY': 'Overweight', 'Prevalence': 8.757145258910558}, {'AGE_BRACKET': '60+', 'BMI_CATEGORY': 'Normal/Under', 'Prevalence': 11.959208128417005}, {'AGE_BRACKET': '60+', 'BMI_CATEGORY': 'Obese', 'Prevalence': 33.86688353695729}, {'AGE_BRACKET': '60+', 'BMI_CATEGORY': 'Overweight', 'Prevalence': 18.062909603360286}]
INCOME_DATA = {'$15-25k': 18.98828468883321, '$25-35k': 15.994791799637467, '$35-50k': 13.748678011729643, '$50-75k': 11.530364372469635, '<$15k': 22.345587847460312, '>$75k': 9.50968248693393}
EDU_DATA = {'College Grad': 9.997460372519992, 'HS Diploma': 16.00520282903829, 'No HS Diploma': 22.986080586080586, 'Some College': 14.093134199350072}
RACE_DATA = {'Am. Indian/Alaskan Native': 20.435049019607842, 'Asian': 10.307265440345288, 'Black': 21.288515406162464, 'Hispanic': 15.859364056590557, 'Multiracial': 14.24007863373879, 'Nat. Hawaiian/Pacific Isl.': 14.061384725196287, 'Other': 13.762886597938145, 'White': 12.766864018163886}
BP_DATA = {'No': 6.251476354703075, 'Yes': 24.731194889328776}
CHOL_DATA = {'No': 9.365491175811153, 'Yes': 23.310935631133013}
RISK_RATIOS = {'obesity': 3.73887467576189, 'exercise': 1.722255302533886, 'cholesterol': 2.489024354786607, 'smoking': 1.4157190247667792}
# -----------------------------------------------------------------

# --- HELPER FUNCTION ---
def create_highlighted_bar_chart(data_dict, title, height=270, x_title="Diabetes Prevalence (%)", y_order=None):
    data = pd.Series(data_dict).dropna()
    if data.empty: return go.Figure().update_layout(title_text=f"{title}<br>No data available", height=height)
    max_val_index = data.idxmax()
    colors = ['#EF553B' if i == max_val_index else 'grey' for i in data.index]
    fig = px.bar(data, y=data.index, x=data.values, orientation='h', text_auto='.1f', labels={'x':x_title, 'y':''})
    fig.update_traces(marker_color=colors, textfont_size=12, textangle=0, textposition="outside")
    fig.update_layout(margin=dict(l=0, r=20, t=40, b=5), height=height, title_text=title, title_x=0.5, yaxis={'categoryorder':'total ascending'})
    if y_order: fig.update_layout(yaxis={'categoryorder':'array', 'categoryarray': y_order})
    return fig

# --- MAIN DASHBOARD FUNCTION ---
def display_dashboard():
    st.markdown("#### 🩺 Type 2 Diabetes in the USA: A Strategic Analysis of Interconnected Risk")
    st.markdown("---")
    
    st.markdown("<h6>Key National Risk Multipliers</h6>", unsafe_allow_html=True)
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    ratios = RISK_RATIOS
    df_map_data = pd.DataFrame(MAP_DATA)
    natl_avg = df_map_data['Prevalence_Rate_%'].mean()
    kpi1.metric("National Prevalence", f"{natl_avg:.1f}%")
    kpi2.metric("Obesity Risk", f"{ratios.get('obesity', 0):.1f}x", "vs. Normal/Under")
    kpi3.metric("No Exercise Risk", f"{ratios.get('exercise', 0):.1f}x", "vs. Exercised")
    kpi4.metric("High Chol. Risk", f"{ratios.get('cholesterol', 0):.1f}x", "vs. No")
    kpi5.metric("Former Smoker Risk", f"{ratios.get('smoking', 0):.1f}x", "vs. Never Smoked")
    st.markdown("---")

    col1, col2, col3 = st.columns([0.3, 0.4, 0.3])

    with col1:
        st.markdown("<h6>The Disparity Story</h6>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["By Economic Status", "By Demographics"])
        with tab1:
            fig_income = create_highlighted_bar_chart(INCOME_DATA, "Income Disparities", y_order=['>$75k', '$50-75k', '$35-50k', '$25-35k', '$15-25k', '<$15k'], height=250)
            st.plotly_chart(fig_income, use_container_width=True)
            fig_edu = create_highlighted_bar_chart(EDU_DATA, "Education Disparities", y_order=['College Grad', 'Some College', 'HS Diploma', 'No HS Diploma'], height=230)
            st.plotly_chart(fig_edu, use_container_width=True)
        with tab2:
            fig_race = create_highlighted_bar_chart(RACE_DATA, "Prevalence by Race/Ethnicity", height=480)
            st.plotly_chart(fig_race, use_container_width=True)

    with col2:
        st.markdown("<h6>Geographic & Core Risk Factors</h6>", unsafe_allow_html=True)
        fig_map = px.choropleth(df_map_data, locations='STATE_ABBR', locationmode="USA-states", color='Prevalence_Rate_%', scope="usa",
                              color_continuous_scale="Reds", hover_name='STATE_ABBR',
                              hover_data={'Poverty_Rate_2015': ':.1f', 'Prevalence_Rate_%':':.1f', 'STATE_ABBR':False})
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=250, title_text="Hover on a State to see its Poverty Rate", title_x=0.5, title_font_size=14)
        st.plotly_chart(fig_map, use_container_width=True)

        df_interaction = pd.DataFrame(INTERACTION_DATA)
        color_map = {'Obese': '#EF553B', 'Overweight': 'grey', 'Normal/Under': 'grey'}
        fig_interaction = px.bar(df_interaction, x='AGE_BRACKET', y='Prevalence', color='BMI_CATEGORY', barmode='group',
                                 labels={'Prevalence':'Diabetes Prevalence (%)', 'AGE_BRACKET':''}, text_auto='.1f',
                                 color_discrete_map=color_map, title="The Amplifying Effect of Age on BMI Risk")
        fig_interaction.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300, legend_title_text='BMI')
        st.plotly_chart(fig_interaction, use_container_width=True)

    with col3:
        st.markdown("<h6>Clinical Insights & Personal Risk</h6>", unsafe_allow_html=True)
        fig_bp = create_highlighted_bar_chart(BP_DATA, "Impact of High Blood Pressure", x_title="Diabetes Prevalence (%)", height=140)
        st.plotly_chart(fig_bp, use_container_width=True)
        
        fig_chol = create_highlighted_bar_chart(CHOL_DATA, "Impact of High Cholesterol", x_title="Diabetes Prevalence (%)", height=140)
        st.plotly_chart(fig_chol, use_container_width=True)
        
        st.markdown("<h6 style='margin-top:20px;'>Personal Risk Estimator</h6>", unsafe_allow_html=True)
        
        age_points = {'18-24': 0, '25-34': 2, '35-44': 6, '45-54': 12, '55-64': 18, '65+': 25}
        bmi_points = {'Normal/Under': 0, 'Overweight': 7, 'Obese': 18}
        bp_points = {'No': 0, 'Yes': 15}
        chol_points = {'No': 0, 'Yes': 8}
        race_points = {'White': 0, 'Asian': -2, 'Hispanic': 4, 'Black': 9, 'Am. Indian/Alaskan Native': 9}
        edu_points = {'College Grad': -2, 'Some College': 2, 'HS Diploma': 4, 'No HS Diploma': 6}

        calc_col1, calc_col2 = st.columns(2)
        with calc_col1:
            age_input = st.selectbox("Age Group:", list(age_points.keys()), index=3)
            bmi_input = st.selectbox("BMI Category:", list(bmi_points.keys()), index=1)
            bp_input = st.radio("High Blood Pressure?", list(bp_points.keys()), index=1)
        with calc_col2:
            chol_input = st.radio("High Cholesterol?", list(chol_points.keys()), index=1)
            race_input = st.selectbox("Race/Ethnicity:", list(race_points.keys()))
            edu_input = st.selectbox("Education Level:", list(edu_points.keys()), index=1)

        if st.button("Estimate My Risk", use_container_width=True, type="primary"):
            score = 2.0 
            score += age_points[age_input]
            score += bmi_points[bmi_input]
            score += bp_points[bp_input]
            score += chol_points[chol_input]
            score += race_points[race_input]
            score += edu_points[edu_input]
            risk_prob = min(score, 75.0)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_prob,
                number={'suffix': '%', 'font': {'size': 36}},
                title={'text': "Estimated Prevalence for Your Profile"},
                gauge={'axis': {'range': [None, 75]}, 'bar': {'color': "#EF553B"},
                       'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': natl_avg}}
            ))
            fig_gauge.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=160)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            risk_level = "HIGH" if risk_prob > natl_avg * 1.5 else ("MODERATE" if risk_prob > natl_avg * 0.75 else "LOW")
            st.markdown(f"For a group with your profile, the estimated prevalence of Type 2 diabetes is **{risk_prob:.1f}%**.")
            st.caption("*This is a statistical estimate based on 2015 data, not a medical diagnosis.*")

# --- LOGIN LOGIC ---

def login_page():
    # Use columns to center the content
    left_space, login_col, right_space = st.columns([1, 1.5, 1])

    with login_col:
        # Centered logo image using HTML
        st.markdown(
            """
            <div style='text-align: center;'>
                <img src='https://www.aub.edu.lb/osb/wids/PublishingImages/OSB-MSBA-burgundy.png' width='350'/>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Centered titles
        st.markdown("<h1 style='text-align: center;'>Login Page</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Dashboard Access</h3>", unsafe_allow_html=True)

        # Login form
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In", use_container_width=True, type="primary")

            if submitted:
                if password == "msba":
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("Incorrect password. Please try again.")


# The rest of your app.py script (the MAIN APP ROUTER) remains unchanged.
# --- MAIN APP ROUTER ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    display_dashboard()
else:
    login_page()
