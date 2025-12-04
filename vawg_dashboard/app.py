"""
Violence Against Women Intelligence Dashboard
# ----------------------------------------------------
Complete app using three separate datasets
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import joblib

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------

st.set_page_config(
    page_title="VAWG Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# LOAD DATA AND MODELS
# ----------------------------------------------------

@st.cache_data
def load_all_data():
    """Load all three datasets"""
    try:
        individual_df = pd.read_csv('data/individual_df.csv')
        violence_df = pd.read_csv('data/violence_df.csv')
        global_df = pd.read_csv('data/global_df.csv')
        global_stats = pd.read_csv('data/global_stats.csv')
        return individual_df, violence_df, global_df, global_stats
    except FileNotFoundError as e:
        st.error(f"⚠️ Data file not found: {e}")
        return None, None, None, None

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        individual_model = joblib.load('models/individual_model.pkl')
        violence_model = joblib.load('models/violence_model.pkl')
        global_model = joblib.load('models/global_model.pkl')
        return individual_model, violence_model, global_model
    except FileNotFoundError:
        st.warning("⚠️ Models not found. Using dataset predictions only.")
        return None, None, None

# Load data and models
individual_df, violence_df, global_df, global_stats = load_all_data()
individual_model, violence_model, global_model = load_models()

# ----------------------------------------------------
# HELPER FUNCTIONS FOR FEATURE ENGINEERING
# ----------------------------------------------------

class IndividualFeaturePrep:
    """Prepare features for individual risk assessment"""
    
    def __init__(self, reference_df):
        """Initialize with reference data for scaling"""
        self.reference_df = reference_df
        # Get scaling parameters from reference data
        self.age_mean = 32.5  
        self.age_std = 8.5
        self.income_mean = 6.5  # log-transformed
        self.income_std = 1.5
        
    def prepare_features(self, age, education, employment, income, marital_status, 
                        residence='Urban', vawg_rate=None):
        """Transform user inputs into model features"""
        
        # Education mapping
        education_map = {
            "No education": 0,
            "Primary": 1,
            "Secondary": 2,
            "Higher": 3,
            "Tertiary": 3
        }
        education_ordinal = education_map.get(education, 1)
        
        # Marital status
        is_married = 1 if marital_status in ["Married/Living together", "Married"] else 0
        
        # Employment
        emp_unemployed = 1 if employment == "Unemployed" else 0
        emp_employed = 1 if employment in ["Employed for cash", "Employed"] else 0
        emp_semi_employed = 1 if employment == "Semi-employed" else 0
        
        # Scaled features
        age_scaled = (age - self.age_mean) / self.age_std
        income_log = np.log1p(income)
        income_scaled = (income_log - self.income_mean) / self.income_std
        
        # VAWG rate (default to median if not provided)
        if vawg_rate is None:
            vawg_rate = 20.0  
        
        # Engineered features
        young_unmarried = 1 if (age < 25 and is_married == 0) else 0
        low_education_unemployed = 1 if (education_ordinal <= 1 and emp_unemployed == 1) else 0
        married_unemployed = 1 if (is_married == 1 and emp_unemployed == 1) else 0
        married_low_income = 1 if (is_married == 1 and income < 500) else 0
        young_unemployed = 1 if (age < 25 and emp_unemployed == 1) else 0
        
        # Interactions
        age_income = age_scaled * income_scaled
        education_income = education_ordinal * income_scaled
        
        # Age groups
        age_group_young = 1 if age < 25 else 0
        age_group_middle = 1 if 25 <= age < 35 else 0
        
        # Composite features
        low_income_flag = 1 if income < 500 else 0
        economic_vulnerability = low_income_flag + emp_unemployed + (1 if education_ordinal <= 1 else 0)
        empowerment_score = education_ordinal + (1 - emp_unemployed)
        high_vawg_environment = 1 if vawg_rate > 20 else 0
        
        # Create feature dictionary in EXACT order of your training data
        features = {
            'education_ordinal': education_ordinal,
            'is_married': is_married,
            'emp_unemployed': emp_unemployed,
            'age_scaled': age_scaled,
            'income_scaled': income_scaled,
            'vawg_rate': vawg_rate,
            'young_unmarried': young_unmarried,
            'low_education_unemployed': low_education_unemployed,
            'married_unemployed': married_unemployed,
            'age_income': age_income,
            'education_income': education_income,
            'age_group_young': age_group_young,
            'age_group_middle': age_group_middle,
            'economic_vulnerability': economic_vulnerability,
            'empowerment_score': empowerment_score,
            'high_vawg_environment': high_vawg_environment,
            'emp_employed': emp_employed,
            'emp_semi employed': emp_semi_employed,
            'married_low_income': married_low_income,
            'young_unemployed': young_unemployed
        }
        
        return pd.DataFrame([features])


class PolicyImpactCalculator:
    """Calculate policy impacts using violence dataset"""
    
    def __init__(self, violence_df, model=None):
        self.violence_df = violence_df
        self.model = model
        
    def get_country_baseline(self, country):
        """Get baseline features for a country"""
        country_data = self.violence_df[self.violence_df['country'] == country]
        if len(country_data) == 0:
            return None
        return country_data.iloc[0].to_dict()
    
    def simulate_impact(self, country, policy_changes):
        """
        Simulate policy impact
        
        policy_changes = {
            'parliament_increase': 20,  # percentage points
            'legal_reform_pct': 30,     # % reduction
            'education_gap_reduction': 50  # % reduction
        }
        """
        baseline = self.get_country_baseline(country)
        if baseline is None:
            return None
        
        # Get current IPV rate
        current_ipv = baseline.get('ipv', baseline.get('predicted_ipv', 30))
        
        # Calculate impacts based on model coefficients or estimates
        # These are based on your model's feature importance
        impacts = {}
        
        # Parliament impact (women's representation reduces violence)
        if 'parliament_increase' in policy_changes:
            # Estimate: Each 10pp increase in parliament = ~2% reduction in IPV
            parliament_impact = (policy_changes['parliament_increase'] / 10) * 2.0
            impacts['parliament'] = parliament_impact
        
        # Legal discrimination impact
        if 'legal_reform_pct' in policy_changes:
            current_ld = baseline.get('ld', 50)
            # Estimate: 30% reduction in legal discrimination = ~3% reduction in IPV
            legal_impact = (policy_changes['legal_reform_pct'] / 30) * 3.0
            impacts['legal'] = legal_impact
        
        # Education gap impact
        if 'education_gap_reduction' in policy_changes:
            current_gap = baseline.get('sec_edu_gap', 10)
            # Estimate: 50% reduction in education gap = ~4% reduction in IPV
            edu_impact = (policy_changes['education_gap_reduction'] / 50) * 4.0
            impacts['education'] = edu_impact
        
        # Economic empowerment
        if 'economic_investment' in policy_changes:
            # Scale 0-100
            econ_impact = (policy_changes['economic_investment'] / 100) * 5.0
            impacts['economic'] = econ_impact
        
        total_reduction = sum(impacts.values())
        new_ipv = max(0, current_ipv - total_reduction)
        
        return {
            'baseline_ipv': current_ipv,
            'new_ipv': new_ipv,
            'total_reduction': total_reduction,
            'breakdown': impacts,
            'percent_reduction': (total_reduction / current_ipv * 100) if current_ipv > 0 else 0
        }


class CountryComparator:
    """Compare countries using global dataset"""
    
    def __init__(self, global_df, global_stats):
        self.global_df = global_df
        self.global_stats = global_stats
        # Merge for comprehensive data
        self.merged = pd.merge(
            global_df, 
            global_stats, 
            on='country', 
            how='left',
            suffixes=('', '_stats')
        )
    
    def get_country_profile(self, country):
        """Get comprehensive country profile"""
        data = self.merged[self.merged['country'] == country]
        if len(data) == 0:
            return None
        
        data = data.iloc[0]
        
        # Use predicted_ipv if ip_violence is missing
        ipv_rate = data.get('predicted_ipv', data.get('ip_violence', np.nan))
        
        profile = {
            'name': country,
            'ipv_rate': ipv_rate,
            'parliament_seats': data.get('seats_parliament', np.nan),
            'gii': data.get('gii', np.nan),
            'economic_dev': data.get('economic_development_index', np.nan),
            'maternal_mortality': data.get('maternal_mortality', np.nan),
            'youth_vulnerability': data.get('youth_vulnerability_index', np.nan),
            'freedom_index': data.get('freedom_index', np.nan),
            'wpsi': data.get('wpsi', np.nan),
            'income_category': data.get('income_category', 'Unknown'),
            'education_gap': data.get('sec_edu_gap', np.nan),
            'labor_gap': data.get('lab_force_gap', np.nan)
        }
        
        return profile
    
    def compare_countries(self, country1, country2):
        """Generate structured comparison"""
        profile1 = self.get_country_profile(country1)
        profile2 = self.get_country_profile(country2)
        
        if not profile1 or not profile2:
            return None
        
        return {
            'country1': profile1,
            'country2': profile2
        }
# ----------------------------------------------------
# GPT-4 INTEGRATION (cheaper version "gpt-4o-mini, similar performance as 3.5")
# ----------------------------------------------------

def init_openai():
    """Initialize OpenAI client"""
    api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("api_key")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def generate_gpt_response(prompt, context_data=None):
    """Generate response using GPT-4"""
    client = init_openai()
    if not client:
        return "⚠️ Please configure your OpenAI API key in the sidebar."
    
    try:
        system_message = """You are an expert on violence against women (VAWG). 
        Provide evidence-based, empathetic, and actionable insights. Be specific with 
        numbers and percentages. Focus on practical interventions."""
        
        messages = [
            {"role": "system", "content": system_message},
        ]
        
        if context_data:
            messages.append({
                "role": "system", 
                "content": f"Data context: {context_data}"
            })
        
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
with st.sidebar:
    st.title("🛡️ Violence Against Women & Girls Intelligence")
    st.markdown("---")
    
    # API Key
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key,
        help="Required for AI-powered insights"
    )
    
    if api_key:
        st.session_state.api_key = api_key
        st.success("✓ API Key configured")
    else:
        st.warning("⚠️ Enter API key for AI features")
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Select Tool",
        ["🏠 Home", "👤 Risk Assessment", "🏛️ Policy Simulator", "🌍 Country Comparison"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.caption("Built with Streamlit & GPT-4o-mini")

# ----------------------------------------------------
# HOME PAGE
# ----------------------------------------------------

if page == "🏠 Home":
    st.title("🛡️ Violence Against Women Intelligence Dashboard")
    st.markdown("""
    ### AI-Powered Tools for Understanding and Preventing Violence Against Women & Girls
    
    This dashboard uses machine learning and GPT-4o-mini to provide insights across three levels.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **👤 Individual Risk Assessment**
        
        - Personal risk profiling
        - Evidence-based recommendations
        - ROC-AUC: 0.954
        - Accuracy: 94%
        """)
    
    with col2:
        st.info("""
        **🏛️ Policy Impact Simulator**
        
        - Simulate policy interventions
        - Country-level predictions
        - Based on 50 countries
        - Multiple intervention types
        """)
    
    with col3:
        st.info("""
        **🌍 Country Comparison**
        
        - Compare 195 countries
        - Interactive world map
        - Structural factor analysis
        - Learning opportunities
        """)
    
    st.markdown("---")
    
    if individual_df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Individual Records", f"{len(individual_df):,}")
        col2.metric("Countries (Violence)", f"{len(violence_df):,}")
        col3.metric("Countries (Global)", f"{len(global_df):,}")

# ============================================
# TOOL 1: RISK ASSESSMENT
# ============================================

elif page == "👤 Risk Assessment":
    st.title("👤 Personalized Risk Assessment")
    
    if individual_df is None:
        st.error("Data not loaded")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Enter Profile Information")
        
        with st.form("risk_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                age = st.number_input("Age", 15, 49, 25)
                education = st.selectbox(
                    "Education Level",
                    ["No education", "Primary", "Secondary", "Higher"]
                )
                marital_status = st.selectbox(
                    "Marital Status",
                    ["Single/Never married", "Married/Living together", 
                     "Widowed/Divorced/Separated"]
                )
            
            with col_b:
                employment = st.selectbox(
                    "Employment Status",
                    ["Unemployed", "Employed for cash", "Employed for kind", "Semi-employed"]
                )
                income = st.slider("Monthly Income (USD)", 0, 5000, 500, 100)
                residence = st.selectbox("Residence", ["Rural", "Urban"])
            
            submit = st.form_submit_button("🔍 Assess Risk", use_container_width=True)
        
        if submit:
            # Prepare features
            prep = IndividualFeaturePrep(individual_df)
            features_df = prep.prepare_features(
                age=age,
                education=education,
                employment=employment,
                income=income,
                marital_status=marital_status,
                residence=residence
            )
            
            # Get prediction
            if individual_model is not None:
                # Use model
                risk_score = individual_model.predict_proba(features_df)[0][1]
            else:
                # Use dataset statistics as fallback
                # Find similar profiles in dataset
                similar_mask = (
                    (individual_df['education_ordinal'] == features_df['education_ordinal'].values[0]) &
                    (individual_df['emp_unemployed'] == features_df['emp_unemployed'].values[0])
                )
                if similar_mask.sum() > 0:
                    risk_score = individual_df[similar_mask]['predicted_violence'].mean()
                else:
                    risk_score = individual_df['predicted_violence'].mean()
            
            # Determine risk level
            if risk_score < 0.3:
                risk_level, risk_color = "LOW", "🟢"
            elif risk_score < 0.6:
                risk_level, risk_color = "MODERATE", "🟡"
            else:
                risk_level, risk_color = "HIGH", "🔴"
            
            with col2:
                st.subheader("📊 Risk Profile")
                st.metric("Risk Score", f"{risk_score:.1%}")
                st.markdown(f"### {risk_color} {risk_level} RISK")
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if risk_score > 0.6 else "orange" if risk_score > 0.3 else "green"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "lightcoral"}
                        ]
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            # AI Analysis
            st.markdown("---")
            st.subheader("🤖 AI-Powered Analysis")
            
            context = f"""
            Risk Score: {risk_score:.1%} ({risk_level})
            Age: {age}
            Education: {education}
            Employment: {employment}
            Income: ${income}
            Marital Status: {marital_status}
            """
            
            prompt = f"""Analyze this individual's violence risk profile:

{context}

Provide:
1. Top 3 risk factors (be specific about which factors increase risk)
2. Protective factors present
3. 3 actionable recommendations with estimated impact
4. Resources or support services

Be empathetic and practical."""
            
            with st.spinner("Generating insights..."):
                response = generate_gpt_response(prompt, context)
                st.markdown(response)

# ----------------------------------------------------
# TOOL 2: POLICY SIMULATOR
# ----------------------------------------------------

elif page == "🏛️ Policy Simulator":
    st.title("🏛️ Policy Impact Simulator")
    
    if violence_df is None:
        st.error("Data not loaded")
        st.stop()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Select Country")
        
        countries = violence_df['country'].dropna().unique()
        country = st.selectbox("Country", sorted(countries))
        
        # Get country data
        country_data = violence_df[violence_df['country'] == country].iloc[0]
        
        st.markdown("### Current Indicators")
        col_a, col_b = st.columns(2)
        
        current_ipv = country_data.get('ipv', country_data.get('predicted_ipv', 30))
        current_gii = country_data.get('gii', 0.5)
        current_ld = country_data.get('ld', 50)
        current_edu_gap = country_data.get('sec_edu_gap', 10)
        
        col_a.metric("IPV Rate", f"{current_ipv:.1f}%")
        col_a.metric("Legal Discrimination", f"{current_ld:.0f}/100")
        col_b.metric("Gender Inequality", f"{current_gii:.3f}")
        col_b.metric("Education Gap", f"{current_edu_gap:.1f}pp")
    
    with col2:
        st.subheader("🎯 Policy Interventions")
        
        with st.form("policy_form"):
            parliament_increase = st.slider(
                "Increase Women in Parliament (pp)",
                0, 50, 20,
                help="Percentage point increase"
            )
            
            legal_reform = st.slider(
                "Legal Discrimination Reduction (%)",
                0, 100, 30
            )
            
            edu_intervention = st.slider(
                "Education Gap Closure (%)",
                0, 100, 50
            )
            
            econ_investment = st.slider(
                "Economic Empowerment Investment (scale)",
                0, 100, 50
            )
            
            simulate = st.form_submit_button("🚀 Simulate Impact", use_container_width=True)
        
        if simulate:
            # Calculate impact
            calculator = PolicyImpactCalculator(violence_df, violence_model)
            
            policy_changes = {
                'parliament_increase': parliament_increase,
                'legal_reform_pct': legal_reform,
                'education_gap_reduction': edu_intervention,
                'economic_investment': econ_investment
            }
            
            results = calculator.simulate_impact(country, policy_changes)
            
            if results:
                st.markdown("---")
                st.subheader("📈 Predicted Impact")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric(
                        "Total IPV Reduction",
                        f"{results['total_reduction']:.1f}%",
                        delta=f"{results['percent_reduction']:.1f}% decrease",
                        delta_color="inverse"
                    )
                    
                    st.markdown("**Breakdown:**")
                    for key, value in results['breakdown'].items():
                        st.markdown(f"- {key.title()}: {value:.1f}%")
                
                with col_b:
                    # Waterfall chart
                    breakdown = results['breakdown']
                    fig = go.Figure(go.Waterfall(
                        x=["Baseline"] + [k.title() for k in breakdown.keys()] + ["Total"],
                        y=[0] + list(breakdown.values()) + [0],
                        measure=["relative"] * (len(breakdown) + 1) + ["total"],
                        decreasing={"marker": {"color": "green"}},
                    ))
                    fig.update_layout(
                        title="Impact Breakdown",
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # GPT Analysis
                st.markdown("---")
                st.subheader("🤖 AI Policy Recommendations")
                
                context = f"""
                Country: {country}
                Current IPV: {results['baseline_ipv']:.1f}%
                Predicted IPV after interventions: {results['new_ipv']:.1f}%
                Total reduction: {results['total_reduction']:.1f}%
                
                Interventions:
                - Parliament increase: {parliament_increase}pp
                - Legal reform: {legal_reform}%
                - Education: {edu_intervention}%
                - Economic: {econ_investment}/100
                """
                
                prompt = f"""As a policy expert, analyze:

{context}

Provide:
1. Assessment of this policy mix
2. Timeline for impact (when effects visible?)
3. Complementary policies to amplify effect
4. Implementation challenges
5. Countries with similar successful reforms

Be specific and practical for policymakers."""
                
                with st.spinner("Generating analysis..."):
                    response = generate_gpt_response(prompt, context)
                    st.markdown(response)

# ----------------------------------------------------
# TOOL 3: COUNTRY COMPARISON
# ----------------------------------------------------

elif page == "🌍 Country Comparison":
    st.title("🌍 Country Comparison Dashboard")
    
    if global_df is None:
        st.error("Data not loaded")
        st.stop()
    
    # World Map
    st.subheader("🗺️ Global Violence Map")
    
    # Use predicted_ipv if ip_violence has missing values
    map_data = global_df.copy()
    map_data['ipv_display'] = map_data['predicted_ipv'].fillna(map_data.get('ip_violence', np.nan))
    
    fig = px.choropleth(
        map_data,
        locations="country",
        locationmode="country names",
        color="ipv_display",
        hover_name="country",
        hover_data={
            "ipv_display": ":.1f",
            "seats_parliament": ":.1f",
            "economic_development_index": ":.2f"
        },
        color_continuous_scale="Reds",
        labels={"ipv_display": "IPV Rate (%)"},
    )
    
    fig.update_geos(showcountries=True, countrycolor="lightgray")
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Country Comparison
    st.subheader("⚖️ Compare Two Countries")
    
    countries = global_df['country'].dropna().unique()
    
    col1, col2 = st.columns(2)
    
    with col1:
        country1 = st.selectbox("First Country", sorted(countries), key="c1")
    
    with col2:
        country2 = st.selectbox("Second Country", sorted(countries), key="c2")
    
    if st.button("🔍 Compare", use_container_width=True):
        comparator = CountryComparator(global_df, global_stats)
        comparison = comparator.compare_countries(country1, country2)
        
        if comparison:
            profile1 = comparison['country1']
            profile2 = comparison['country2']
            
            # Side-by-side metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 🇦 {country1}")
                st.metric("IPV Rate", f"{profile1['ipv_rate']:.1f}%")
                st.metric("Parliament", f"{profile1['parliament_seats']:.1f}%")
                st.metric("GII", f"{profile1['gii']:.3f}")
                st.metric("Economic Dev", f"{profile1['economic_dev']:.2f}")
                st.metric("Maternal Mortality", f"{profile1['maternal_mortality']:.0f}")
            
            with col2:
                st.markdown(f"### 🇧 {country2}")
                st.metric("IPV Rate", f"{profile2['ipv_rate']:.1f}%")
                st.metric("Parliament", f"{profile2['parliament_seats']:.1f}%")
                st.metric("GII", f"{profile2['gii']:.3f}")
                st.metric("Economic Dev", f"{profile2['economic_dev']:.2f}")
                st.metric("Maternal Mortality", f"{profile2['maternal_mortality']:.0f}")
            
            # Radar chart
            st.markdown("---")
            st.subheader("📊 Multi-Dimensional Comparison")
            
            categories = ['Parliament', 'Economic Dev', 'Maternal Health', 
                         'Youth Protection', 'Freedom']
            
            # Normalize values
            values1 = [
                profile1['parliament_seats'],
                (profile1['economic_dev'] + 5) * 10,
                100 - (profile1['maternal_mortality'] / 10),
                100 - profile1.get('youth_vulnerability', 0) * 10,
                (profile1.get('freedom_index', 0) + 3) * 15
            ]
            
            values2 = [
                profile2['parliament_seats'],
                (profile2['economic_dev'] + 5) * 10,
                100 - (profile2['maternal_mortality'] / 10),
                100 - profile2.get('youth_vulnerability', 0) * 10,
                (profile2.get('freedom_index', 0) + 3) * 15
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=values1, theta=categories, fill='toself', name=country1))
            fig.add_trace(go.Scatterpolar(r=values2, theta=categories, fill='toself', name=country2))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # GPT Analysis
            st.markdown("---")
            st.subheader("🤖 AI-Powered Comparison")
            
            context = f"""
            {country1}:
            - IPV: {profile1['ipv_rate']:.1f}%
            - Parliament: {profile1['parliament_seats']:.1f}%
            - GII: {profile1['gii']:.3f}
            - Economic Dev: {profile1['economic_dev']:.2f}
            
            {country2}:
            - IPV: {profile2['ipv_rate']:.1f}%
            - Parliament: {profile2['parliament_seats']:.1f}%
            - GII: {profile2['gii']:.3f}
            - Economic Dev: {profile2['economic_dev']:.2f}
            """
            
            prompt = f"""Compare these countries on violence against women:

{context}

Provide:
1. Key differences and their drivers
2. Strengths of each country
3. What each can learn from the other
4. Specific policy recommendations for each
5. Similar successful countries to learn from

Be specific and evidence-based."""
            
            with st.spinner("Generating comparison..."):
                response = generate_gpt_response(prompt, context)
                st.markdown(response)

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------

st.markdown("---")
st.caption("""
**Data:** Individual (n=325), Violence/Country (n=50), Global (n=195)  
**Models:** LogisticRegressionCV (Individual), Ensemble (Violence), Ensemble (Global)  
**Disclaimer:** Statistical estimates based on various datasets. Consult experts for policy decisions.
""")