import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="H5216 Stars Improvement Dashboard", layout="wide")

# 1. Load in Data for H5216 Members
@st.cache_data
def load_member_data():
    member_data = pd.read_csv('preventive_visit_gap_scores.csv')
    member_data_h5216 = member_data[member_data['contract'] == 'H5216'].reset_index(drop=True)
    return member_data_h5216

# 2. Load in Data for H5216 Stars, Scores and Cutpoints
@st.cache_data
def load_measure_data():
    measure_data_h5216 = pd.read_csv('measure_scores_and_cutpoints.csv')
    measure_data_h5216.rename(columns={'Measure': 'measure'}, inplace=True)
    measure_data_h5216['Score'] = measure_data_h5216['Score'].str.replace('%', '').astype(float)
    measure_data_h5216['5 Star Cut Point'] = measure_data_h5216['5 Star Cut Point'].str.replace('%', '').astype(float)
    measure_data_h5216['4 Star Cut Point'] = measure_data_h5216['4 Star Cut Point'].str.replace('%', '').astype(float)
    measure_data_h5216['3 Star Cut Point'] = measure_data_h5216['3 Star Cut Point'].str.replace('%', '').astype(float)
    measure_data_h5216['2 Star Cut Point'] = measure_data_h5216['2 Star Cut Point'].str.replace('%', '').astype(float)
    measure_data_h5216.drop(columns='1 Star Cut Point', inplace=True)
    return measure_data_h5216

member_data_h5216 = load_member_data()
measure_data_h5216 = load_measure_data()

# 3. Define the constants
BASE_REVENUE = 1_341_920_160
BASE_STARS = 4.07  # Original base stars rating
BASE_WEIGHT_SUM = 69
NON_CHANGEABLE_MEASURES_SCORE = 295
REVENUE_INCREASE_RATE = 0.075  # 0.075% revenue increase per 0.01 increase in stars

# 4. Specify the ATEs for each characteristic
treatment_effects = {
    'veteran_ind': 0.1012,               # 10.12%
    'unattributed_provider': 0.0786,     # 7.86%
    'disabled_ind': 0.0756,              # 7.56%
    'is_healthy': 0.2056                 # 20.56%
}

# 5. Define which users are influenced by which tactics
tactics = {
    'unattributed_provider': {'Incentivized PCP Sign Up', 'Provider Incentive Programs', 'Provider Opportunity Reports'},
    'is_healthy': {'Telehealth Incentive Program for First PCP Visit', 'Provider Incentive Programs', 'Provider Opportunity Reports'},
    'disabled_ind': {'Accessibility - Transportation', 'Accessibility - Homecare Program'},
    'veteran_ind': {'Veteran Education Initiatives'}
}

# 6. Tactic costs
tactics_costs = {
    'Incentivized PCP Sign Up': 1_215_120,
    'Provider Incentive Programs': 2_256_695,
    'Provider Opportunity Reports': 3_000_000,
    'Telehealth Incentive Program for First PCP Visit': 927_810,
    'Accessibility - Transportation': 3_723_475,
    'Accessibility - Homecare Program': 912_000,
    'Veteran Education Initiatives': 343_900,
}

# Sidebar for tactic selection
st.sidebar.title("Select Tactics")
available_tactics = list(tactics_costs.keys())

# Display tactics with their costs and checkboxes
st.sidebar.markdown("### Available Tactics and Costs")
chosen_tactics = []

for tactic in available_tactics:
    cost = tactics_costs[tactic]
    selected = st.sidebar.checkbox(f"{tactic}: ${cost:,.2f}", value=False)
    if selected:
        chosen_tactics.append(tactic)

# 7. Log the costs for the chosen tactics
Costs = sum([tactics_costs[tactic] for tactic in chosen_tactics])

# 8. Calculate the conditional ATE based on chosen tactics
# Calculate the adjusted ATE for each group
adjusted_ATEs = {}
for group, effect in treatment_effects.items():
    # Get the tactics relevant to this group
    relevant_tactics = tactics[group]
    # Count how many of the relevant tactics are chosen
    chosen_relevant_tactics = [tactic for tactic in relevant_tactics if tactic in chosen_tactics]
    # Adjust the ATE based on the proportion of chosen tactics
    if chosen_relevant_tactics:
        adjusted_ATEs[group] = effect * (len(chosen_relevant_tactics) / len(relevant_tactics))
    else:
        adjusted_ATEs[group] = 0  # If no tactics are chosen, set ATE to 0

# 9. Calculate the new scores
def calculate_post_intervention_score(row):
    # Initialize original score
    score = row['score']
    # Calculate counterfactual score for each binary characteristic
    for col in adjusted_ATEs.keys():
        if row[col] == 1:  # Only apply effect if the individual falls into the group
            effect = adjusted_ATEs[col]
            score = score * (1 - effect)  # Apply joint probability effect
    return score

# Create a new column for post-intervention score
member_data_h5216['post_intervention_score'] = member_data_h5216.apply(calculate_post_intervention_score, axis=1)

# 10. Filter out members who move from >= 0.5 to < 0.5 probability to miss a visit
effective_intervention_members_h5216 = member_data_h5216[
    (member_data_h5216['score'] > 0.5) & (member_data_h5216['post_intervention_score'] <= 0.5)
]

# 11. Calculate the proportionate increase in measure compliances
measures = [
    'C01: Breast Cancer Screening',
    'C06: Care for Older Adults - Medication Review',
    'C07: Care for Older Adults - Pain Assessment',
    'C02: Colorectal Cancer Screening',
    'C11: Controlling High Blood Pressure',
    'C10: Diabetes Care - Blood Sugar Controlled',
    'C09: Diabetes Care - Eye Exam',
    'D10: Medication Adherence for Cholesterol (Statins)',
    'D08: Medication Adherence for Diabetes Medications',
    'D09: Medication Adherence for Hypertension (RAS antagonists)',
    'C14: Medication Reconciliation Post-Discharge',
    'C08: Osteoporosis Management in Women who had a Fracture',
    'C16: Statin Therapy for Patients with Cardiovascular Disease',
    'D12: Statin Use in Persons with Diabetes (SUPD)'
]

# Create an empty dictionary to store results
proportional_increases = {}

# Loop through each measure column
for measure in measures:
    # Total number of members eligible for the measure
    eligible_count = member_data_h5216[member_data_h5216[measure] == 1].shape[0]
    # Number of eligible members who were affected by the intervention
    successfully_intervened_count = effective_intervention_members_h5216[
        effective_intervention_members_h5216[measure] == 1
    ].shape[0]
    # Calculate the proportional increase
    proportional_increase = (successfully_intervened_count / eligible_count) * 100 if eligible_count > 0 else 0
    # Store the results in the dictionary
    proportional_increases[measure] = proportional_increase

# Convert the dictionary to a DataFrame
proportional_increases_df = pd.DataFrame(list(proportional_increases.items()), columns=['measure', 'proportional_increase'])

# 13. Append missing measures and fill missing values
all_measures = measure_data_h5216['measure'].unique()
missing_measures = set(all_measures) - set(proportional_increases_df['measure'])
missing_df = pd.DataFrame({'measure': list(missing_measures), 'proportional_increase': np.nan})
proportional_increases_df = pd.concat([proportional_increases_df, missing_df], ignore_index=True)

# Fill missing proportional increases with the mean
mean_increase = proportional_increases_df['proportional_increase'].mean()
proportional_increases_df['proportional_increase'].fillna(mean_increase, inplace=True)
proportional_increases_df['proportional_increase'] = proportional_increases_df['proportional_increase'].round(0)

# 14. Merge with measure data
measure_data_h5216 = measure_data_h5216.merge(proportional_increases_df, on='measure', how='left')

# 15. Calculate new scores and new star ratings
measure_data_h5216['New Score'] = measure_data_h5216['Score'] + measure_data_h5216['proportional_increase']
measure_data_h5216['New Score'] = measure_data_h5216['New Score'].clip(upper=99)

def calculate_new_star_rating(row):
    if row['New Score'] >= row['5 Star Cut Point']:
        return 5
    elif row['New Score'] >= row['4 Star Cut Point']:
        return 4
    elif row['New Score'] >= row['3 Star Cut Point']:
        return 3
    elif row['New Score'] >= row['2 Star Cut Point']:
        return 2
    else:
        return 1

measure_data_h5216['New Star Rating'] = measure_data_h5216.apply(calculate_new_star_rating, axis=1)

# 16. Calculate new overall stars
total_weight = BASE_WEIGHT_SUM + measure_data_h5216['Weight'].sum()
weighted_stars = NON_CHANGEABLE_MEASURES_SCORE + (measure_data_h5216['New Star Rating'] * measure_data_h5216['Weight']).sum()
new_stars_rating = weighted_stars / total_weight
new_stars_rating = round(new_stars_rating, 2)

# 17. Calculate expected impact on revenue
stars_increase = new_stars_rating - BASE_STARS
if abs(stars_increase) > 1e-20:
    revenue_increase_percentage = stars_increase * 100 * REVENUE_INCREASE_RATE
    revenue_impact = BASE_REVENUE * (revenue_increase_percentage / 100)
else:
    revenue_impact = 0.0

# 18. Calculate ROI
ROI = ((revenue_impact - Costs) / Costs) * 100 if Costs != 0 else 0.0

# Main dashboard
st.markdown("<h1 style='text-align: center;'>H5216 Stars Improvement Dashboard</h1>", unsafe_allow_html=True)

# Display the highlighted metrics at the top
st.markdown("<h2 style='text-align: center;'>Key Performance Indicators</h2>", unsafe_allow_html=True)

# Display the highlighted metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("New Stars Rating", f"{new_stars_rating}")
col2.metric("Expected Revenue Uplift", f"${revenue_impact:,.2f}")
col3.metric("Total Costs", f"${Costs:,.2f}")
col4.metric("ROI", f"{ROI:.2f}%")

# Display the measure data
st.header("Measure Performance")
st.dataframe(measure_data_h5216[['measure', 'Score', 'New Score', 'New Star Rating', 'Weight']])

# Optional: Display tactic costs whether or not they are chosen
#st.header("Tactic Costs")
#tactics_df = pd.DataFrame.from_dict(tactics_costs, orient='index', columns=['Cost'])
#tactics_df.reset_index(inplace=True)
#tactics_df.rename(columns={'index': 'Tactic'}, inplace=True)
#tactics_df['Chosen'] = tactics_df['Tactic'].apply(lambda x: 'Yes' if x in chosen_tactics else 'No')
#st.dataframe(tactics_df[['Tactic', 'Cost', 'Chosen']])

# Footer
st.markdown("---")
st.markdown("© 2023 Healthcare Analytics Dashboard")
