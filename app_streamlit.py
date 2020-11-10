from datetime import datetime
import altair as alt
import joblib
import pandas as pd
import pytz
import streamlit as st
import os
from invesscience.joanna_clean_data import clean_cat, clean_country, clean_state, clean_date, clean_series\
, clean_participants, clean_amount, clean_rounds, clean_study, clean_school, clean_female,\
clean_founded_before, clean_worked_before, clean_founder

df = pd.read_csv('raw_data/support/companies_test.csv')

st.markdown(f"# Invesscience \n # Invest in the most promising start-ups")

def convert_bool(x):
    if x=="Yes":
        return 1
    else:
        return 0

def colors(x, cut):
    if x < (2/3)*cut:
        return "low"
    elif x > (5/4)*cut:
        return "high"
    else:
        return "average"

def create_chart(column,title):
    #pivot_table
    pivot_categories = pd.pivot_table(df, values='id', index=[column],
                columns=['target'], aggfunc= "count", margins = True)
    pivot_categories = pivot_categories.fillna(0).rename(columns={0:'no_exit', 1:'acquisition_or_ipo'})
    #pivot_categories[0] = pivot_categories[0]/pivot_categories.All
    pivot_categories['acquisition_or_ipo'] = pivot_categories['acquisition_or_ipo']/pivot_categories.All
    cut = pivot_categories.loc["All","acquisition_or_ipo"]
    pivot_categories["likelihood"] = pivot_categories["acquisition_or_ipo"].map(lambda x: colors(x,cut))
    pivot_categories = pivot_categories[['acquisition_or_ipo',"likelihood"]]

    c = alt.Chart(pivot_categories.reset_index()).mark_bar(
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3).encode(
    y=alt.Y("acquisition_or_ipo:Q", title="Chances of IPO or Acquisition [%]"),
    x=alt.X(f"{column}:O", title=title),
    color= "likelihood"
    ).properties(width=700, height=400)
    st.write(c)





def main(reference="a"):
    analysis = st.sidebar.selectbox("choose activity", ["prediction", "data visualization"])
    if analysis == "data visualization":
        st.header("Insights Data Analysis on Crunchbase Dataset")
        #st.markdown("**Have fun immplementing your own Taxifare Dataviz**")

        st.subheader("Exits according to category")
        st.markdown("**network_hosting, web and public relations** are more likely to have an exit")
        st.markdown("**medical and analytics** are less likely to have an exit")
        #st.bar_chart(category_chart['acquisition_or_ipo'])
        create_chart('category_code',"Categories")

        create_chart('state_code',"States")
        create_chart('country_code',"Countries")













    if analysis == "prediction":
        st.markdown("**This model will help you decide whether you should invest in the Series A of a company of your choice.**")

        pipeline = joblib.load('monday_model.joblib')
        print("loaded model")

        st.header("Please tell us more about your investment opportunity")

        COLS = ['category_code', 'country_code', 'state_code', 'founded_at', f'timediff_founded_series_{reference}',
                 f'time_diff_series_{reference}_now', f'participants_{reference}', f'raised_amount_usd_{reference}',
                 'rounds_before_a', 'mean_comp_worked_before', 'founder_count', 'graduate', 'MBA_bool', 'cs_bool', 'top_20_bool', 'mean_comp_founded_before',
                 'female_ratio',
                 ]


        categories = ['advertising',
                         'analytics',
                         'biotech',
                         'cleantech',
                         'consulting',
                         'ecommerce',
                         'enterprise',
                         'finance',
                         'hardware',
                         'medical',
                         'mobile',
                         'music',
                         'network_hosting',
                         'public_relations',
                         'search',
                         'security',
                         'semiconductor',
                         'social',
                         'software',
                         'web',"other_category"]

        #founded_ats = ['pre_1996', '1996', '1997', '1998', '1999', '2000','2001', '2002','2003', '2004', '2005', '2006', '2007', '2008', '2009',
       #'2010', '2011', '2012', '2013', "2014","2015","2016","2017","2018", "2019", "2020", "2021"]

        countries = ['CAN',
                     'CHN',
                     'DEU',
                     'FRA',
                     'GBR',
                     'IND',
                     'ISR',
                     'SWE',
                     'USA',
                     'other_country']
        states = ['not_US', 'CA',
                 'CO',
                 'FL',
                 'IL',
                 'MA',
                 'MD',
                 'NC',
                 'NJ',
                 'NY',
                 'PA',
                 'TX',
                 'VA',
                 'WA',
                 'other_state']

        universities = ['Massachusetts Institute Of Technology (MIT) ',
                         'Stanford University',
                         'Harvard University',
                         'California Institute Of Technology (CALTECH)',
                         'University Of Cambridge',
                         'University Of Oxford',
                         'UCL (University College London)',
                         'Imperial College London',
                         'University Of Chicago',
                         'Eth Zurich (Swiss Federal Institute Of Technology)',
                         'Nanyang Technological University (Ntu)',
                         'Ecole Polytechnique Fédérale De Lausanne (EPFL)',
                         'Princeton University',
                         'Cornell University',
                         'National University Of Singapore (NUS)',
                         'Yale University',
                         'Johns Hopkins University',
                         'Columbia University',
                         'University Of Pennsylvania',
                         'Australian National University (ANU)']




        # inputs from user
        st.subheader("1. Personal behaviour")
        time_diff_series_now =  st.number_input("How long are you ready to wait for returns?",min_value=1,max_value=10 ,value=5)

        st.subheader("2. General Information")
        category_code = st.selectbox("Category", categories, index=0)
        founded_at = st.number_input("Founding year",min_value=1900,value=2018)
        country_code = st.selectbox("Country", countries ,index=8)
        state_code = st.selectbox("State", states , index=1)

        st.subheader("3. Investment rounds information")
        participants = st.number_input("# investors in current round (0 if amount is not know yet)",min_value=0,value=0)
        raised_amount_usd = st.number_input("Raised amount in current round (0 if amount is not know yet)",min_value=0,value=0)
        rounds_before_a = st.number_input("# funding rounds before current",min_value=0,value=0)

        st.subheader("4. Founding team information")
        founder_count = st.number_input("# founders",min_value=0,value=0)
        female_ratio = st.number_input("# female founders",min_value=0,value=0)
        #undergrad = st.radio(label="Any founder with an undergraduate degree?", options=["Yes", "No"])
        graduate = st.radio(label="Any founder with a graduate degree?", options=["Yes", "No"])
        #professional = st.radio(label="Any founder with a professional degree?", options=["Yes", "No"])
        MBA_bool = st.radio(label="Any founder with an MBA?", options=["Yes", "No"])
        cs_bool = st.radio(label="Any founder with a Computer Science degree?", options=["Yes", "No"])
        #phd_bool = st.radio(label="Any founder with a PhD?", options=["Yes", "No"])
        top_20_bool = st.radio(label="Any founder studied in any of the universities below?", options=["Yes", "No"])
        st.write(universities)
        mean_comp_founded_before = st.radio(label="Any founder a founded company before this one?", options=["Yes", "No"])
        mean_comp_worked_before = st.number_input("Average # companies founders worked at before",min_value=0.0,value=1.0, max_value=100.0, step=0.01)


        ## formatting

        investment_datetime = datetime.utcnow().year
        timediff_founded_series = round(investment_datetime - int(founded_at))


        formated_input = {
                #"normalized_name":input["normalized_name"].lower()
                "category_code": clean_cat(category_code),
                "country_code": clean_country(country_code),
                "state_code": clean_state(state_code),
                "founded_at": clean_date(int(founded_at)),
                f"participants_{reference}": clean_participants(int(participants)),
                f'raised_amount_usd_{reference}': clean_amount(int(raised_amount_usd)),
                f'timediff_founded_series_{reference}': timediff_founded_series,
                'rounds_before_a': clean_rounds(int(rounds_before_a)),
                "graduate" : convert_bool(graduate),
                #"undergrad" : convert_bool(undergrad),
                #"professional" : convert_bool(professional),
                "MBA_bool" : convert_bool(MBA_bool),
                "cs_bool" : convert_bool(cs_bool),
                #"phd_bool" : convert_bool(phd_bool),
                "founder_count" : clean_founder(int(founder_count)),
                "top_20_bool" : convert_bool(top_20_bool),
                "female_ratio" : clean_female(int(female_ratio)),
                "mean_comp_founded_before" : convert_bool(mean_comp_founded_before),
                "mean_comp_worked_before" : clean_worked_before(mean_comp_worked_before),
                f"time_diff_series_{reference}_now" : clean_series(time_diff_series_now)
                }



# inputs =        [{
#     "normalized_name":"Facebook",
#     "category_code": "medical",
#     "founded_at": "2012-12-03 13:10:00",
#     "description": "lorem ipsum",
#     "country_code": "USA",
#     "state_code":  "CA",
#     f"participants_{reference}": 1,
#     f"raised_amount_usd_{reference}": 0,
#     f'rounds_before_a': 1,
#     "graduate": 0,
#     "undergrad": 0,
#     "professional": 0,
#     "MBA_bool" : 0,
#     "cs_bool": 1,
#     "phd_bool": 0,
#     "founder_count":4,
#     "top_20_bool": 0,
#     "female_ratio": 1,
#     "mean_comp_founded_before":1,
#     "mean_comp_worked_before": 2,
#     f"time_diff_series_{reference}_now" : 5
#     }]



        X = pd.DataFrame([formated_input])
        print(X.T)
        X = X[COLS]
        if type(reference) != str:
            X = X.drop(columns="rounds_before_a")
        results = pipeline.predict_proba(X)
        print(results)

        st.write(f"The probability for the company to exit within {time_diff_series_now} years is", round(results[0][1],2))
        # st.map(data=data)


# print(colored(proc.sf_query, "blue"))
# proc.test_execute()
if __name__ == "__main__":
    #df = read_data()
    main()
