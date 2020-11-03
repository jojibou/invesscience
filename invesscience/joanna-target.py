import pandas as pd
import numpy as np
import os


def target_ipo(ipos, rounds,companies,reference="a"):
    rounds.funded_at = pd.to_datetime(rounds.funded_at)
    ipos.public_at = pd.to_datetime(ipos.public_at)

    ipos_rounds = ipos[["object_id","public_at"]].merge(rounds, on="object_id", how="inner")

    tmp1 = ipos_rounds[ipos_rounds.funding_round_code == reference].sort_values(by="funded_at")\
    .groupby("object_id",as_index=False).first()
    tmp2 = ipos_rounds[ipos_rounds.funding_round_type == f'series-{reference}']\
    .sort_values(by="funded_at").groupby("object_id",as_index=False).first()
    tmp3 = pd.concat([tmp1,tmp2]).drop_duplicates('object_id')\
    .rename(columns={'funded_at':f"date_series_{reference}"})

    tmp3 = tmp3.drop(columns=["id","funding_round_id","raised_amount","raised_currency_code","is_first_round",\
    "is_last_round","source_url","source_description","created_by","created_at", "updated_at",\
    "post_money_valuation","post_money_currency_code", "pre_money_valuation_usd", "pre_money_currency_code",\
    "post_money_valuation_usd","pre_money_valuation","funding_round_type"])
    tmp3 = tmp3.rename(columns={"object_id":"id","raised_amount_usd":f"raised_amount_usd_{reference}",\
                        "participants":f"participants_{reference}", "public_at":"exit_date"})
    tmp3["funding_round_code"] = reference
    tmp3["exit"]="ipo"

    companies = companies.merge(tmp3, how="inner", on="id") #.sort_values(by="public_at")
    print(companies.head())

    return companies

def target_acq(acq, rounds,companies,reference="a"):
    acq.acquired_at = pd.to_datetime(acq.acquired_at)
    acq = acq.rename(columns={"acquired_object_id":"object_id"})
    rounds.funded_at = pd.to_datetime(rounds.funded_at)

    acq_rounds = acq[["object_id","acquired_at"]].merge(rounds, on="object_id", how="inner")

    tmp1 = acq_rounds[acq_rounds.funding_round_code == reference].sort_values(by="funded_at")\
    .groupby("object_id",as_index=False).first()
    tmp2 = acq_rounds[acq_rounds.funding_round_type == f'series-{reference}']\
    .sort_values(by="funded_at").groupby("object_id",as_index=False).first()
    tmp3 = pd.concat([tmp1,tmp2]).drop_duplicates('object_id')\
    .rename(columns={'funded_at':f"date_series_{reference}"})

    tmp3 = tmp3.drop(columns=["id","funding_round_id","raised_amount","raised_currency_code","is_first_round",\
    "is_last_round","source_url","source_description","created_by","created_at", "updated_at",\
    "post_money_valuation","post_money_currency_code", "pre_money_valuation_usd", "pre_money_currency_code",\
    "post_money_valuation_usd","pre_money_valuation","funding_round_type"])
    tmp3 = tmp3.rename(columns={"object_id":"id","raised_amount_usd":f"raised_amount_usd_{reference}",\
                        "participants":f"participants_{reference}","acquired_at":"exit_date"})
    tmp3["funding_round_code"] = reference
    tmp3["exit"]="acquisition"

    companies = companies.merge(tmp3, how="inner", on="id") #.sort_values(by="acquired_at")
    print(companies.head())

    return companies

def target_no_exit(rounds, companies, reference="a"):
    no_rounds = companies.merge(rounds, on="object_id", how="inner")
    tmp1 = no_rounds[no_rounds.funding_round_code == reference].sort_values(by="funded_at")\
    .groupby("object_id",as_index=False).first()
    tmp2 = no_rounds[no_rounds.funding_round_type == f'series-{reference}']\
    .sort_values(by="funded_at").groupby("object_id",as_index=False).first()
    tmp3 = pd.concat([tmp1,tmp2]).drop_duplicates('object_id')\
    .rename(columns={'funded_at':f"date_series_{reference}"})

    tmp3 = tmp3.drop(columns=["id","funding_round_id","raised_amount","raised_currency_code","is_first_round",\
    "is_last_round","source_url","source_description","created_by","created_at", "updated_at",\
    "post_money_valuation","post_money_currency_code", "pre_money_valuation_usd", "pre_money_currency_code",\
    "post_money_valuation_usd","pre_money_valuation","funding_round_type"])
    tmp3 = tmp3.rename(columns={"object_id":"id","raised_amount_usd":f"raised_amount_usd_{reference}",\
                        "participants":f"participants_{reference}"})
    tmp3["funding_round_code"] = reference
    tmp3["exit"]="no exit"
    tmp3["exit_date"]=np.nan
    print(tmp3.head())

    return tmp3



def get_company_target(ipos, acq, rounds,companies,reference="a"):
    companies_ipo = target_ipo(ipos, rounds,companies,reference).set_index("id")
    companies_acq = target_acq(acq, rounds, companies, reference).set_index("id")
    companies_ipo_acq  = pd.concat([companies_ipo,companies_acq], axis=0)
    rows_exit = companies_ipo_acq.index
    companies_no_exit = companies.set_index('id').drop(rows_exit).reset_index().rename(columns={"id":"object_id"})
    companies_no_exit = target_no_exit(rounds, companies_no_exit,reference).set_index("id")

    companies = pd.concat([companies_ipo_acq,companies_no_exit], axis=0)

    print(companies.sample(20))

    return companies






















ipos = pd.read_csv(os.path.join('..',"raw_data","ipos.csv"))
acq = pd.read_csv(os.path.join('..',"raw_data","acquisitions.csv"))
rounds = pd.read_csv(os.path.join('..',"raw_data","funding-rounds.csv"))
companies = pd.read_csv(os.path.join('..',"raw_data","companies.csv"))

companies=get_company_target(ipos, acq, rounds,companies,reference="a")
#print(companies.head())
print(companies.shape)

