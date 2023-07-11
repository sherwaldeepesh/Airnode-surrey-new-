import streamlit as st

# print Tree



def app():
    import leafmap.foliumap as leafmap
    import json
    import pandas as pd
    import requests

    import warnings

    import matplotlib.pyplot as plt
    import seaborn as sns
    # import sm as sm
    from datetime import datetime
    from statsmodels.tsa.arima_model import ARIMA
    import statsmodels.api as sm
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import math
    import itertools


    st.title("AirNode")

    st.header("PM2.5 particles Analysis over Different Locations")
    # st.write(
    #     "Test 1"
    # )

    data = r"data_details/india_details.csv"

    df = pd.read_csv(data)

    # st.header("Example")

    selected_location = st.selectbox("Select Location", df['location'])

    m = leafmap.Map(tiles = 'cartodbpositron', control_scale=True)

    df_updated = df[df['location'] == selected_location]
    
    m.add_circle_markers_from_xy(df_updated, x="longitude", y="latitude", radius=10, color="blue", fill_color="black")
    m.set_center(df_updated['logitude'].values[0] , df_updated['latitude'].values[0], zoom=10)
    m.to_streamlit(width=700, height=500)

    location_id_ = int(df_updated['location_id'].values[0])
    print(location_id_)
    def get_data_by_location(l_id = 8039, date_from = '2023-02-21', date_to = '2023-02-31'):
        return_dict = []
        url = f"https://api.openaq.org/v2/measurements?location_id={l_id}&parameter=pm25&date_from={date_from}&date_to={date_to}&limit=20000"
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)
        # print(response.text)
        measurements_list = json.loads(response.text)
        return_dict.extend(measurements_list['results'])
        
        return return_dict


    date_start = '2023-04-01'
    date_end = '2023-05-31'

    data_list = get_data_by_location(l_id = location_id_, date_from = date_start, date_to = date_end)

    sample_dataframe = pd.DataFrame.from_dict(data_list)




    color = sns.color_palette()
    # print('Please wait. Importing data...')
    # df = pd.read_csv("data.csv", encoding = "ISO-8859-1")
    # print('import completed.')

    ## Data Preprocessing
    sample_dataframe['date_time'] = pd.to_datetime(sample_dataframe['date'].apply(lambda x: x['utc']))

    def replace_unknown(x):
        if x == -999:
            return None
        else:
            return x

    sample_dataframe['value'] = sample_dataframe['value'].apply(lambda x : replace_unknown(x))

    sample_dataframe['value'] = sample_dataframe['value'].fillna(method="ffill")

    new_data_sample = sample_dataframe.loc[:,['date_time','value']]

    new_data_sample = new_data_sample.set_index('date_time').resample('60min').mean()

    # new_data_sample.plot(figsize=[15, 8])
    # plt.xlabel("Date Time")
    # plt.ylabel("PM2.5 CONCENTRATION")
    # plt.title("PM2.5 CONCENTRATION OF Mumbai")
    # plt.show()


    ## finding pdq values

    # Define the p, d and q parameters to take any value between 0 and 2
    # p = d = q = range(0, 2)

    # # Generate all different combinations of p, q and q triplets
    # pdq = list(itertools.product(p, d, q))
        
    # # Generate all different combinations of seasonal p, q and q triplets  for a day hacing 24 hours
    # seasonal_pdq = [(x[0], x[1], x[2], 24) for x in list(itertools.product(p, d, q))]

    warnings.filterwarnings("ignore") # specify to ignore warning messages

    # df_aic = pd.DataFrame(columns=['aic', 'param', 'seasonal_param'])
    # i = 0
    # for param in pdq:
    #     for param_seasonal in seasonal_pdq:
    #         try:
    #             mod = sm.tsa.statespace.SARIMAX(new_data_sample,
    #                                             order=param,
    #                                             seasonal_order=param_seasonal,
    #                                             enforce_stationarity=False,
    #                                             enforce_invertibility=False)

    #             results = mod.fit()

    #             #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
    #             df_aic.loc[i] = [results.aic, param, param_seasonal]
    #             i+=1
    #         except:
    #             continue

    # df_aic = df_aic.sort_values(by='aic', ascending=1)
    # print(df_aic)

    mod = sm.tsa.statespace.SARIMAX(new_data_sample,
                                    order=(1, 0, 1),
                                    seasonal_order=(0, 1, 1, 24),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()

    # print(results.summary().tables[1])

    # results.plot_diagnostics(figsize=(15, 12))
    # plt.show()


    #Update values for beloww


    pred = results.get_prediction(start=pd.to_datetime('2023-05-01 00:00:00+00:00'), dynamic=False)
    pred_ci = pred.conf_int()


    ax = new_data_sample.plot(figsize=[15, 8], label='observed')
    pred.predicted_mean.plot(figsize=[15, 8], ax=ax, label='One-step ahead Forecast', alpha=.7)


    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date Time')
    ax.set_ylabel('PM2.5 Levels')
    plt.legend()

    new_data_sample_forecasted = pred.predicted_mean
    new_data_sample_truth = new_data_sample['2023-05-01 00:00:00+00:00':]

    new_data_sample_truth = new_data_sample_truth.resample('60min').mean()

    # correct values and then try to find rmse again

    # Compute the mean square error
    # rmse = (((new_data_sample_forecasted - new_data_sample_truth) ** 2).mean()) ** 0.5
    # print('The Root Mean Squared Error of our prediction is {}'.format(round(rmse, 2)))

    forecast = results.forecast(50)
    # print(forecast)
    forecast.plot(figsize=[15, 8], color='green', label='future predictions')
    plt.legend()
    # plt.show()

    st.pyplot(plt)
