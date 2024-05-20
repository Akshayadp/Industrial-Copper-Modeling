
import streamlit as st
from streamlit_option_menu import option_menu
import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pickle

# page config in this project.
def set_page_config():
    st.set_page_config(
        page_title="INDUSTRIAL COPPER MODELING",
        layout="wide"
    )

set_page_config()
st.markdown("<h1 style='text-align: center; '>INDUSTRIAL COPPER MODELING</h1>", unsafe_allow_html=True)


# selecting option menu
selected = option_menu(None, ["HOME", "PRICE PREDICTION", "STATUS"],
                       default_index=0,
                       orientation="horizontal",
                       styles={"nav-link": { 
                            "text-align": "center", "margin": "0px",
                                           "--hover-color": "#6495ED"},
                               "nav-link-selected": {"background-color": "#93cbf2"}})
text_process = st.expander("Text Processing", expanded=False)

# Home page 
if selected=='HOME':
    st.write("""
    The Industrial Copper Modeling project addresses challenges in the copper industry related to sales pricing and lead classification. 
    The project leverages machine learning techniques to predict selling prices of copper products and classify leads into 'Won' or 'Lost' 
    categories. This project details the steps taken to preprocess data, build models, and create a Streamlit application for interactive predictions.""")

    st.write('### TECHNOLOGY USED')
    st.write('- PYTHON   (PANDAS, NUMPY)')
    st.write('- SCIKIT-LEARN')
    st.write('- DATA PREPROCESSING')
    st.write('- EXPLORATORY DATA ANALYSIS')
    st.write('- STREAMLIT')


# Price Prediction
if selected=='PRICE PREDICTION':
        
        # Load the necessary files
        #with open('country.pkl', 'rb') as file:
        #    encode_country = pickle.load(file)
        with open('status.pkl', 'rb') as file:
            encode_status = pickle.load(file)
        with open('item type.pkl', 'rb') as file:
            encode_item = pickle.load(file)
        with open('scaling.pkl', 'rb') as file:
            scaled_data = pickle.load(file)
        with open('Extratreeregressor.pkl', 'rb') as file:
            trained_model = pickle.load(file)


        #item_list=['W', 'S', 'Others', 'PL', 'WI', 'IPL']
        status_list=['Won', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised','Offered', 'Offerable']
        country_list=['28', '32', '38', '78', '27', '30', '25', '77', '39', '40', '26', '84', '80', '79','113', '89']
        application_list=[10, 41, 28, 59, 15, 4, 38, 56, 42, 26, 27, 19, 20, 66,
                          29, 22, 40, 25, 67, 79, 3, 99, 2, 5,39, 69, 70, 65, 58, 68]

        product_list=[1670798778, 1668701718, 628377, 640665, 611993, 1668701376,
                      164141591, 1671863738, 1332077137,     640405, 1693867550, 1665572374,
                      1282007633, 1668701698, 628117, 1690738206, 628112, 640400,
                      1671876026, 164336407, 164337175, 1668701725, 1665572032, 611728,
                      1721130331, 1693867563, 611733, 1690738219, 1722207579, 929423819,
                      1665584320, 1665584662, 1665584642]
        st.write(
            '##### ***<span>Fill all the fields and Press the :red[predicted price]  </span>***',
            unsafe_allow_html=True)

        c1,c2,c3=st.columns([2,2,2])
        with c1:
            quantity=st.number_input(label='Enter Quantity in tons   (Min:611728 & Max:1722207579)')
            thickness = st.number_input(label='Enter Thickness (Min:0.18 & Max:400)')
            width = st.number_input(label='Enter Width  (Min:1 & Max:2990)')


        with c2:
            country = country= st.number_input(label='Country Code')
            status = st.number_input(label='Status (Min:0.0 & Max:8.0)')
            item = st.number_input(label='Item Type (Min:1 & Max:6)')

        with c3:
            application = st.selectbox('Application Type', application_list)
            product = st.selectbox('Product Reference', product_list)
            item_order_date = st.date_input("Order Date", datetime.date(2020, 1, 1))
            item_delivery_date = st.date_input("Estimated Delivery Date", datetime.date(2020, 1, 1))
        with c1:
            for _ in range(3):
                st.write(" ")
            if st.button('PREDICT PRICE'):
                try:
                    data = []

                #    transformed_country = encode_country.transform([country])
                #    encoded_ct = transformed_country[0]

                    transformed_status = encode_status.transform([status])
                    encode_st = transformed_status[0]

                    transformed_item = encode_item.transform([item])
                    encode_it = transformed_item[0]

                    order = datetime.datetime.strptime(str(item_order_date), "%Y-%m-%d")
                    delivery = datetime.datetime.strptime(str(item_delivery_date), "%Y-%m-%d")
                    day = (delivery - order).days

                    data.append(float(quantity))
                    data.append(float(thickness))
                    data.append(float(width))
                    data.append(country)
                    data.append(encode_st)
                    data.append(encode_it)
                    data.append(application)
                    data.append(product)
                    data.append(day)

                    x = np.array(data).reshape(1, -1)
                    pred_model = scaled_data.transform(x)
                    price_predict = trained_model.predict(pred_model)
                    predicted_price = str(price_predict)[1:-1]
                    st.write(f'Predicted Selling Price : :green[₹] :green[{predicted_price}]')

                except ValueError as e:
                    st.error(f"Input error: {e}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        


# Status (WON/LOST)
if selected == 'STATUS':

    #with open('country.pkl', 'rb') as file:
    #    encode_country_cls = pickle.load(file)
    with open('item type.pkl', 'rb') as file:
        encode_item_cls = pickle.load(file)
    with open('scaling_classification.pkl', 'rb') as file:
        scaled_data_cls = pickle.load(file)
    with open('randomforest_classification.pkl', 'rb') as file:
        trained_model_cls = pickle.load(file)


    #item_list_cls = ['W','S','Others','PL','WI','IPL']
    country_list_cls = ['28','32','38','78','27','30','25','77','39','40','26','84','79','113','89']
    application_list_cls = [10,41,28,59,15,4,38,56,42,26,27,19,20,66,29,22,40,25,67,79,3,99,2,5,39,69,70,65,58,68]
    product_list_cls = [1670798778, 1668701718, 628377, 640665, 611993, 1668701376,164141591, 1671863738, 1332077137, 640405, 1693867550, 1665572374,
                        1282007633, 1668701698, 628117, 1690738206, 628112, 640400,1671876026, 164336407, 164337175, 1668701725, 1665572032, 611728,
                        1721130331, 1693867563, 611733, 1690738219, 1722207579, 929423819,1665584320, 1665584662, 1665584642]
    st.write('#### ***Fill all the fields and press the predict status to check :red[WON / LOST] </span>***',unsafe_allow_html=True)
    cc1, cc2, cc3 = st.columns([2, 2, 2])
    with cc1:
        quantity_cls = st.number_input(label='Enter Quantity  (Min:611728 & Max:1722207579) in tons')
        thickness_cls = st.number_input(label='Enter Thickness (Min:0.18 & Max:400)')
        width_cls= st.number_input(label='Enter Width  (Min:1, Max:2990)')

    with cc2:
        selling_price_cls= st.number_input(label='Enter Selling Price  (Min:1, Max:100001015)')
        item_cls = st.number_input(label='Item Type (Min:1 & Max:6)')
        country_cls= st.selectbox('Country Code', country_list_cls)

    with cc3:
        application_cls = st.selectbox('Application Type', application_list_cls)
        product_cls = st.selectbox('Product Reference', product_list_cls)
        item_order_date_cls = st.date_input("Order Date", datetime.date(2023, 1, 1))
        item_delivery_date_cls = st.date_input("Estimated Delivery Date", datetime.date(2023,1, 1))
    with cc1:
        for _ in range(3):
            st.write(" ")
        if st.button('PREDICT STATUS'):
            try:
                data_cls = []

            #    transformed_country_cls = encode_country_cls.transform([country_cls])
            #    encoded_ct_cls = transformed_country_cls[0]

                transformed_item_cls = encode_item_cls.transform([item_cls])
                encoded_it_cls = transformed_item_cls[0]

                order_cls = datetime.datetime.strptime(str(item_order_date_cls), "%Y-%m-%d")
                delivery_cls = datetime.datetime.strptime(str(item_delivery_date_cls), "%Y-%m-%d")
                day_cls = (delivery_cls - order_cls).days

                data_cls.append(float(quantity_cls))
                data_cls.append(float(thickness_cls))
                data_cls.append(float(width_cls))
                data_cls.append(float(selling_price_cls))
                data_cls.append(country_cls)
                data_cls.append(encoded_it_cls)
                data_cls.append(application_cls)
                data_cls.append(product_cls)
                data_cls.append(day_cls)

                x_cls = np.array(data_cls).reshape(1, -1)
                scaling_model_cls = scaled_data_cls.transform(x_cls)
                pred_status = trained_model_cls.predict(scaling_model_cls)

                if pred_status == 6:  
                    st.write(f'Predicted Status : :green[WON]')
                else:
                    st.write(f'Predicted Status : :red[LOST]')
            except ValueError as e:
                st.error(f"Input error: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")