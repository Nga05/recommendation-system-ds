import streamlit as st
from pathlib import Path
import re
import zipfile
import pickle
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

### extract model and dataset
path_to_zip_file_model = Path(__file__).parent/ "models.zip"
directory_to_extract_to_model = Path(__file__).parent/ "models"
path_to_zip_file_dataset = Path(__file__).parent/ "dataset.zip"
directory_to_extract_to_dataset = Path(__file__).parent/ "dataset"

if os.path.exists(directory_to_extract_to_model)==False:
    with zipfile.ZipFile(path_to_zip_file_model, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to_model)

if os.path.exists(directory_to_extract_to_dataset)==False:
    with zipfile.ZipFile(path_to_zip_file_dataset, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to_dataset)


### Load dataset after work tokenize with underthesea
products = pd.read_csv('dataset/dataset/Product_wt.csv', index_col=0)


### load review dataset
review = pd.read_csv('dataset/dataset/Review.csv',lineterminator='\n',index_col=0)


# load ALS model
# ALS_model = spark.read.parquet('models/models/Recommender_System.parquet')
ALS_model = pd.read_parquet('models/models/Recommender_System.parquet',engine='pyarrow')


## load Gensim model 
dict_name = 'models/models/dictionary_gensim.h5'
with open(dict_name, 'rb') as f:
    dictionary_new = pickle.load(f)

tfidf_name = 'models/models/tfidf_gensim.h5'
with open(tfidf_name, 'rb') as f:
    tfidf_new = pickle.load(f)

index_name = 'models/models/index_gensim.h5'
with open(index_name, 'rb') as f:
    index_new = pickle.load(f) 


### load Cosine-similarity model
CBF_model = pd.read_csv('models/models/Content_based_RS.csv')


### Gensim Recommendation define function          
def recommender(view_product, dictionary, tfidf, index, df):
  # convert search words into Sparse Vectors
  pattern = r'[0-9]+|\–|\+|\:|\(|\)|\"|\%|\$|\&|\#|\@|\!|\*|\^|\;|\[|\]|\=|\{|\}|\.|\,|\'|\-|\|\<|\>|\xa0|\n|\…|\≥|\•|\//|\±|\”|\“'
  view_product = view_product.lower().split()
  view_product = [re.sub(pattern,'', e) for e in view_product]
  kw_vector = dictionary.doc2bow(view_product) 
  # similarity calculation
  sim = index[tfidf[kw_vector]]
  # print result
  list_id = []
  list_score = []
  for i in range(len(sim)):
    list_id.append(i)
    list_score.append(sim[i])
  df_result = pd.DataFrame({'id': list_id, 'score': list_score})
  # five highest scores
  five_highest_score = df_result.sort_values(by='score', ascending=False).head(6)
  idToList = list(five_highest_score['id'])
  # products_find = products[products.index.isin(idToList)]
  products_find = df[df.index.isin(idToList)] 
  results = products_find[['index','item_id','name']]
  results = pd.concat([results,  five_highest_score], axis=1).sort_values(by='score', ascending=False)
  return results


### main
def main():
    # First some code.
    st.image('dataset/dataset/tiki_logo.jpg')
    st.title("Recommendation App for TiKi")

    menu = ['Overview', 'CBF - Gensim Model', 'CBF - Cosine Similarity Model', 'CF - ALS Model']
    choice = st.sidebar.selectbox("Menu", menu)
    
    ## Overview
    if choice == 'Overview':
        st.subheader('Overview')
        st.markdown('####### Tiki là một hệ sinh thái thương mại “all in one”, trong đó có tiki.vn, là một website thương mại điện tử đứng top 2 của Việt Nam, top 6 khu vực Đông Nam Á.\nTrên trang này đã triển khai nhiều tiện ích hỗ trợ nâng cao trải nghiệm người dùng và họ muốn xây dựng nhiều tiện ích hơn nữa.\nGiả sử công ty này chưa triển khai Recommender System và bạn được yêu cầu triển khai hệ thống này, bạn sẽ làm gì?')
        st.markdown('##### Sử dụng các mô hình Recommendation System để xây dựng hệ thống đề xuất cho các mặt hàng công nghệ của Tiki:')
        st.markdown('+ Content Based Filtering: Gensim và Consine Similarity model')
        st.markdown('+ Collaborative Filtering: ALS model')
        
        st.markdown('**Danh sách các sản phẩm công nghệ**')
        st.dataframe(products.drop(['index','product_wt'], axis=1).head(10))
        st.write('\n')

        st.markdown('**Thông tin đánh giá của khách hàng**')
        st.dataframe(review.head(10))

    ## Gensim model
    elif choice == 'CBF - Gensim Model':
        st.subheader('Content Based Filtering Recommender System')
        lst_product = products['item_id']
        search_item = st.selectbox('Pick an Item ID',tuple(lst_product))
        # create df_new
        df_new = products.copy(deep=True)
        df_new['product'] = df_new['name'] + df_new['description']
        df_new = df_new.reset_index()
        df_new = df_new[['index','item_id','name','description','url','product_wt', 'brand', 'rating', 'price', 'image']]        
        # num_of_rec = st.sidebar.number_input('Number of items recommended',4,30,7)
        if st.button('Recommend'):
            if search_item is not None:
                product = df_new[df_new.item_id==search_item]
                product_name = products.loc[products.item_id==search_item, 'name'].to_string(index=False)[:40]
                product_viewing = product['product_wt'].to_string(index=False)
                results = recommender(product_viewing, dictionary_new, tfidf_new, index_new, df_new)
                # remove choiced id
                results = results[results.item_id != product_viewing]
                recommend = pd.merge(results, df_new, on='item_id', how='left')
                results =recommend[['item_id', 'brand', 'name_x', 'rating', 'score', 'price', 'url', 'image']]

                st.markdown('**Product is viewing:**')
                st.write(product_name)
                st.markdown('**Products are recommened:**')
                st.write(results)
                st.write('\n')
                st.markdown('**List Recommened Products**')

                # display product image
                imgs = results['image'].tolist()
                names = results['name_x'].tolist()
                prodIds = results['item_id'].tolist()
                urls = results['url'].tolist()
                prices = results['price'].tolist()
                for i in range(results.shape[0]):
                    with st.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(imgs[i], width=300)
                        with col2:    
                            link = '[product link](' + urls[i] + ')'
                            st.markdown(link, unsafe_allow_html=True)
                            st.write('Name: ', names[i])
                            st.write('ID: ', str(prodIds[i]))
                            st.write('Price: '+ str(prices[i]) + ' VND')                

    ## Consine Similarity model
    elif choice == 'CBF - Cosine Similarity Model':
        st.subheader('Content Based Filtering Recommender System')
        lst_product = products['item_id']
        search_item = st.selectbox('Pick an Item ID',tuple(lst_product))
        if st.button('Recommend'):
            if search_item is not None:
                try:
                    result_CBF = CBF_model[CBF_model.product_id == search_item]
                    product_name = products.loc[products.item_id==search_item, 'name'].to_string(index=False)[:40]
                    recommend = pd.merge(result_CBF, products, left_on='recomment_product_id', right_on='item_id', how='left')
                    results =recommend[['product_id','recomment_product_id','name', 'score', 'price', 'url', 'image']]
                except:
                    results = 'Not Found'
                st.markdown('**Product is viewing:**')
                st.write(product_name)
                st.markdown('**Products are recommened:**')
                st.write(results)
                st.write('\n')
                st.markdown('**List Recommened Products**')

                # display product image
                imgs = results['image'].tolist()
                names = results['name'].tolist()
                prodIds = results['recomment_product_id'].tolist()
                urls = results['url'].tolist()
                prices = results['price'].tolist()
                for i in range(results.shape[0]):
                    with st.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(imgs[i], width=300)
                        with col2:    
                            link = '[product link](' + urls[i] + ')'
                            st.markdown(link, unsafe_allow_html=True)
                            # st.write('View detail: ', urls[i])                    
                            st.write('Name: ', names[i])
                            st.write('ID: ', str(prodIds[i]))
                            st.write('Price: '+ str(prices[i]) + ' VND')
                  
    ## ALS model
    else:
        st.subheader('Collaborative Filtering Recommender System')
        lst_user = review['customer_id']
        search_item = st.selectbox('Pick an User ID',tuple(lst_user))
        # num_of_rec = st.sidebar.number_input('Number of items recommended',4,30,7)
        if st.button('Recommend'):
            if search_item is not None:
                customer_id = search_item
                find_result = ALS_model[ALS_model['customer_id']==customer_id]
                product_lst = []
                rating_lst = []
                customer_lst = []
                result = pd.DataFrame()
                for i in find_result['recommendations'].values:
                    for elem in i:
                        product_lst.append(elem['product_id'])
                        rating_lst.append(elem['rating'])
                        customer_lst.append(find_result['customer_id'].values[0])
                result['customer_id'] = customer_lst
                result['product_id'] = product_lst
                result['rating'] = rating_lst
                result.sort_values(by='rating', axis=0, ascending=False, inplace=True)

                st.text('Danh sách 10 sản phẩm được đề xuất có xếp hạng cao nhất')
                st.table(result)

if __name__ == '__main__':
    main()

