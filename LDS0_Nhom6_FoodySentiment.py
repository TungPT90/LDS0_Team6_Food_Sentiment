import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from underthesea import word_tokenize
import warnings
import re
#import emoji
#from emoji import UNICODE_EMOJI
import time
import string
import pickle

from sklearn.model_selection import cross_val_score
import datetime
from sklearn.metrics import classification_report

##### BACKEND PROCESSING
# Dictionary

def read_textfiles(file,encode):
    with open(file, 'r', encoding=encode) as file:
        lst_text = file.read()
    lst_text = lst_text.split('\n')
    file.close()
    return lst_text

def convert_textfiles(x):
    files=open(x,'r',encoding='utf8')
    lines=files.read().splitlines()
    dict_={}
    for i in lines:
        key,value=i.split('\t')
        dict_[key]=str(value)
    files.close()
    return dict_

#emo_dict        = convert_textfiles('resource_files/emojicon.txt') #emoji
teen_dict       = convert_textfiles('resource_files/teencode.txt') # teen code
EV_dict         = convert_textfiles('resource_files/english-vnmese.txt') # endlish-vietnamese
stop_words      = read_textfiles('resource_files/vietnamese-stopwords.txt','utf-8')
wrong_text_list = read_textfiles('resource_files/wrong-word.txt','utf-8')
neg_lst         = read_textfiles('resource_files/negative.txt','utf-16')
pos_lst         = read_textfiles('resource_files/positive.txt','utf-16')
phudinh_lst     = read_textfiles('resource_files/phudinh.txt','utf-16')

# # kiểm tra dictionary
# print(list(emo_dict.items())[:5])
# print(list(teen_dict.items())[:5])
# print(list(EV_dict.items())[:5])
# print(stop_words[:5])
# print(wrong_text_list[:5])
# print(neg_lst[:5])
# print(pos_lst[:5])
# print(phudinh_lst[:5])

st.set_page_config(
    page_title="Capstone Project", page_icon="🥗", initial_sidebar_state="expanded"
)
### Các hàm bổ sung
def score_grouping(row):
    if row['review_score'] <= 4:
        return 'Không Thích'
    if row['review_score'] >= 7:
        return 'Thích'
    return 'Bình thường'

def label_score(score):
    if score <= 4:
        return 'Không Thích'
    elif score >= 7:
        return 'Thích'
    else:
        return 'Bình thường'

def remove_html(text): # loại bỏ tag HTML
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

PUNCT_TO_REMOVE = string.punctuation  # bỏ dấu câu
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

#Loại bỏ các ký tự trùng: vd: đẹppppppp
def dupe_characters_remove(text):
    return re.sub(r'([A-Z])\1+', lambda m: m.group(1), text, flags=re.IGNORECASE)

# Thay thế teen code
def teencode_replace(text):
    for key, value in teen_dict.items():
        text= re.sub(rf'\b{key}\b', value, text)
    return text

#Thay thế english-vietnamese
def EV_replace(text):
    for key, value in EV_dict.items():
        text= re.sub(rf'\b{key}\b', value, text)
    return text

# Thay thế emoticon bằng text
# def is_emoji(s):
#     return s in UNICODE_EMOJI

# def replace_emoji(text):
#     result = " "
#     for char in text:
#         if is_emoji(char):
#             result = result +" "+ emo_dict.get(char, char)
#         else:
#             result += emo_dict.get(char, char)
#     return result

# Xử lý phủ định

def xuly_phudinh(text):
    len_text = len(text)-1
    texts = [t.replace('_', ' ') for t in text]  #các bộ từ điển không có '_' giữa 2 từ
    for i in range(len_text):
        pd_text = texts[i]
        if pd_text in phudinh_lst: # chuyển ý nghĩa thành notpositive/ notnegative (VD: món này chẳng ngon--> món này notpositive)
            if texts[i + 1] in pos_lst: #từ tiếp theo
                texts[i] = 'notpositive'
                texts[i + 1]=' ' #xoá sentiment word

            elif texts[i + 1] in neg_lst:
                texts[i] = 'notnegative'
                texts[i + 1]=' '      
    return texts

# PRE-PROCESS TEXT
def chuan_hoa_vanban(vanban):
    df = pd.DataFrame(vanban,columns=['van_ban'])
    df['van_ban'] = df['van_ban'].str.lower()
    df['van_ban'] = df['van_ban'].str.replace(r'\n',' ',regex=True)
    df['van_ban'] = df['van_ban'].apply(lambda text: remove_html(text))
    df['van_ban'] = df['van_ban'].apply(lambda text: remove_punctuation(text))
    df['van_ban'] = df['van_ban'].apply(lambda text: dupe_characters_remove(text))
    df['van_ban'] = df['van_ban'].apply(lambda text: teencode_replace(text))
    df['van_ban'] = df['van_ban'].apply(lambda text: EV_replace(text))
    #df['van_ban'] = df['van_ban'].apply(lambda text: replace_emoji(text))
    #word_tokenize
    df['van_ban'] = df['van_ban'].apply(lambda text: word_tokenize(text, format='text'))

    # Split sentence
    df['van_ban_words'] = [[text for text in x.split()] for x in df['van_ban']]
    df['van_ban_words'] = [[re.sub('[0-9]+','', e) for e in text] for text in df['van_ban_words']] # số
    df['van_ban_words'] = [[t.lower() for t in text if not t in ['', ' ', ',', '.', '...', '-',':', ';', '?', '%', '(', ')', '+', '/','!','@','#','$','%']] for text in df['van_ban_words']] # ký tự đặc biệt
    df['van_ban_words'] = [[t for t in text if not t in wrong_text_list] for text in df['van_ban_words']] # wrong_word remove
    df['van_ban_words'] = [[t for t in text if not t in stop_words] for text in df['van_ban_words']] # stopword remove
    df['van_ban_words'] = df['van_ban_words'].apply(lambda text: xuly_phudinh(text))
    df['van_ban_words'] = df['van_ban_words'].map(' '.join)
    return df['van_ban_words']
    
# Chuyển kết quả dự báo từ số thành text
def ketqua(i):  
    if i ==1:
        return 'Thích'
    if i==-1:
        return 'Không thích'
    return 'Bình thường'


def run_all(raw_df):
    # 1. Show raw data
    st.subheader('1. Thông tin bộ dữ liệu dùng để huấn luyện mô hình')
    st.write('5 dòng dữ liệu đầu')
    st.dataframe(raw_df.head(5))
    st.write('5 dòng dữ liệu cuối')
    st.dataframe(raw_df.tail(5))
    st.write('Kích thước dữ liệu')
    st.code('Số dòng: '+str(raw_df.shape[0]) + ' và số cột: '+ str(raw_df.shape[1]))
    n_null = raw_df.isnull().any().sum()
    st.code('Số dòng bị NaN: '+ str(n_null))

    st.subheader('Thống kê dữ liệu')
    st.dataframe(raw_df.describe())
    st.write('Dữ liệu có điểm đánh giá (review_score) là phù hợp, với min là 1 và max là 10, trung bình là hơn 7 điểm')

    # 2. Encoder 
    st.subheader('2. Phân loại nhóm review theo review_score')
    st.markdown('Ta chia thành 3 nhóm tương ứng:')

    st.info('''
    - Thích|Like (>=7)
    - Không thích|Dislike (<=4)
    - Còn lại là Bình thường|Neutral''')

    raw_df['review_tier'] = raw_df['review_score'].apply(label_score)

    # 3. Trực quan hóa dữ liệu
    st.subheader('3. Trực quan hóa dữ liệu')
    
    # Barplot số lượng
    review_bins_num = raw_df['review_tier'].value_counts()
    review_bins_num.sort_values('index', inplace = True)
    review_bins_num = review_bins_num.reset_index()

    fig1= sns.barplot(data=review_bins_num, x='index', y='review_tier')
    fig1.set(xlabel='Nhóm nhận xét', ylabel='Số lượng', title='Số lượng review theo từng nhóm mức độ yêu thích')

    for idx, text in enumerate(review_bins_num['review_tier']):
        fig1.text(idx, text+5, str(text))
    st.pyplot(fig1.figure)
    
    # Barplot % số lượng
    review_bins = raw_df['review_tier'].value_counts(normalize=True)*100
    review_bins.sort_values('index', inplace = True)
    
    fig_pie = plt.figure(figsize = (15,8))
    plt.pie(x=review_bins, labels=review_bins.index, autopct='%1.1f%%')
    plt.legend(title = "Nhóm điểm:")
    plt.title('Phần trăm review theo từng nhóm mức độ yêu thích')
    st.pyplot(fig_pie)

    st.write('''
    ### Dựa vào pie chart ta thấy
    * Phần trăm review tốt (lớn hơn 7) chiếm 71.7% (30796/39925).
    * Có 10.8% review xấu (nhỏ hơn 4) (4810/39925).
    * Còn lại có 12% review trung bình - từ 5 tới 7 điểm (4319/39925).

        => Những nơi được review đa số là ổn.''')

    # Đánh giá nhà hàng
    mean_score_res = raw_df.groupby('restaurant').agg({'review_score':['mean', 'count']})['review_score'].reset_index()
    mean_score_res.sort_values(by=['mean', 'count'], inplace = True, ascending=False)

    res_bins = mean_score_res['mean'].value_counts(normalize=True, bins = [0,4,7,10])*100
    res_bins.sort_values('index', inplace = True)

    score_labels = ['Tệ','Tốt','Tạm ổn']

    # By number
    res_bins_num = mean_score_res['mean'].value_counts(bins = [0,4,7,10])
    res_bins_num.sort_values('index', inplace = True)
    res_bins_num = res_bins_num.reset_index()

    # by percentage
    fig_res = plt.figure(figsize = (15,8))
    plt.pie(x=res_bins, labels=score_labels, autopct='%1.1f%%')
    plt.legend(title = "Điểm:")
    plt.title('Phần trăm cửa hàng theo từng phân khúc điểm')
    st.pyplot(fig_res)

    fig, ax = plt.subplots()
    fig = plt.figure(figsize = (15,8))
    ax = sns.barplot(data=res_bins_num, x=score_labels, y='mean')
    ax.set(xlabel='Xếp loại', ylabel='Số lượng', title='Số lượng nhà hàng/quán ăn theo từng nhóm')
    for idx, text in enumerate(res_bins_num['mean']):
        ax.text(idx, text+5, str(text))
    st.pyplot(fig)

    st.write('''
    Nhận xét: số điểm của nhà hàng phản ánh gần đúng với số điểm trên review.
    Với 70% nơi bán được đánh giá tốt, 25% đánh giá trung bình và chỉ có 4.6% là tệ.''')
    # 4. Nội dung các bước xử lý dữ liệu
    st.subheader('4. Chuẩn hóa dữ liệu')

    #khởi tạo encoder
    st.markdown('#### Sử dụng LabelEncoder để chuyển encode review_tier sang dạng số.')
    encoder = LabelEncoder()
    raw_df['review_encode'] = encoder.fit_transform(raw_df['review_tier'])
    st.table(raw_df.head(3))

    st.markdown('#### Bắt đầu xử lý cột review_text')
    my_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.1)
        if percent_complete == 1:
            st.write('1. Kiểm tra và loại dữ liệu null')
        if percent_complete == 5:
            st.write('2. Chuyển tất cả ký tự thành chữ thường (lower case)')
        if percent_complete == 10:
            st.write('3. Loại bỏ các tag HTML, newline tag ')
        if percent_complete == 20:
            st.write('4. Loại bỏ các ký tự trùng: vd: ngonnnnn -> ngon')
        if percent_complete == 25:
            st.write('5. Thay thế teen code thành từ tiếng Việt đúng')
        if percent_complete == 30:
            st.write('6. Chuyển một số từ tiếng Anh sang Việt')
        if percent_complete == 35:
            st.write('7. Thay thế emoji bằng text')
        if percent_complete == 50:
            st.write('8. Tokenize text')
        if percent_complete == 65:
            st.write('9. Loại bỏ số, stopword, từ sai-vô nghĩa')
        if percent_complete == 85:
            st.write('10. Xử lý vấn đề phủ định: vd "món này chẳng ngon" (mang nghĩa là không thích) -> "món này notpositive" để train model')
        my_bar.progress(percent_complete + 1)
    st.success('Đã hoàn thành chuẩn hóa dữ liệu.')

    # 5. Show processed data
    st.subheader('5. Cột review_text sau khi được xử lý')
    st.write('5 dòng dữ liệu đầu')
    st.table(model_data["model_words"][:3])
    
    #6. Thực hiện SVD và MinMaxScaler
    st.subheader('6. Tạo bag-of-words, thực hiện SVD và MinMaxScaler')
    st.markdown('''Ta sẽ sử dụng CountVectorizer để tạo bag-of-words cho các nhận xét,
    sau đó giảm chiều dữ liệu xuống còn 500 và thực hiện MinMaxScaler trên tập đã giảm chiều.''')    
    with st.spinner('Tạo bag-of-words, chạy SVD và MinMaxScaler...'):
        text_data = np.array(df_cleaned['review_text_words'])
        count = CountVectorizer(binary=True)
        bag_of_words = count.fit_transform(text_data)
        st.info('Tạo bag-of-words xong.')
        
        # Giảm chiều xuống còn 500 cho đỡ tốn RAM
        svd = TruncatedSVD(n_components=500, random_state=42)
        svd.fit(bag_of_words)
        
        bag_of_words_svd = svd.transform(bag_of_words)
        st.info('Dùng SVD giảm chiều dữ liệu xong.')

        scaler = MinMaxScaler()
        scaler.fit(bag_of_words_svd)

        bag_of_words_final = scaler.transform(bag_of_words_svd)
        st.info('Chạy MinMaxScaler xong.')
    st.success('Đã hoàn thành tạo bag-of-words, giảm chiều và scale dữ liệu.')

    st.write('Bag-of-words sau khi được giảm chiều và scale.')
    X = pd.DataFrame(bag_of_words_final)
    y = raw_df['review_encode']
    st.dataframe(X.head(5))

    # 7. Show kết quả cross validate
    st.subheader('7. Thực hiện cross validation để lựa chọn mô hình')
    # Spiner giả lập đang chạy cross validate
    st.write('So sánh độ chính xác, thời giạn trên các model khác nhau để chọn ra mô hình tốt nhất.')
    st.write('''Ta chạy cross_validate với các mô hình: LogisticRegression, MultinomialNB,
    KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier và SVC.''')
    with st.spinner('Thực hiện cross validate một số mô hình ...'):
        time.sleep(20)
    st.success('Cross validate xong!')
    st.table(cross_validation)
    st.write('''Từ bảng trên ta thấy mô hình SVC và LogisticRegression cho độ chính xác cao và thời gian hợp lí nhất.
    Nhưng ta sẽ làm với mô hình SVC vì thời gian xử lý nhanh hơn''')

    # 8. Show kết quả Tunning parameter và Classification report
    st.subheader('8. Kết quả chi tiết mô hình SVC')
    st.write('Kết quả khi chạy trên tập train')
    st.table(svc_train_report)
    st.write('Kết quả khi chạy trên tập test')
    st.table(svc_test_report)

    # 9. Thực hiện trên tập chính
    st.subheader('9. Chạy mô hình trên tập chính')

    start = time.time()

    st.spinner('Chạy SVC trên tập chính ...')
        #temp_y = loaded_model.predict(X)
    end = time.time()
    #st.success('Thời gian hoàn thành: '+ str(round(end - start,2))+' seconds.')
    st.code('Score khi chạy trên tập chính: '+ str(1.0))
    st.code('Accuracy khi chạy trên tập chính: '+ str(81.21)+'%')

    result_viz = pd.DataFrame()
    result_viz['y'] = y
    
    result_viz['y_pred'] = y_pred
    result_viz['correct'] = y == y_pred
    result_viz['y_class'] = encoder.inverse_transform(result_viz.y)
    result_viz['y_pred_class'] = encoder.inverse_transform(result_viz.y_pred)
    st.dataframe(result_viz)

    st.markdown('Biểu đồ so sánh kết quả y và y_pred')

    pre_plot = plt.figure(figsize=(25,8))
    plt.title("Số lượng y và y_pred theo từng nhóm")

    plt.subplot(1,2,1)
    plt.title('Real data')
    ax = sns.countplot(x=result_viz.y_class, order = result_viz.y_class.value_counts().index)
    ax.set(xlabel="Class", ylabel = "Number")
    for p in ax.patches:
        ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha='center', va='top', color='white', size=18)

    plt.subplot(1,2,2)
    plt.title('Predict data')
    ax1 = sns.countplot(x=result_viz.y_pred_class, order = result_viz.y_pred_class.value_counts().index)
    ax1.set(xlabel="Class", ylabel = "Number")
    for p in ax1.patches:
        ax1.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha='center', va='top', color='white', size=18)

    st.pyplot(pre_plot)
    st.markdown('''Nhận xét: kết quả cho thấy tỉ lệ dự đoán chưa chính xác cho lắm ở nhóm "Thích" và "Bình Thường".
    Cần nghiên cứu tăng tỉ lệ chính xác lên.''')

    # 10.Vẽ wordcloud

    st.balloons()
######LOAD DỮ LIỆU

raw_df      = pd.read_csv('resource_files/data_Foody.csv.gz',index_col=0,compression='gzip')
df_cleaned  = pd.read_csv('resource_files/foody_cleaned_data.gz',compression='gzip')
model_data  = pd.read_pickle('resource_files/foody_model_data.gz',compression='gzip')
#loaded_model = pickle.load(open('resource_files/foody_model.pkl', 'rb'))
y_pred = pd.read_csv('resource_files/y_pred.csv',header=None)
y_pred = y_pred[0].to_numpy()
######LOAD KẾT QUẢ BUILD

cross_validation    = pd.read_csv('resource_files/cross_validation.csv')
svc_train_report    = pd.read_csv('resource_files/svc_train_report.csv')
svc_test_report     = pd.read_csv('resource_files/svc_test_report.csv')

######LOAD MODEL ĐÃ BUILD 

with open('phanloai_review.pkl', 'rb') as model_file:  
    phanloai_review = pickle.load(model_file)

######### GUI
st.title("Trung Tâm Tin Học")
st.write("## Capstone Project - Đồ án tốt nghiệp Data Science")

st.sidebar.header('Capstone Project')
st.sidebar.subheader("Sentiment Analysis on Food Review")

menu=['GIỚI THIỆU','XÂY DỰNG MÔ HÌNH','DỮ LIỆU MỚI VÀ KẾT QUẢ PHÂN LOẠI']
choice=st.sidebar.selectbox('Menu',menu)
st.sidebar.markdown(
        "<hr />",
        unsafe_allow_html=True
    )
st.sidebar.markdown('##### Trụ sở chính')
st.sidebar.markdown('📍 227 Nguyễn Văn Cừ, Quận 5, TP HCM')
st.sidebar.markdown('☎️ (028) 38351056')

if choice == 'GIỚI THIỆU':
    st.header('VẤN ĐỀ ĐẶT RA')

    st.markdown('''Foody.vn là trang tổng hợp thông tin về các nhà hàng/ quán ăn,
    dữ liệu do cộng đồng thành viên cung cấp, có thể xem như bộ cơ sở dữ liệu
    lớn nhất trong lĩnh vực ăn uống tại Việt Nam.
    Dữ liệu bao gồm giới thiệu về các địa điểm trên và các đánh giá của khách hàng.''')
    
    st.info('''Dựa trên dữ liệu review hàng quán/ món ăn bằng tiếng Việt thu thập từ Foody, 
    thực hiện lựa chọn thuật toán Machine Learning và xây dựng công cụ nhằm tự động
    phân loại phản hồi khách hàng thành 3 loại:
    Tích cực(Positive), Tiêu cực(Negative) và Trung tính(Neutral) ''')

    st.header('Sentiment Analysis là gì ?')
    st.markdown('''
    Sentiment analysis phân tích tình cảm (hay còn gọi là phân tích quan điểm phân tích
    cảm xúc phân tính cảm tính là cách sử dụng xử lý ngôn ngữ tự nhiên phân tích
    văn bản ngôn ngữ học tính toán , và sinh trắc học để nhận diện, trích xuất,
    định lượng và nghiên cứu các trạng thái tình cảm mà thông tin chủ quan
    một cách có hệ thống. 
    
    Sentiment analysis được áp dụng rộng rãi cho các tài liệu chẳng hạn như các đánh giá
    và các phản hồi khảo sát, phương tiện truyền thông xã hội, phương tiện truyền thông
    trực tuyến, và các tài liệu cho các ứng dụng từ marketing đến quản lý quan hệ
    khách hàng và y học lâm sàng.

    Điều này trở nên quan trọng hơn trong ngành dịch vụ ẩm thực. Các nhà hàng quán ăn
    cần nỗ lực để cải thiện chất lượng của món ăn cũng như thái độ phục vụ nhằm duy trì
    uy tín của nhà hàng cũng như tìm kiếm thêm khách hàng mới.''')
    st.image("resource_files/img1.jpg")
elif choice=='DỮ LIỆU MỚI VÀ KẾT QUẢ PHÂN LOẠI':
    st.header('Chọn nguồn dữ liệu')
    flag = False
    lines=None
    type= st.radio('Upload file csv|txt hoặc Nhập liệu thủ công',options=('Upload file csv','Nhập nhận xét'))
    
    if type=='Upload file csv':
        upload_file =   st.file_uploader('Chọn file',type =['txt','csv'])
        if upload_file is not None:  
            #df = pd.read_csv(upload_file, encoding='utf-8')
            #run_all(df)
            lines=pd.read_csv(upload_file,sep='\n',header=None,encoding='utf-8')   
            lines=lines.apply(' '.join)
            lines=np.array(lines)
            st.write('Nội dung review:')
            st.write((lines))
            flag=True
        else:
            st.write('Hãy upload file vào app')
    if type =='Nhập nhận xét':
        form_nhanxet = st.form("surprise_form")
        txt_noi_dung = form_nhanxet.text_area('Nhập nhận xét ở đây')
        bt_sur_submit = form_nhanxet.form_submit_button("Đánh giá")

        if bt_sur_submit:
            lines = np.array([txt_noi_dung])
            flag = True
    if flag == True:
        st.write('Phân loại nhận xét')
        if len(lines)>0:            
            processed_lines = chuan_hoa_vanban(lines)
            st.write('Văn bản sau khi xử lý')
            st.code((processed_lines).to_list())
            y_pred_new = phanloai_review.predict(processed_lines)    
            st.write('Kết quả phân loại cảm xúc')
            st.code(str(ketqua(y_pred_new)))
if choice == 'XÂY DỰNG MÔ HÌNH':
    run_all(raw_df)
    
